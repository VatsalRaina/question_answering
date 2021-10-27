#! /usr/bin/env python

"""
Prepare predictions for official SQuAD evaluation
"""
# TODO There is a fair amount of code repetition here which can probably be refactored for readability
# TODO: Split the code up in this file into smaller more readable functions and add more comments

import os
import sys
import json
import collections

dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.append(dirname+'/..')

import numpy as np
import scipy as sp
import copy as c

from datasets import load_dataset
from transformers import ElectraTokenizer, ElectraConfig
from utils import get_args_prep
from uncertainty.logits import ensemblelogits

# Get all arguments for postprocessing
args = get_args_prep().parse_args()


def strip_accents(text):
    text = text.replace("ö", "o").replace("ü", "u").replace("á", "a").replace("é", "e").replace("í", "i")
    text = text.replace("ó", "o").replace("ú", "u").replace("ñ", "n").replace("ç", "c").replace("â", "a")
    text = text.replace("ê", "e").replace("î", "i").replace("ô", "o").replace("û", "u").replace("à", "a")
    text = text.replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u").replace("ë", "e")
    text = text.replace("ï", "i").replace("ä", "a").replace("ć", "c").replace("ń", "n").replace("ś", "s")
    text = text.replace("ź", "z").replace("ł", "l").replace("ż", "z").replace("ą", "a").replace("ę", "e")
    text = text.replace("š", "s").replace("ř", "r").replace("ů", "u").replace("č", "c").replace("ě", "e")
    text = text.replace("ž", "z").replace("ý", "y").replace("ā", "a").replace("ē", "e").replace("ī", "i")
    text = text.replace("ō", "o").replace("ū", "u").replace("õ", "o").replace("\u1ea1", "a").replace("\u1eb1", "a")
    text = text.replace("\u1ec7", "e").replace("å", "a")
    return text


def clean_answer(answer, context):
    if answer != "":
        noSpaceToWithSpace = []
        for pos, char in enumerate(context):
            if char != ' ':
                noSpaceToWithSpace.append(pos)
        passage_noSpace = strip_accents(context.lower().replace(" ", ""))
        answer_noSpace = strip_accents(answer.replace(" ", ""))
        if answer_noSpace in passage_noSpace:
            start_char_noSpace = passage_noSpace.find(answer_noSpace)
            end_char_noSpace = start_char_noSpace + len(answer_noSpace) - 1
            start_char = noSpaceToWithSpace[start_char_noSpace]
            end_char = noSpaceToWithSpace[end_char_noSpace]
            answer = context[start_char:end_char + 1]
    return answer


def get_best_indices(start_indices, end_indices, start_logits, end_logits):

    # Keep track of all preliminary predictions
    PredictionInfo = collections.namedtuple("PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"])

    prediction_info = [PredictionInfo(
        start_index = 0, end_index = 0,
        start_logit = start_logits[0],
        end_logit = end_logits[0]
    )]

    for start_index in start_indices:
        for end_index in end_indices:
            # Throw out invalid predictions
            if end_index < start_index: continue

            # Append valid predictions
            prediction_info.append(PredictionInfo(
                start_index=start_index, end_index=end_index,
                start_logit=start_logits[start_index],
                end_logit=end_logits[end_index]
            ))

    # Sort preliminary predictions by their score
    prediction_info = sorted(prediction_info, key = lambda x: (x.start_logit + x.end_logit), reverse = True)

    # Return the prediction with the jointly highest score
    return prediction_info[0]


def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/postprocess.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Number of models
    n = len(args.load_dirs)

    # Save all predictions within this dictionary
    logit_predictions = {'start': [], 'end': []}

    # Iterate over all directory paths
    for path in args.load_dirs:

        # Load start logits
        file = os.path.join(path, "pred_start_logits.npy")
        logit_predictions['start'].append(np.load(file))

        # Load end logits
        file = os.path.join(path, "pred_end_logits.npy")
        logit_predictions['end'].append(np.load(file))

    # Convert into numpy arrays
    # The logits will have dimension (num models, dataset size, maxlen)
    for key, item in logit_predictions.items():
        logit_predictions[key] = np.array(item)

    # Ensemble start and end logits
    logit_predictions['ensemble_start'] = sp.special.logsumexp(logit_predictions['start'], axis = 0) - np.log(n)
    logit_predictions['ensemble_end'] = sp.special.logsumexp(logit_predictions['end'], axis=0) - np.log(n)

    # Loading dev data
    dev_data = load_dataset(args.dataset, split='validation')

    # TODO: This should be specified based on what model arch we are using, see test script
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator', do_lower_case=True)

    # Find predictions as word spans
    span_predictions = {}

    # Get all uncertainties
    unc_predictions = {}

    for i, ex in enumerate(dev_data):

        # Process each example separately
        start_logits = logit_predictions['ensemble_start'][i]
        end_logits = logit_predictions['ensemble_end'][i]

        # Get the question, context and id
        question, context, qid = ex["question"], ex["context"], ex["id"]

        # The input to the transformer model is based on concatenating question and context
        # this is serparated by a special SEP token
        concatenation = question + " [SEP] " + context

        # Tokenizer converts input to be compatible with chosen model
        # Note we truncate inputs exceesding max length
        input_encodings_dict = tokenizer(concatenation, truncation=True, max_length=args.max_len, padding="max_length")

        # Extract necessary pieces from tokenizer output, sequence of token ids
        input_ids = input_encodings_dict['input_ids']

        # Find occurrence of SEP tokens to isolate the context
        first_sep_idx = input_ids.index(102) + 1
        last_sep_idx  = input_ids[::-1].index(102) + 1

        # From first occurrence of the SEP token to the last occurence of the SEP token
        context_start_logits = start_logits[first_sep_idx:-last_sep_idx]
        context_end_logits = end_logits[first_sep_idx:-last_sep_idx]
        context_length = len(context_start_logits)

        # Tokenize contest and remove special tokens
        context_ids = tokenizer.encode(context)[1:-1]

        # Sort our start and end logits from largest to smallest, keeping track of the index
        start_idx_and_logit = sorted(enumerate(context_start_logits), key=lambda x: x[1], reverse=True)
        end_idx_and_logit = sorted(enumerate(context_end_logits), key=lambda x: x[1], reverse=True)

        # Select top 10 indexes for computational efficiency
        start_indices = [idx for idx, _ in start_idx_and_logit[:10]]
        end_indices = [idx for idx, _ in end_idx_and_logit[:10]]

        # Get best predictions
        best = get_best_indices(start_indices, end_indices, context_start_logits, context_end_logits)

        # Extract answer from context
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                context_ids[best.start_index:best.end_index + 1]
            )
        )

        # Check for unanswerability
        if args.dataset == "squad_v2":

            # Get the logits for all models in the ensemble (num models, seqlen)
            all_start_logits = logit_predictions['start'][:, i, first_sep_idx:-last_sep_idx]
            all_end_logits = logit_predictions['end'][:, i, first_sep_idx:-last_sep_idx]

            # Initialise estimator and get the uncertainties
            estimator = ensemblelogits()
            uncertainties = estimator(args, all_start_logits, all_end_logits)

            # Set uncertainties for later processing
            for unc_name in uncertainties:
                if unc_name not in unc_predictions:
                    unc_predictions[unc_name] = {}
                unc_predictions[unc_name][qid] = uncertainties[unc_name]

        # The answer after detokenizing often doesn't even end up being an extract from the context
        # due to spacing around various characters e.g. punctuation
        # Hence, it is necessary to match the answer to an extract from the context
        span_predictions[qid] = clean_answer(answer, context)

    if args.dataset != "squad_v2":
        with open(os.path.join(args.save_dir, 'predictions.json'), 'w') as fp:
            json.dump(span_predictions, fp)
        return

    # Now process the uncertainties and set certain answer
    for unc_name, uncs in unc_predictions.items():

        # Copy the span predictions
        unc_span_predictions = c.deepcopy(span_predictions)

        # According to threshold fraction convert
        threshold = np.array(list(uncs.values()))
        threshold = np.quantile(threshold, 1 - args.threshold_frac)

        # Now any uncertainty exceeding this threshold will have its answer set to nan
        for qid, answer in unc_span_predictions.items():

            # If the uncertainty exceeds the threshold then set the answer to ""
            unc_span_predictions[qid] = "" if uncs[qid] > threshold else answer

        with open(os.path.join(args.save_dir, unc_name + '_predictions.json'), 'w') as fp:
            json.dump(unc_span_predictions, fp)


if __name__ == '__main__':
    main(args)