#! /usr/bin/env python

"""
Prepare predictions for official SQuAD evaluation
"""
# TODO There is a fair amount of code repetition here which can probably be refactored for readability
# TODO: Split the code up in this file into smaller more readable functions and add more comments

import os
import sys

from utils.parser import get_args_prep

from datasets import load_dataset
from transformers import ElectraTokenizer, ElectraConfig
import json
import collections
import numpy as np
from scipy.stats import entropy

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


def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/postprocess.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    all_start_logits, all_end_logits = [], []

    for f_s, f_e in zip(os.listdir(args.load_start_dir), os.listdir(args.load_end_dir)):
        assert f_s == f_e
        if f_s.endswith(".npy"):
            all_start_logits.append(np.load(os.path.join(args.load_start_dir, f_s)))
            all_end_logits.append(np.load(os.path.join(args.load_start_dir, f_e)))

    all_start_logits = np.asarray(all_start_logits)
    all_end_logits = np.asarray(all_end_logits)

    # logits will have dimension (ens, dataset, maxlen)

    # Ensemble start and end logits
    ens_start_logits = np.mean(all_start_logits, axis=0 )
    ens_end_logits = np.mean(all_end_logits, axis=0 )

    dev_data = load_dataset(args.dataset, split='validation')
    
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator', do_lower_case=True)

    # Find predictions as word spans
    span_predictions = {}

    for i, ex in enumerate(dev_data):
        #print(i)
        start_logits = ens_start_logits[i, :]
        end_logits = ens_end_logits[i, :]
        question, passage, qid = ex["question"], ex["context"].replace("\n", ""), ex["id"]

        combo = question + "[SEP]" + passage
        input_ids = tokenizer.encode(combo)
        #input_ids = input_ids[:512]

        # sort our start and end logits from largest to smallest, keeping track of the index
        start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
        end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

        # Select top 10 indexes
        start_indexes = [idx for idx, logit in start_idx_and_logit[:10]]
        end_indexes = [idx for idx, logit in end_idx_and_logit[:10]]

        # question tokens are defined as those between the CLS token (101, at position 0) and first SEP (102) token
        question_indexes = list(range(1, input_ids.index(102) + 1))

        # keep track of all preliminary predictions
        PrelimPrediction = collections.namedtuple("PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"])

        prelim_preds = []
        prelim_preds.append(
                PrelimPrediction(
                    start_index = 0,
                    end_index = 0,
                    start_logit = start_logits[0],
                    end_logit = end_logits[0]
                )
            )
        for start_index in start_indexes:
            if start_index in question_indexes:
                continue
            # Ignore [CLS]
            if start_index==0:
                continue
            for end_index in end_indexes:
                # throw out invalid predictions
                if end_index in question_indexes:
                    continue
                if end_index > len(input_ids):
                    continue
                if end_index < start_index:
                    continue

                # Reject if the answer span is too long
                #if (end_index - start_index) > 10:
                #    continue
                prelim_preds.append(
                    PrelimPrediction(
                        start_index = start_index,
                        end_index = end_index,
                        start_logit = start_logits[start_index],
                        end_logit = end_logits[end_index]
                    )
                )

        # sort preliminary predictions by their score
        prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        best = prelim_preds[0]
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[best.start_index:best.end_index+1]))

        # Check for unanswerability
        if args.squad_version == 2:
            #TODO different uncertainty measures need to be considered

            # From first occurence of the SEP token to the last occurence of the SEP token
            context_start_logits = start_logits[input_ids.index(102) + 1  :  -1 * (input_ids[::-1].index(102) + 1) ]
            context_end_logits = end_logits[input_ids.index(102) + 1  :  -1 * (input_ids[::-1].index(102) + 1) ]

            # Apply softmax
            start_probs = np.exp(context_start_logits) / sum(np.exp(context_start_logits))
            end_probs = np.exp(context_end_logits) / sum(np.exp(context_end_logits))

            start_entropy = entropy(start_probs, base=2)
            end_entropy = entropy(end_probs, base=2)
            avg_entropy = (start_entropy+end_entropy)/2
            if avg_entropy > args.threshold:
                answer = ""


        # The answer after detokenizing often doesn't even end up being an extract from the context
        # due to spacing around various characters e.g. punctuation
        # Hence, it is necessary to match the answer to an extract from the context
        if answer!="":
            noSpaceToWithSpace = []
            for pos, char in enumerate(passage):
                if char!=' ':
                    noSpaceToWithSpace.append(pos)
            passage_noSpace = strip_accents(passage.lower().replace(" ", ""))
            answer_noSpace = strip_accents(answer.replace(" ", ""))
            if answer_noSpace in passage_noSpace:
                start_char_noSpace = passage_noSpace.find(answer_noSpace)
                end_char_noSpace = start_char_noSpace + len(answer_noSpace) - 1
                start_char = noSpaceToWithSpace[start_char_noSpace]
                end_char = noSpaceToWithSpace[end_char_noSpace]
                answer = passage[start_char:end_char+1]
            # else:
            #     # Couldn't find the answer as an extract of the context
            #     print(passage.lower())
            #     print(answer)
            #     print(' ')

        span_predictions[qid] = answer

    with open(args.save_dir+'predictions.json', 'w') as fp:
        json.dump(span_predictions, fp)

if __name__ == '__main__':
    main(args)