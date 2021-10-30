#! /usr/bin/env python

"""
Calculate F1 scores for detecting unanswerable examples
"""

import os
import sys

dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.append(dirname+'/..')

import numpy as np
import scipy as sp
import copy as c

from datasets import load_dataset
from transformers import ElectraTokenizer
from utils import get_args_prep
from uncertainty.logits import ensemblelogits

from sklearn.metrics import precision_recall_curve

# Get all arguments for postprocessing
args = get_args_prep().parse_args()


def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/unanswerability.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Number of models
    n = len(args.load_dirs)

    # Save all predictions within this dictionary
    logit_predictions = {'start': [], 'end': []}

    # Iterate over all directory paths
    for path in args.load_dirs:
        filename = "" if args.dataset == 'squad' else "_" + args.dataset

        # Load start logits
        file = os.path.join(path, "pred_start_logits{}.npy".format(filename))
        logit_predictions['start'].append(np.load(file))

        # Load end logits
        file = os.path.join(path, "pred_end_logits{}.npy".format(filename))
        logit_predictions['end'].append(np.load(file))

    # Convert into numpy arrays
    # The logits will have dimension (num models, dataset size, maxlen)
    for key, item in logit_predictions.items():
        logit_predictions[key] = np.array(item)

    # Loading dev data
    dev_data = load_dataset(args.dataset, split='validation')

    # TODO: This should be specified based on what model arch we are using, see test script
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator', do_lower_case=True)

    # Get all uncertainties
    unc_predictions = {}

    # Get binary labels for unanswerability
    unans_labels = []

    for i, ex in enumerate(dev_data):

        # Get the question, context and id
        question, context = ex["question"], ex["context"]

        if len(ex["answers"]["text"]) == 0:
            unans_labels.append(1)
        else:
            unans_labels.append(0)

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
        # Get the logits for all models in the ensemble (num models, seqlen)
        all_start_logits = logit_predictions['start'][:, i, first_sep_idx:-last_sep_idx]
        all_end_logits = logit_predictions['end'][:, i, first_sep_idx:-last_sep_idx]

        # Initialise estimator and get the uncertainties
        estimator = ensemblelogits()
        uncertainties = estimator(args, all_start_logits, all_end_logits)

        # Set uncertainties for later processing
        for unc_name in uncertainties:
            if unc_name not in unc_predictions:
                unc_predictions[unc_name] = []
            unc_predictions[unc_name].append(uncertainties[unc_name])

    # Get precision-recall curve for each uncertainty measure

    y_true = np.asarray(unans_labels)

    for unc_name in unc_predictions.keys():
        y_pred = np.asarray(unc_predictions[unc_name])

        precision, recall, threshold = precision_recall_curve(y_true, 1. / (1 + np.exp(-1 * y_pred)) )

        f_score = (2 * precision * recall) / ( precision + recall)
        nan_pos = np.squeeze(np.argwhere(np.isnan(f_score)))
        f_score = np.delete(f_score, nan_pos, None)
        best = np.amax(f_score)

        pos = np.where( (2 * precision * recall) / ( precision + recall) ==best)
        precision = precision[pos]
        print()
        print(unc_name)
        print("precision", precision)
        recall = recall[pos]
        print("recall", recall)
        threshold = threshold[pos]
        print("threshold", threshold)
        print("F1", best)

if __name__ == '__main__':
    main(args)