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
from uncertainty.logits import load_class as load_uncertainty_class

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# Get all arguments for postprocessing
args = get_args_prep().parse_args()

def ood_detection(domain_labels, measures, mode, rev):

    # Convert to high bit float
    measures = np.asarray(measures, dtype=np.float128)

    if rev: measures *= -1.0

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, measures)
        aupr = auc(recall, precision)

        # Find the best threshold for F1 score
        best_result = find_best_f1(precision, recall, thresholds)

        return aupr, best_result

    elif mode == 'ROC':
        # fpr, tpr, thresholds = roc_curve(domain_labels, measures)
        auroc = roc_auc_score(domain_labels, measures)
        return auroc


def find_best_f1(precision, recall, threshold):

    # Calculate F1 score
    f_score = (2 * precision * recall) / (precision + recall)

    # Find all nan positions
    nan_pos = np.squeeze(np.argwhere(np.isnan(f_score)))

    # Remove them from the f1 score
    f_score_masked = np.delete(f_score, nan_pos, None)

    # Find the best score
    best = np.amax(f_score_masked)

    # Find the position of best score
    pos = np.where(f_score == best)

    return precision[pos], recall[pos], threshold[pos], best

def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/postprocess_saliency.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Number of models
    n = len(args.load_dirs)

    # Save all predictions within this dictionary
    grad_predictions = {'start': []}

    # Iterate over all directory paths
    for path in args.load_dirs:
        filename = "" if args.dataset == 'squad' else "_" + args.dataset

        # Load start grads
        file = os.path.join(path, "pred_saliency_start_grads{}.npy".format(filename))
        grad_predictions['start'].append(np.load(file))

    for key, item in grad_predictions.items():
        grad_predictions[key] = np.array(item)

    grad_predictions['ensemble_start'] = sp.special.logsumexp(grad_predictions['start'], axis = 0) - np.log(n)

    # Loading dev data
    dev_data = load_dataset(args.dataset, split='validation')
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator', do_lower_case=True)

    # Get all uncertainties
    unc_predictions = {}

    # Store unanswerability labels for the detection task
    unans_labels = []

    # Store qids to ensure ordering
    qids = []

    for i, ex in enumerate(dev_data):

        # Get the unanswerability label
        unans_labels.append(1 if len(ex["answers"]["text"]) == 0 else 0)

        # Store qid for uncertainties later on
        qids.append(ex["id"])

        # Process each example separately
        start_logits = grad_predictions['ensemble_start'][i]
        end_logits = grad_predictions['ensemble_end'][i]

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

        # Check for unanswerability
        if args.dataset == "squad_v2":

            # Get the logits for all models in the ensemble (num models, seqlen)
            all_start_logits = grad_predictions['start'][:, i, first_sep_idx:-last_sep_idx]
            all_end_logits = grad_predictions['end'][:, i, first_sep_idx:-last_sep_idx]

            # Initialise estimator and get the uncertainties
            estimator = load_uncertainty_class(args)
            uncertainties = estimator(args, all_start_logits, all_end_logits)

            # Set uncertainties for later processing
            for unc_name in uncertainties:
                if unc_name not in unc_predictions:
                    unc_predictions[unc_name] = {}
                unc_predictions[unc_name][qid] = uncertainties[unc_name]


    # Get detection performance for each uncertainty
    domain_labels = np.array(unans_labels)

    for unc_name, uncs in unc_predictions.items():

        # Get the uncertainty measures
        measures = np.array([unc_predictions[unc_name][qid] for qid in qids])

        # Get the aupr and best threshold performance
        aupr, [pr, re, th, f1] = ood_detection(domain_labels, measures, mode='PR', rev = False)

        # Get the auroc
        auroc = ood_detection(domain_labels, measures, mode='ROC', rev = False)

        print("\n\nDetection of Unanswerability")
        print(unc_name)
        print("Precision:", pr)
        print("Recall:   ", re)
        print("Threshold:", th)
        print("F1:       ", f1)
        print("AUROC:    ", auroc)
        print("AUPR:     ", aupr)

if __name__ == '__main__':
    main(args)