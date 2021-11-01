#! /usr/bin/env python

"""
Generate figures to explore potential length bias in answerable
and unanswerable examples.
"""

import os
import sys

dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.append(dirname+'/../..')

import numpy as np
import scipy as sp

from datasets import load_dataset
from transformers import ElectraTokenizer
from utils import get_args_prep

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from uncertainty.logits import ensemblelogits

# Get all arguments for postprocessing
args = get_args_prep().parse_args()

def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/gen_figures.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    info = {'num_tokens_unanswerable': [], 'num_tokens_answerable': []}
    all_lengths = []


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

    dev_data = load_dataset(args.dataset, split='validation')
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator', do_lower_case=True)

    # Get all uncertainties
    unc_predictions = {}

    for i, ex in enumerate(dev_data):

        # Get the question, context and id
        question, context= ex["question"], ex["context"]

        if len(ex["answers"]["text"]) == 0:
            answerable = True
        else:
            answerable = False

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

        truncated_context_ids = input_ids[first_sep_idx:-last_sep_idx]

        num_ids = len(truncated_context_ids)

        if answerable:
            info['num_tokens_answerable'].append(num_ids)
        else:
            info['num_tokens_unanswerable'].append(num_ids)
        all_lengths.append(num_ids)


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
    
    # Create pandas dataframes

    l_num_tokens = info['num_tokens_unanswerable'] + info['num_tokens_answerable']
    l_type = ['Unanswerable'] * len(info['num_tokens_unanswerable']) + ['Answerable'] * len(info['num_tokens_answerable'])
    df = pd.DataFrame(list(zip(l_num_tokens, l_type)), columns=['Length', 'Type'])
    # sns.set_context("poster")
    sns.histplot(data=df, x="Length", hue="Type", multiple="stack")
    plt.savefig(args.save_dir + 'lengths_true_frac.png')
    plt.clf()

    # for unc_name, uncs in unc_predictions.items():

    #     info_pred = {'num_tokens_unanswerable': [], 'num_tokens_answerable': []}

    #     # According to threshold fraction convert
    #     threshold = np.array(list(uncs))
    #     # Force threshold at 0.5 to mimic evaluation set
    #     threshold = np.quantile(threshold, 0.5)

    #     # Now any uncertainty exceeding this threshold will have its answer set to nan
    #     for k, unc in enumerate(uncs):
    #         if unc > threshold:
    #             info_pred['num_tokens_unanswerable'].append(all_lengths[k])
    #         else:
    #             info_pred['num_tokens_answerable'].append(all_lengths[k])

    #     l_num_tokens = info_pred['num_tokens_unanswerable'] + info_pred['num_tokens_answerable']
    #     l_type = ['Unanswerable'] * len(info_pred['num_tokens_unanswerable']) + ['Answerable'] * len(info_pred['num_tokens_answerable'])
    #     df = pd.DataFrame(list(zip(l_num_tokens, l_type)), columns=['Length', 'Type'])
    #     # sns.set_context("poster")
    #     sns.histplot(data=df, x="Length", hue="Type", multiple="stack")
    #     plt.savefig(args.save_dir + unc_name + '_lengths_pred.png')
    #     plt.clf()


if __name__ == '__main__':
    main(args)