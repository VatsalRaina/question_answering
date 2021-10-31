#! /usr/bin/env python

"""
Evaluate a simple question-answering model for extractive reading comprehension
"""

import os
import sys
import time
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from qadatasets import load_squad
from uncertainty import multiheadattention
from utils import get_args, load_model
from utils import get_default_device, format_time

# Get all arguments for training
args = get_args().parse_args()


@torch.no_grad()
def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/test.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    # Load tokenizer
    tokenizer = load_model(args, tokenizer_only=True)

    # Load trained model
    model = torch.load(args.model_path, map_location = device)
    model.eval().to(device)

    # Load validation dataset
    all_data = load_squad(
        args = args,
        tokenizer = tokenizer,
        device = device,
        split = 'validation'
    )

    input_ids = all_data['input_ids']
    token_type_ids = all_data['token_type_ids']
    attention_masks = all_data['attention_masks']

    # Create the DataLoader for validation set.
    eval_data = TensorDataset(input_ids, token_type_ids, attention_masks)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    # Store logit predictions
    pred_start_logits = []
    pred_end_logits = []

    # Store uncertainties
    pred_attention_uncertainties = {}

    # Measure how long the evaluation takes.
    t0 = time.time()

    for step, (b_input_ids, b_tok_typ_ids, b_att_msks) in enumerate(eval_dataloader, start = 1):

        outputs = model(
            input_ids=b_input_ids.to(device),
            attention_mask=b_att_msks.to(device),
            token_type_ids=b_tok_typ_ids.to(device),
            output_attentions=args.attention_uncertainty
        )

        if args.attention_uncertainty and args.dataset == 'squad_v2':

            # This will be a tuple of attention predictions from all layers
            attentions = outputs[-1]

            # Extract the attentions for the last layer
            # attentions = attentions[-1]
            # estimator = multiheadattention(last=True)

            # Extract the attentions for the first layer
            attentions = attentions[0]
            estimator = multiheadattention(last=False)

            # Initialise estimator and get the uncertainties
            uncertainties = estimator(args, attentions, b_input_ids)

            # Set uncertainties for later processing
            for unc_name in uncertainties:
                if unc_name not in pred_attention_uncertainties:
                    pred_attention_uncertainties[unc_name] = []
                pred_attention_uncertainties[unc_name] += uncertainties[unc_name].detach().cpu().numpy().tolist()

        # b_start_logits = outputs.start_logits
        # b_end_logits   = outputs.end_logits
        b_start_logits = outputs[0]
        b_end_logits = outputs[1]

        # Store all predictions
        pred_start_logits += b_start_logits.detach().cpu().numpy().tolist()
        pred_end_logits += b_end_logits.detach().cpu().numpy().tolist()

        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)

        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(eval_dataloader), elapsed))

    # Convert to numpy
    pred_start_logits, pred_end_logits = np.array(pred_start_logits), np.array(pred_end_logits)

    filename = "" if args.dataset == 'squad' else "_" + args.dataset
    np.save(os.path.join(args.predictions_save_path, "pred_start_logits{}.npy".format(filename)), pred_start_logits)
    np.save(os.path.join(args.predictions_save_path, "pred_end_logits{}.npy".format(filename)), pred_end_logits)

    if args.attention_uncertainty and args.dataset == 'squad_v2':
        # Save all uncertainties in separate files
        for unc_name, uncs in pred_attention_uncertainties.items():

            # Save to its own numpy file
            np.save(os.path.join(args.predictions_save_path, "pred_{}.npy".format(unc_name)), np.array(uncs))


if __name__ == '__main__':
    main(args)