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

from utils import get_args, load_model
from utils import get_default_device, format_time
from qadatasets import load_squad_v1

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
    all_data = load_squad_v1(
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

    # Measure how long the evaluation takes.
    t0 = time.time()

    for step, (b_input_ids, b_tok_typ_ids, b_att_msks) in enumerate(eval_dataloader, start = 1):

        outputs = model(
            input_ids=b_input_ids.to(device),
            attention_mask=b_att_msks.to(device),
            token_type_ids=b_tok_typ_ids.to(device),
            # return_dict=True
        )

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
    
    pred_start_logits, pred_end_logits = np.asarray(pred_start_logits), np.asarray(pred_end_logits)
    np.save(args.predictions_save_path + "pred_start_logits.npy", pred_start_logits)
    np.save(args.predictions_save_path + "pred_end_logits.npy", pred_end_logits)


if __name__ == '__main__':
    main(args)