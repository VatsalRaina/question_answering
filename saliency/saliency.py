#! /usr/bin/env python

"""
Get average saliency values for word embedding by back-propagating the sum of the start logits
"""

import os
import sys
import time
import numpy as np

dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.append(dirname+'/..')

import torch
from torch.utils.data import TensorDataset, DataLoader

from qadatasets import load_squad
from uncertainty import multiheadattention
from utils import get_args, load_model
from utils import get_default_device, format_time

# Get all arguments for training
args = get_args().parse_args()

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
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False)

    # Store logit predictions
    pred_start_grads = []
    pred_end_grads = []

    # Measure how long the evaluation takes.
    t0 = time.time()

    for step, (b_input_ids, b_tok_typ_ids, b_att_msks) in enumerate(eval_dataloader, start = 1):

        if step==10:
            break

        embedding_matrix = model.electra.embeddings.word_embeddings
        b_inputs_embeds = torch.tensor(embedding_matrix(b_input_ids.to(device)), requires_grad=True)

        outputs = model(
            inputs_embeds=b_inputs_embeds.to(device),
            attention_mask=b_att_msks.to(device),
            token_type_ids=b_tok_typ_ids.to(device)
        )

        # b_start_logits = outputs.start_logits
        # b_end_logits   = outputs.end_logits
        b_start_logits = outputs[0]
        b_end_logits = outputs[1]

        # Sum the start logits
        start_sum = torch.sum(torch.squeeze(b_start_logits))

        # Back-propagate to calculate gradients that influence the start position
        start_sum.backward()

        start_saliency = torch.norm(b_inputs_embeds.grad.data.abs(), dim=1)
        # Store all predictions
        pred_start_grads += start_saliency.detach().cpu().numpy().tolist()

        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)

        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(eval_dataloader), elapsed))

    # Convert to numpy
    pred_start_grads, pred_end_grads = np.array(pred_start_grads), np.array(pred_end_grads)

    filename = "" if args.dataset == 'squad' else "_" + args.dataset
    np.save(os.path.join(args.predictions_save_path, "pred_saliency_start_grads{}.npy".format(filename)), pred_start_grads)
    np.save(os.path.join(args.predictions_save_path, "pred_saliency_end_grads{}.npy".format(filename)), pred_end_grads)


if __name__ == '__main__':
    main(args)