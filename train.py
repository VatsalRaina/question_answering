#! /usr/bin/env python

"""
Train a simple question-answering model for extractive reading comprehension
"""

import os
import sys
import time

import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)

from transformers import ElectraTokenizer
from transformers import ElectraForQuestionAnswering, AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

from utils import get_args, load_model
from utils import format_time, get_default_device, set_seed
from qadatasets import load_squad_v1

# Get all arguments for training
args = get_args().parse_args()


def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value to make this reproducible.
    set_seed(args)
    print("===> Using random seed {}".format(args.seed))

    # Choose device
    device = get_default_device()
    print("===> Using device {}".format(device))

    # Set model and load tokenizer
    print("===> Loading model")
    model, tokenizer = load_model(args)
    model.to(device)

    # Set model optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # Load SQuAD v1 dataset
    # TODO: Move this to a different location to make this script dataset agnostic
    print("===> Loading dataset")
    all_data = load_squad_v1(
            args = args,
            tokenizer = tokenizer,
            device = device,
            split = 'train'
    )
    print("===> Finished loading dataset")

    # TODO: Make data processing script model agnostic
    input_ids = all_data['input_ids']
    start_positions_true = all_data['start_positions_true']
    end_positions_true = all_data['end_positions_true']
    token_type_ids = all_data['token_type_ids']
    attention_masks = all_data['attention_masks']

    # Create the DataLoader for training set.
    train_data = TensorDataset(input_ids, start_positions_true, end_positions_true, token_type_ids, attention_masks)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = args.warmup_steps,
        num_training_steps = total_steps,
    )

    # Training loop
    for epoch in range(args.epochs):
        
        # Perform one full pass over the training set.
        print('\n======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Set model into training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:

                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Get batch of inputs, specific to transformers
            b_input_ids = batch[0].to(device)
            b_start_pos_true = batch[1].to(device)
            b_end_pos_true = batch[2].to(device)
            b_tok_typ_ids = batch[3].to(device)
            b_att_msks = batch[4].to(device)

            model.zero_grad()

            # This will automatically get loss and predictions
            # TODO: Might need to be changed for flexible choice in loss function
            outputs = model(
                input_ids = b_input_ids,
                attention_mask = b_att_msks,
                token_type_ids = b_tok_typ_ids,
                start_positions = b_start_pos_true,
                end_positions = b_end_pos_true
            )

            # First part of output is the loss
            loss = outputs[0]

            # Track the epoch loss
            total_loss += loss.item()

            # Gradient back prop
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("===> Average training loss: {0:.2f}".format(avg_train_loss))
        print("===> Training epoch took: {:}".format(format_time(time.time() - t0)))

    # Save the model to a file
    # TODO: Make script model saving agnostic
    file_path = args.save_path + args.arch + '_seed' + str(args.seed) + '.pt'
    torch.save(model, file_path)


if __name__ == '__main__':
    main(args)