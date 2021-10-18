#! /usr/bin/env python


"""
Train a simple question-answering model for extractive reading comprehension
"""

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import ElectraTokenizer
from transformers import ElectraForQuestionAnswering, AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--max_len', type=int, default=512, help='Specify the maximum number of input tokens')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    print("Didn't find match, return <no answer>")
    return -1,0

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    # Load SQuAD v1 dataset
    train_data = load_dataset('squad', split='train')

    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    start_positions_true = []
    end_positions_true = []
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for ex in train_data:
        question, context, answer = ex["question"], ex["context"], ex["answers"]["text"][0]
        combo = question + " [SEP] " + context
        input_encodings_dict = tokenizer(combo, truncation=True, max_length=args.max_len, padding="max_length")
        inp_ids = input_encodings_dict['input_ids']
        inp_att_msk = input_encodings_dict['attention_mask']
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        
        ans_ids = tokenizer.encode(answer)[1:-1]    # Strip off start and tokens from answer IDs
        start_idx, end_idx = _find_sub_list(ans_ids, inp_ids)
        if start_idx == -1:
            print("Didn't find answer")
            print(answer)
            print(context)
            continue
        
        start_positions_true.append(start_idx)
        end_positions_true.append(end_idx)
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        attention_masks.append(inp_att_msk)

    print("Completed preprocessing")

    start_positions_true = torch.tensor(start_positions_true)
    start_positions_true = start_positions_true.long().to(device)
    end_positions_true = torch.tensor(end_positions_true)
    end_positions_true = end_positions_true.long().to(device)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)

    # Create the DataLoader for training set.
    train_data = TensorDataset(input_ids, start_positions_true, end_positions_true, token_type_ids, attention_masks)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Load pre-trained model
    model = ElectraForQuestionAnswering.from_pretrained(electra_large)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = args.learning_rate, eps = args.adam_epsilon)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 306,
                                                num_training_steps = total_steps)

    for epoch in range(args.n_epochs):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_start_pos_true = batch[1].to(device)
            b_end_pos_true = batch[2].to(device)
            b_tok_typ_ids = batch[3].to(device)
            b_att_msks = batch[4].to(device)
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids, start_positions=b_start_pos_true, end_positions=b_end_pos_true)
            loss = outputs[0]
            total_loss += loss.item()
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
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # Save the model to a file
    file_path = args.save_path+'electra_qa_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)