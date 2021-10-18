#! /usr/bin/env python

"""
Evaluate a simple question-answering model for extractive reading comprehension
"""

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import datetime

from datasets import load_dataset
from transformers import ElectraTokenizer
from transformers import ElectraForQuestionAnswering, AdamW, ElectraConfig

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")

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
    with open('CMDs/test.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    dev_data = load_dataset('squad', split='validation')
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    input_ids = []
    attention_masks = []
    token_type_ids = []

    for ex in dev_data:
        question, context = ex["question"], ex["context"]
        combo = question + " [SEP] " + context
        input_encodings_dict = tokenizer(combo, truncation=True, max_length=args.max_len, padding="max_length")
        inp_ids = input_encodings_dict['input_ids']
        inp_att_msk = input_encodings_dict['attention_mask']
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        attention_masks.append(inp_att_msk)

    print("Completed preprocessing")

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)

    eval_data = TensorDataset(input_ids, token_type_ids, attention_masks)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    pred_start_logits = []
    pred_end_logits = []
    count = 0

    for b_input_ids, b_tok_typ_ids, b_att_msks in eval_dataloader:
        print(count)
        count+=1
        b_input_ids, b_tok_typ_ids, b_att_msks = b_input_ids.to(device), b_tok_typ_ids.to(device), b_att_msks.to(device)
        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids, return_dict=True)
        b_start_logits, b_end_logits = outputs.start_logits, outputs.end_logits
        pred_start_logits += b_start_logits.detach().cpu().numpy().tolist()
        pred_end_logits += b_end_logits.detach().cpu().numpy().tolist()
    
    pred_start_logits, pred_end_logits = np.asarray(pred_start_logits), np.asarray(pred_end_logits)
    np.save(args.predictions_save_path + "pred_start_logits.npy", pred_start_logits)
    np.save(args.predictions_save_path + "pred_end_logits.npy", pred_end_logits)