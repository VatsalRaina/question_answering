import torch
from datasets import load_dataset

from utils import _find_sub_list


__all__ = [
    'load_squad_v1'
]


def load_squad_v1(args, tokenizer, device):

    # Load SQuAD v1 dataset
    train_data = load_dataset('squad', split='train')

    # For dataset creation
    input_ids = []
    start_positions_true = []
    end_positions_true = []
    token_type_ids = []

    # Needed for transformer models
    # TODO: How is data loading made model agnostic?
    attention_masks = []

    # Process every example manually
    for ex in train_data:

        # Get question and context
        question, context = ex["question"], ex["context"]

        # Get answer
        # TODO: Make class compatible with multiple answers
        answer = ex["answers"]["text"][0]

        # The input to the transformer model is based on concatenating question and context
        # this is serparated by a special SEP token
        concatenation = question + " [SEP] " + context

        # Tokenizer converts input to be compatible with chosen model
        # Note we truncate inputs exceesding max length
        input_encodings_dict = tokenizer(concatenation, truncation=True, max_length=args.max_len, padding="max_length")

        # Extract necessary pieces from tokenizer output, sequence of token ids
        inp_ids = input_encodings_dict['input_ids']

        # Transformer attention mask
        inp_att_msk = input_encodings_dict['attention_mask']

        # Token type identifies if input is question or context
        # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        tok_type_ids = [0 if i <= inp_ids.index(102) else 1 for i in range(len(inp_ids))]

        # Encode answer into sequence of ids and remove special starting and ending tokens
        ans_ids = tokenizer.encode(answer)[1:-1]

        # Find where in the input the answer is
        start_idx, end_idx = _find_sub_list(ans_ids, inp_ids)
        if start_idx == -1:
            print("Didn't find answer")
            print(answer)
            print(context)
            continue

        # Storing all here and later converting to tensors
        input_ids.append(inp_ids)
        start_positions_true.append(start_idx)
        end_positions_true.append(end_idx)
        token_type_ids.append(tok_type_ids)
        attention_masks.append(inp_att_msk)

    print("Completed preprocessing")

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    start_positions_true = torch.tensor(start_positions_true)
    start_positions_true = start_positions_true.long().to(device)
    end_positions_true = torch.tensor(end_positions_true)
    end_positions_true = end_positions_true.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)

    return input_ids, start_positions_true, end_positions_true, token_type_ids, attention_masks