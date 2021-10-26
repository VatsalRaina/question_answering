import torch
from datasets import load_dataset

from utils import _find_sub_list


__all__ = [
    'load_squad'
]


def load_squad(args, tokenizer, device, split ='train'):

    # Load SQuAD v1 dataset or SQuAD v2
    dataset = load_dataset(args.dataset, split = split)

    # Loading is different for validation
    use_train = (split == 'train')

    # Store all data within dictionary
    # Attention masks needed for transformer models
    # TODO: How is data loading made model agnostic?
    all_data = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_masks': [],
    }

    # For training get labels
    if use_train:
        all_data['start_positions_true'] = []
        all_data['end_positions_true'] = []
        all_data['answerable_true'] = []

    # Process every example manually
    for example in dataset:

        # Get question and context
        question, context = example["question"], example["context"]

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

        # Get answer
        # TODO: Make class compatible with multiple answers
        if use_train:

            # If there is no answer, set the idx to the cls token
            if len(example["answers"]["text"]) == 0:
                start_idx, end_idx = 0, 0

            else:
                answer = example["answers"]["text"][0]

                # Find where in the input the answer is
                start_idx, end_idx = _find_sub_list(
                    tokenizer.encode(answer)[1:-1],
                    tokenizer.encode(context)[1:-1]
                )

                # Add the question length to start and end
                shift = len(tokenizer.encode(question))
                start_idx, end_idx = start_idx + shift, end_idx + shift

                if start_idx == -1:
                    print("Didn't find answer")
                    print(answer)
                    print(context)
                    continue

        # Storing all here and later converting to tensors
        all_data['input_ids'].append(inp_ids)
        all_data['token_type_ids'].append(tok_type_ids)
        all_data['attention_masks'].append(inp_att_msk)

        if use_train:
            all_data['start_positions_true'].append(start_idx)
            all_data['end_positions_true'].append(end_idx)
            all_data['answerable_true'].append(start_idx != 0)

    # Convert to long tensors on device
    for key, value in all_data.items():
        all_data[key] = torch.tensor(value).long().to(device)

    return all_data