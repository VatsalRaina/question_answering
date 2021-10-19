from transformers import ElectraTokenizer
from transformers import ElectraForQuestionAnswering

__all__ = [
    'qa_electra_large'
]


def qa_electra_large(args, tokenizer_only = False, **kwargs):

    # Model identifier
    electra_large = "google/electra-large-discriminator"

    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    if tokenizer_only: return tokenizer

    # Load pre-trained model
    model = ElectraForQuestionAnswering.from_pretrained(electra_large)

    return model, tokenizer