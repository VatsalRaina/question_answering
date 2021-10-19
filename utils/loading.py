import models as models


def load_model(args, tokenizer_only = False, **kwargs):
    """
    This function will load any model by defining model loading function
    in models directory. The model loading function should have the following
    interface: qa_func(args, tokenizer_only, **kwargs) -> Union[model, (model, tokenizer)]
    """

    # Model arch name should start with qa
    assert args.arch.startswith('qa')

    return models.__dict__[args.arch](
        args = args,
        tokenizer_only = tokenizer_only,
        **kwargs
    )