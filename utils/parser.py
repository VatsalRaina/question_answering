import models as models
import argparse

__all__ = [
    'get_args',
    'get_args_prep'
]


def get_model_names():
    # Get all possible model names as defined in the models directory
    # The chosen architecture must be in this list
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    return model_names


def get_args():
    parser = argparse.ArgumentParser(description='QA Training/Evaluation')

    # Get all possible model names
    model_names = get_model_names()

    # Model architecture
    parser.add_argument('--arch', '-a',
                        type=str, default='qa_electra_large', choices = model_names,
                        help='Specify the type of QA model. Possible choices: ' + ' | '.join(model_names)
    )
    parser.add_argument('--dataset', '-d', type=str, default='squad', choices=["squad", "squad_v2"], help='Squad dataset')
    parser.add_argument('--permute', type=int, default=0, help='Whether to permute dataset to create unanswerable examples')
    parser.add_argument('--permute_directed', type=int, default=0, help='Whether to permute dataset with directed shuffling to create unanswerable examples')

    # Training parameters
    parser.add_argument('--log_every', type=int, default=40, help='Print a logging statement every n batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
    parser.add_argument('--accumulate_gradient_steps', type=int, default=1, help='Number of batch computations before optimizer step')
    parser.add_argument('--epochs', type=int, default=1, help='Specify the number of epochs to train for')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    # Data processing
    parser.add_argument('--max_len', type=int, default=512, help='Specify the maximum number of input tokens')

    # Learning rate params
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Specify the initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Specify the AdamW loss epsilon')
    parser.add_argument('--warmup_fraction', type=float, default=0.1, help='Fraction of warmup in linear schedule')

    # Regularisation
    parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')

    # For models with a separate answerability module
    parser.add_argument('--flatten_train_separation', type=int, default=0, help='Flattening the target for unanswerable examples')
    parser.add_argument('--answer_train_separation', type=int, default=0, help='Training the QA module on only answerable examples')
    parser.add_argument('--answer_alpha', type=float, default=1.0, help='Weight of the answerability loss')

    # For computing attention based uncertainty
    parser.add_argument('--attention_uncertainty', type=int, default=0, help='To compute uncertainty from attention matrices or not')

    # Data paths
    parser.add_argument('--save_all', type=int, default=0, help='Save every epoch checkpoint')
    parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')
    parser.add_argument('--model_path', type=str, help='Load path to trained model')
    parser.add_argument('--predictions_save_path', type=str, help='Where to save predicted values')

    # For self-distribution distillation
    parser = get_self_args(parser)

    return parser


def get_self_args(parser):
    # Self-Distribution Distillation
    parser.add_argument('--self_ratio', type=float, default=0.0, help='Self loss constant size')
    parser.add_argument('--temperature_scale_est', type=float, default=1.0, help='Temperature in estimating dirichlet')
    parser.add_argument('--temperature_scale_num', type=float, default=1.0, help='Temperature in kl loss')
    parser.add_argument('--estimation_iter', type=int, default=1, help='Number of re-estimation steps')
    parser.add_argument('--num_passes', type=int, default=5, help='Number of forward passes through final linear layer')
    parser.add_argument('--reverse', type=int, default=0, help='Use reverse KL loss')
    parser.add_argument('--noise_a', type=float, default=0.0, help='Parameter in stochastic layer')
    parser.add_argument('--noise_b', type=float, default=0.0, help='Parameter in stochastic layer')
    return parser


def get_args_prep():
    parser = argparse.ArgumentParser(description='QA postprocessing for SQuAD evaluation')

    # Dataset and paths
    parser.add_argument('--dataset', '-d', type=str, default='squad', choices=["squad", "squad_v2"], help='Squad dataset')
    parser.add_argument('--permute', type=int, default=0, help='Whether dataset has been permuted to create unanswerable examples')
    parser.add_argument('--permute_directed', type=int, default=0, help='Whether dataset has been permuted with directed shuffling to create unanswerable examples')
    parser.add_argument('--save_dir', type=str, help='Directory to which prepped files will be saved')
    parser.add_argument('--load_dirs', type=str, nargs='+', help='List of directories of saved start and end logits')

    # Max length specification
    parser.add_argument('--max_len', type=int, default=512, help='Specify the maximum number of input tokens')

    # Uncertainty estimation models
    parser.add_argument('--class_uncertainty', type=str, default='ensemblelogits', help="Type of uncertainty estimating class")
    parser.add_argument('--num_samples_uncertainty', type=int, default=100, help="Number of samples drawn when computing uncertainty")
    parser.add_argument('--use_softplus', type=int, default=1, help="Apply softplus to log alphas of dirichlet model similar to training")

    # For uncertainty estimation thresholding in squad v2
    parser.add_argument('--threshold_frac', type=float, default=0.5, help='Threshold for unanswerability')

    # Whether to use explicit/implicit unanswerability probs
    parser.add_argument('--use_unans_probs', type=int, default=0, help='Use explicit unanswerability probs')
    parser.add_argument('--use_unans_probs_implicit', type=int, default=0, help='Use implicit unanswerability probs')

    # Joint thresholding
    parser.add_argument('--use_joint_thresholding', type=int, default=0, help='To use joint thresholding')
    parser.add_argument('--joint_threshold_frac', type=float, default=0, help='Secondary thresholding fraction')
    parser.add_argument('--joint_threshold_metric', type=str, default='unc_logit_conf_expected', help='secondary metric for unanswerability')

    return parser