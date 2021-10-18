import argparse

__all__ = [
    'get_args'
]


def get_args():
    parser = argparse.ArgumentParser(description='QnA Training/Evaluation')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Specify the number of epochs to train for')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    # Data processing
    parser.add_argument('--max_len', type=int, default=512, help='Specify the maximum number of input tokens')

    # Learning rate params
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Specify the initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Specify the AdamW loss epsilon')
    parser.add_argument('--warmup_steps', type=int, default=306, help='Number of warmup steps in linear schedule')

    # Regularisation
    parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')

    # Data paths
    parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')
    parser.add_argument('--model_path', type=str, help='Load path to trained model')
    parser.add_argument('--predictions_save_path', type=str, help='Where to save predicted values')
    return parser