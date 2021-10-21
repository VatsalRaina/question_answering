import os
import errno
import random
import datetime

import torch
import numpy as np

__all__ = [
    'format_time',
    'get_default_device',
    '_find_sub_list',
    'mkdir_p',
    'set_seed',
]


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_default_device(use_cuda = True):
    """
    Returns cuda/cpu device
    """
    return torch.device('cuda') if (use_cuda and torch.cuda.is_available()) else torch.device('cpu')


def _find_sub_list(sl, l):
    """
    TODO: Enter description
    """
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll - 1

    print("Didn't find match, return <no answer>")
    return -1, 0


def mkdir_p(path):
    """
    Make dir if not exist.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_seed(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)