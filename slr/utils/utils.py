import os, sys

import numpy as np
from sys import platform
import random


def only_test_on_windows(func):
    import pytest

    def wrapper(*args, **kwargs):
        return pytest.mark.skipif(
            platform != "win32", reason="only available windows desktop"
        )(func(*args, **kwargs))

    return wrapper


def get_tensorboard_callback(more_info=None):
    import datetime
    import tensorflow as tf

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if more_info:
        log_dir = log_dir + "_" + more_info
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    return tensorboard_callback


def zero_pad_keypoint_seq(arr: np.array, seq_len: int = 200):
    s, f, d = arr.shape
    if s >= seq_len:
        return arr[:seq_len, :, :]

    return np.vstack([np.zeros((seq_len - s, f, d)), arr])


def fix_seed(my_seed=42):
    def my_seed_everywhere_torch(seed: int = 42):
        random.seed(seed)  # random
        np.random.seed(seed)  # numpy
        os.environ["PYTHONHASHSEED"] = str(seed)  # os
        # pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def my_seed_everywhere_tf(seed: int = 42):
        random.seed(seed)  # random
        np.random.seed(seed)  # np
        os.environ["PYTHONHASHSEED"] = str(seed)  # os
        tf.random.set_seed(seed)  # tensorflow

    try:
        import torch

        my_seed_everywhere_torch(my_seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        my_seed_everywhere_tf(my_seed)
    except ImportError:
        pass


def get_device():
    """get torch device"""
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
