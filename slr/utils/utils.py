import os,sys

import numpy as np
from sys import platform
import pytest


def only_test_on_windows(func):

    def wrapper(*args,**kwargs):
        return pytest.mark.skipif(platform != "win32", reason="only available windows desktop")(func(*args,**kwargs))
    
    return wrapper

def get_tensorboard_callback(more_info = None):
    import datetime
    import tensorflow as tf

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if more_info : log_dir = log_dir + "_" + more_info
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback

def zero_pad_keypoint_seq(arr: np.array, seq_len:int = 200):
    s,f,d = arr.shape
    if s>= seq_len:
        return arr[:seq_len,:,:]
    
    return np.vstack([np.zeros((seq_len-s,f,d)),arr])
