import os,sys

import numpy as np

def get_tensorboard_callback():
    import datetime
    import tensorflow as tf

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback

def zero_pad_keypoint_seq(arr: np.array, seq_len:int = 200):
    s,f,d = arr.shape
    if s>= seq_len:
        return arr[:seq_len,:,:]
    
    return np.vstack([np.zeros((seq_len-s,f,d)),arr])
