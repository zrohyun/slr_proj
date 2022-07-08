import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.layers import LSTM,Dense, Dropout, Input

from tensorflow.keras import layers,models, Model
import numpy as np

class BaseLSTM(Model):
    def __init__(self, x,y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = Input(shape = x.shape[1:])
        self.lstm = LSTM
        self.dropout = Dropout(0.3)
        self.dense = Dense(len(set(y)), activation='softmax')
    
    def call(self, inputs):
        x = self.input(inputs)
        x = self.lstm(200,return_sequences=True)(x)
        x = self.lstm(64,return_sequences=False)(x)
        x = self.dropout(x)
        x = self.dense(x)

def lstm(x,y):
    model = models.Sequential()

    # model.add(Input(shape = x.shape[1:] ))
    model.add(LSTM(200,return_sequences=True, input_shape=x.shape[1:]))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10, activation='softmax'))
    opt = tf.keras.optimizers.Adam()

    model.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # model.summary()
    return model