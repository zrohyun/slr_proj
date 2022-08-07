import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.layers import LSTM,Dense, Dropout, Input, Flatten

from tensorflow.keras import layers,models, Model
import numpy as np

class BaseLSTM(Model):
    def __init__(self, x,y, lstm_dims, *args, **kwargs):
        super(BaseLSTM,self).__init__(*args, **kwargs)
        # self.input = Input(shape = x.shape[1:])
        self.x_shape = x.shape[1:]
        self.lstm_dims = lstm_dims

        self.lstm1 = LSTM(self.lstm_dims[0], return_sequences=True, input_shape=self.x_shape)
        self.lstm2 = [LSTM(i, return_sequences=True) for i in self.lstm_dims[1:-1]]
        self.lstm3 = LSTM(self.lstm_dims[-1], return_sequences=False)

        self.dropout = Dropout(0.3)
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')
    
    def call(self, inputs):

        x = self.lstm1(inputs)
        
        for lstm_layer in self.lstm2:
            x = lstm_layer(x)
        x = self.lstm3(x)


        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
def lstm(x,y, lstm_dims = [256, 128], dense_dims = [64], dropout = False, drop_rate = 0.3, class_limit = 10):
    model = models.Sequential()

    # model.add(Input(shape = x.shape[1:] ))
    model.add(LSTM(lstm_dims[0],return_sequences=True, input_shape=x.shape[1:]))
    for l in lstm_dims[1:-1]:
        model.add(LSTM(l, return_sequences=True))
    model.add(LSTM(lstm_dims[-1], return_sequences=False))
    

    for d in dense_dims:
        model.add(Dense(d,activation='relu'))
        
        if dropout: model.add(Dropout(drop_rate))
    
    model.add(Dense(class_limit, activation='softmax'))
    # opt = tf.keras.optimizers.Adam()

    # model.compile(optimizer=opt,
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # model.summary()
    return model