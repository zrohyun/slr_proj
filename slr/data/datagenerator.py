import time
from slr.data.ksl.datapath import DataPath
import numpy as np
import math
from slr.utils.utils import zero_pad_keypoint_seq as zp

from typing import List

from tensorflow.keras.utils import Sequence, to_categorical
from slr.data.ksl.keypoint_json_parser import KeypointSeq
from sklearn.preprocessing import MinMaxScaler as MMS

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
class KeyDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, x_set, y_set, batch_size=32, seq_len = 200, shuffle=True):
        'Initialization'
        self.x_set, self.y_set = x_set, y_set
        self.batch_size = batch_size
        self.n_classes = len(set(y_set))
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.cls_map = {x: i for i,x in enumerate(list(set(self.y_set)))}
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y_set) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        y: List[int] = self._label_encode(indexes)
        X: np.ndarray = self._get_data_with_zero_pad(indexes)
        
        return X, to_categorical(y,num_classes=self.n_classes).argmax(axis=1)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_set))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _label_encode(self, indexes):
        labels = [self.y_set[k] for k in indexes]
        return [self.cls_map[l] for l in labels]
    
    def _get_data_with_zero_pad(self, indexes) -> np.ndarray:
        X = [KeypointSeq(self.x_set[k]).key_arr for k in indexes]
        X = np.array([zp(x,self.seq_len) for x in X])
        
        #MinMaxScaling on 4dim
        eps = 1e-10 #Cover divide error
        Xmin , Xmax = X.max(axis=(1,2))[:,np.newaxis,np.newaxis,:] + eps , X.min(axis=(1,2))[:,np.newaxis,np.newaxis,:]
        X = np.array(((X - Xmin) / (Xmax - Xmin)))

        return X.reshape(*X.shape[:-2],-1)

if __name__ == '__main__':

    st = time.time()
    x,y = DataPath(class_limit=10).data
    a = KeyDataGenerator(x,y,10)
    x,y = a[0]
    print(x.shape)
    print(y)
    print(time.time()-st)

