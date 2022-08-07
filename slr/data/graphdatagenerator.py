import time
from slr.data.datagenerator import KeyDataGenerator
from slr.data.ksl.datapath import DataPath
import numpy as np
import math
from slr.data.ksl.graphseq import GraphSeq
from slr.utils.utils import zero_pad_keypoint_seq as zp
from scipy.spatial.distance import cdist

from typing import List

from tensorflow.keras.utils import Sequence, to_categorical
from slr.data.ksl.keypoint_json_parser import KeypointSeq
from sklearn.preprocessing import MinMaxScaler as MMS


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
class GraphDataGenerator(KeyDataGenerator):
    'Generates data for Keras'
    def __init__(self, x_set, y_set, batch_size=32, seq_len = 200, shuffle=True, scale=False):
        'Initialization'
        super().__init__(x_set,y_set,batch_size,seq_len,shuffle,scale )
    

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        y: List[int] = self._label_encode(indices)
        # X = self._read_data(indices)
        X = self._get_data(indices)
        return X, to_categorical(y, num_classes=self.n_classes).argmax(axis=1)
    def _read_data(self, indices) -> np.ndarray:
        X = [GraphSeq(self.x_set[k]).graph_mat for k in indices]
        X = np.array([zp(x,self.seq_len) for x in X])
        
        return X
    
    def _get_data(self, indices) -> np.ndarray:
        X = self._read_data(indices)
        # X = self._adj_mat(X)
        return X

    def _adj_mat(self,X):
        from sklearn.metrics import pairwise_distances
        delta=0.5
        A = np.stack([
            np.stack([ 
                self._normalize_undigraph(np.exp(- cdist(X[i,w],X[i,w], metric='euclidean') / (2. * delta ** 2)))

                    # self._normalize_undigraph(np.exp(- 1./(2 * 1) * pairwise_distances(X[i,w], metric='sqeuclidean')))
                for w in range(200) ])  
            for i in range(self.batch_size)])
        
        # B = np.stack([
        #     np.stack([ 
        #         self._normalize_undigraph(np.exp(- cdist(X[i,w],X[i,w], metric='euclidean') / (2. * delta ** 2)))

        #             # self._normalize_undigraph(np.exp(- 1./(2 * 1) * pairwise_distances(X[i,w], metric='sqeuclidean')))
        #         for w in range(200) ])  
        #     for i in range(self.batch_size)])

        return A
    
    def _normalize_undigraph(self,A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD