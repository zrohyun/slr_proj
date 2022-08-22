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
    "Generates data for Keras"

    def __init__(
        self, x_set, y_set, batch_size=32, seq_len=200, shuffle=True, scale=True
    ):
        "Initialization"
        self.x_set, self.y_set = x_set, y_set
        self.batch_size = batch_size
        self.n_classes = len(set(y_set))
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.scale = scale
        self.cls_map = {x: i for i, x in enumerate(list(set(self.y_set)))}
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.y_set) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        y: List[int] = self._label_encode(indices)
        X: np.ndarray = self._get_data_with_zero_pad(indices)

        return X, to_categorical(y, num_classes=self.n_classes).argmax(axis=1)

    def on_epoch_end(self):
        "Updates indices after each epoch"
        self.indices = np.arange(len(self.x_set))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def _label_encode(self, indices):
        labels = [self.y_set[k] for k in indices]
        return [self.cls_map[l] for l in labels]

    def _minmax4dim(self, X, eps=1e-5):
        # MinMaxScaling on 4dim
        # Cover divide error by add eps
        Xmax, Xmin = (
            X.max(axis=(0, 1))[None,None, :] + eps,
            X.min(axis=(0, 1))[None,None, :],
        )
        X = np.array(((X - Xmin) / (Xmax - Xmin)))
        return X

    def _get_data_with_zero_pad(self, indices) -> np.ndarray:
        X = self._read_data(indices)
        X = X.reshape(*X.shape[:-2], -1)  # flatten feature
        # X.shape = (batch_size, seq_len, 137s)
        return X

    def _read_data(self, indices) -> np.ndarray:
        X = [KeypointSeq(self.x_set[k]).key_arr for k in indices]

        if self.scale:
            X = [self._minmax4dim(x) for x in X]

        X = np.array([zp(x, self.seq_len) for x in X])

        return X


if __name__ == "__main__":

    st = time.time()
    x, y = DataPath(class_limit=10).data
    a = KeyDataGenerator(x, y, 10)
    x, y = a[0]
    print(x.shape)
    print(y)
    print(time.time() - st)
