from slr.data.ksl.datapath import DataPath
from slr.data.ksl.keypoint_json_parser import KeypointSeq
from slr.data.ksl.graphseq import GraphSeq
from slr.utils.utils import zero_pad_keypoint_seq as zp


import numpy as np
from scipy.spatial.distance import cdist

try:
    import tensorflow as tf
    from tensorflow.keras.utils import Sequence, to_categorical
except Exception:
    pass


import time
from typing import Iterator, List, Optional
from functools import partial


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
            X.max(axis=(0, 1))[None, None, :] + eps,
            X.min(axis=(0, 1))[None, None, :],
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


class GraphDataGenerator(KeyDataGenerator):
    """
    Data Generator for Keras
    Here, `x_set` is list of path to the Adj Matrix
    and `y_set` are the associated classes.
    """

    def __init__(
        self, x_set, y_set, batch_size=32, seq_len=200, shuffle=True, scale=False
    ):
        "Initialization"
        super().__init__(x_set, y_set, batch_size, seq_len, shuffle, scale)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        y: List[int] = self._label_encode(indices)
        # X = self._read_data(indices)
        X = self._get_data(indices)
        return X, to_categorical(y, num_classes=self.n_classes).argmax(axis=1)

    def _read_data(self, indices) -> np.ndarray:
        X = [GraphSeq(self.x_set[k]).graph_mat for k in indices]
        X = np.array([zp(x, self.seq_len) for x in X])

        return X

    def _get_data(self, indices) -> np.ndarray:
        X = self._read_data(indices)
        # X = self._adj_mat(X)
        return X

    def _adj_mat(self, X):
        from sklearn.metrics import pairwise_distances

        delta = 0.5
        A = np.stack(
            [
                np.stack(
                    [
                        self._normalize_undigraph(
                            np.exp(
                                -cdist(X[i, w], X[i, w], metric="euclidean")
                                / (2.0 * delta**2)
                            )
                        )
                        # self._normalize_undigraph(np.exp(- 1./(2 * 1) * pairwise_distances(X[i,w], metric='sqeuclidean')))
                        for w in range(200)
                    ]
                )
                for i in range(self.batch_size)
            ]
        )
        return A

    def _normalize_undigraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD


class KSLTFRecDataGenerator:
    def __init__(self, tfrec_file, comp=None, batch_size=32, channel=2):
        self.dataset = self._get_ksl_dataset_from_tfrec(
            tfrec_file, comp, batch_size, channel
        )
        self._iterator = self.dataset

    def __iter__(self):
        return self

    def _reset(self):
        self._iterator = iter(self.dataset)

    def __next__(self):
        batch = next(self._iterator)
        # return batch
        batch = self._parse_batch(batch)
        return batch["raw_data"], batch["label"]

    def _parse_batch(self, batch):
        # print(batch[0])
        batch["label"] = np.array(list(map(lambda x: int(x.decode()), batch["label"])))
        return batch

    def _get_ksl_dataset_from_tfrec(self, file, comp, batch_size, channel):
        dataset = tf.data.TFRecordDataset(file, comp)
        parsed_dataset = self._get_parsed_dataset(dataset, batch_size, channel)
        parsed_dataset = parsed_dataset.as_numpy_iterator()
        # tfds.as_numpy(parsed_dataset)
        return parsed_dataset

    def _get_parsed_dataset(
        self, dataset: tf.data.TFRecordDataset, batch_size, channel
    ):
        image_feature_description = {
            "raw_data": tf.io.FixedLenFeature([], tf.string),
            "data_shape": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(example):
            example = tf.io.parse_single_example(example, image_feature_description)

            example["raw_data"] = tf.reshape(
                tf.io.decode_raw(example["raw_data"], tf.float32), (-1, 137, channel)
            )

            return {"raw_data": example["raw_data"], "label": example["label"]}

        # session.run(my_example, feed_dict={serialized: my_example_str})
        parsed_dataset = dataset.map(_parse_function).batch(batch_size)
        parsed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return parsed_dataset


class TestDataGenerator:
    """Data Iterator(Generator) for Model debugging"""

    def __init__(self, input_shape: tuple, batch_size: int, iteration: int = 1):
        """using partial for fetching data when user needs"""
        self.x = [
            partial(np.random.random, (batch_size, *input_shape))
            for _ in range(iteration)
        ]
        self.x = iter(self.x)

        self.y = iter(
            [partial(np.random.randint, 10, size=batch_size) for _ in range(iteration)]
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.x)(), next(self.y)()


if __name__ == "__main__":

    st = time.time()
    x, y = DataPath(class_limit=10).data
    a = KeyDataGenerator(x, y, 10)
    x, y = a[0]
    print(x.shape)
    print(y)
    print(time.time() - st)
