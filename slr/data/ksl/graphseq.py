import os, sys
import time
from pathlib import Path

from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
from scipy.spatial.distance import cdist
import pickle


import numpy as np

from slr.data.ksl.keypoint import *
from slr.data.ksl.datapath import DataPath
from slr.data.ksl.keypoint_json_parser import KeypointSeq
from sklearn.metrics import pairwise_distances


@dataclass
class GraphSeq(KeypointSeq):
    data_path: Path
    word_cls: str = ""

    @property
    def graph_mat(self) -> np.ndarray:
        npy_path = self.data_path / (self.data_path.name + "_adj" + ".npy")
        return self._load_data(npy_path)

    def _load_data(self, file_name: Path) -> np.ndarray:
        if file_name.is_file():
            # print('loading graph')
            return np.load(file_name)
        else:
            # print('load_keypoint_data')
            data = KeypointSeq(self.data_path).key_arr
            data = self._minmax4dim(data)
            data = self._adj_mat(data)
            np.save(file_name, data)
            return data

    def _minmax4dim(self, X, eps=1e-10):
        # MinMaxScaling on 4dim
        # Cover divide error by add eps
        Xmax, Xmin = (
            X.max(axis=1)[:, np.newaxis, :] + eps,
            X.min(axis=1)[:, np.newaxis, :],
        )
        X = np.array(((X - Xmin) / (Xmax - Xmin)))
        return X

    def _adj_mat(self, X):

        delta = 0.5
        A = np.stack(
            [
                # self._normalize_undigraph(np.exp(- cdist(X[w],X[w], metric='euclidean') / (2. * delta ** 2)))
                np.exp(-cdist(X[w], X[w], metric="euclidean") / (2.0 * delta**2))
                # self._normalize_undigraph(np.exp(- 1./(2 * 1) * pairwise_distances(X[i,w], metric='sqeuclidean')))
                for w in range(X.shape[0])
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
