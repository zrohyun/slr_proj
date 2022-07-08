from enum import auto
from slr.data.ksl.datapath import *
from sys import platform

import numpy as np

from slr.data.ksl.datapath import DataPath
from slr.utils.utils import zero_pad_keypoint_seq

import pytest

# @pytest.fixture("function")
def only_test_on_windows(func):

    def wrapper(*args):
        return pytest.mark.skipif(platform != "win32", reason="only available windows desktop")(func)
    
    return wrapper

@only_test_on_windows
def test_datapath():
    DataPath(class_limit=10)

@only_test_on_windows
def test_data_path_class_test(k = 10):
    assert len(DataPath(k).class_dict) == k

@only_test_on_windows
def test_zero_pad():
    in_arr,seq_len = np.ones((100,10,10)),10
    assert in_arr[:10,:,:].shape == zero_pad_keypoint_seq(in_arr,seq_len=seq_len).shape

@only_test_on_windows
def test_data_path_case_cnt(lim = 10):
    tmp = sum([len(v) for k,v in DataPath(lim).class_dict.items()])
    angle = 5
    people = 9
    assert tmp == (lim*angle*people)