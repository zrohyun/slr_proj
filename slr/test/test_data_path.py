import os
from sys import platform

from pathlib import Path
import numpy as np

import pytest


@pytest.mark.skipif(platform != "win32", reason="requires python3.10 or higher")
def test_data_path_class_test(k=10):
    from slr.data.ksl.datapath import DataPath

    assert len(DataPath(k).class_dict) == k


@pytest.mark.skipif(platform != "win32", reason="requires python3.10 or higher")
def test_zero_pad():
    from slr.utils.utils import zero_pad_keypoint_seq

    in_arr, seq_len = np.ones((100, 10, 10)), 10
    assert (
        in_arr[:10, :, :].shape == zero_pad_keypoint_seq(in_arr, seq_len=seq_len).shape
    )


@pytest.mark.skipif(platform != "win32", reason="requires python3.10 or higher")
def test_data_path_case_cnt(lim=10):
    from slr.data.ksl.datapath import DataPath

    tmp = sum([len(v) for k, v in DataPath(lim).class_dict.items()])
    angle = 5
    people = len(
        list(
            filter(
                lambda x: ".zip" not in x,
                os.listdir(Path(r"D:\ksl\ksl\수어 영상\1.Training")),
            )
        )
    )
    assert tmp == (lim * angle * people)
