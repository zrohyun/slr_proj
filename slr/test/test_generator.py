from sklearn.model_selection import train_test_split
from slr.data.datagenerator import GraphDataGenerator
from slr.data.ksl.datapath import DataPath
import slr.data.datagenerator as dg
import os, sys
import random as rnd
import pytest

from slr.utils.utils import only_test_on_windows


# testing random numbercase
def gen_random_params():
    return [
        (rnd.randint(1, 100), rnd.randint(1, 64)) for i in range(rnd.randint(1, 10))
    ]


# @only_test_on_windows
@pytest.mark.parametrize("n_cls, batch_size", gen_random_params())
def test_generator(n_cls: int, batch_size: int):
    x, y = DataPath(class_limit=n_cls).data
    a = dg.KeyDataGenerator(x, y, batch_size=batch_size)
    assert len(a) * batch_size <= len(y)


def test_graphgenerator():
    batch_size = 32
    class_lim = 100
    x, y = DataPath(class_limit=class_lim).data

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=66, test_size=0.3
    )

    gtrain = GraphDataGenerator(
        x_train,
        y_train,
        batch_size=batch_size,
    )
    X, _ = gtrain[0]
    assert X.shape[-2:] == (137, 137)
