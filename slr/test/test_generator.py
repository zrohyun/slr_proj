import random as rnd
import pytest
from sys import platform

# testing random numbercase
def gen_random_params():
    return [
        (rnd.randint(1, 100), rnd.randint(1, 64)) for i in range(rnd.randint(1, 10))
    ]


@pytest.mark.skipif(platform != "win32", reason="test only on windows")
@pytest.mark.parametrize("n_cls, batch_size", gen_random_params())
def test_generator(n_cls: int, batch_size: int):
    from slr.data.datagenerator import KeyDataGenerator
    from slr.data.ksl.datapath import DataPath

    x, y = DataPath(class_limit=n_cls).data
    a = KeyDataGenerator(x, y, batch_size=batch_size)
    assert len(a) * batch_size <= len(y)


@pytest.mark.skipif(platform != "win32", reason="test only on windows")
def test_graphgenerator():
    from sklearn.model_selection import train_test_split
    from slr.data.ksl.datapath import DataPath
    from slr.data.datagenerator import GraphDataGenerator

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
