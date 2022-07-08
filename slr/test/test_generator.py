from slr.data.ksl.datapath import DataPath
import slr.data.datagenerator as dg
import os, sys
import random as rnd
import pytest



# testing random numbercase
def gen_random_params():
    return [(rnd.randint(1,100), rnd.randint(1,64)) for i in range(rnd.randint(1,10))]

@pytest.mark.parametrize("n_cls, batch_size",gen_random_params())
def test_generator(n_cls:int, batch_size:int):
    x,y = DataPath(class_limit=n_cls).data
    a = dg.KeyDataGenerator(x,y,batch_size=batch_size)
    assert (len(a)*batch_size <= len(y))
