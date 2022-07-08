from slr.data.ksl.datapath import *
from sys import platform

def test_datapath():

    if platform == "win32":
        DataPath(class_limit=10)
    else:
        pass
