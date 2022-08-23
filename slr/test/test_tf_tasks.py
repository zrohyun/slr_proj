import pytest
import platform


@pytest.mark.skipif(platform != "win32", reason="requires python3.10 or higher")
def test_tensorboard():
    from tensorflow.keras.callbacks import TensorBoard
    from slr.utils.utils import get_tensorboard_callback

    assert isinstance(get_tensorboard_callback(), TensorBoard)


def test_hi():
    assert True == True
