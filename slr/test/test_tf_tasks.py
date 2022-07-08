from slr.utils.utils import get_tensorboard_callback
from tensorflow.keras.callbacks import TensorBoard


def test_tensorboard():
    assert isinstance(get_tensorboard_callback(),TensorBoard)