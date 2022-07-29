
from tensorflow.keras.callbacks import TensorBoard
from slr.utils.utils import get_tensorboard_callback

def test_tensorboard():

    assert isinstance(get_tensorboard_callback(),TensorBoard)