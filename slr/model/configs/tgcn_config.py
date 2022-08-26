from importlib import import_module


class CFG_TGCN_v2:
    """TGCN class 100 config"""

    model_name = "tgcn_v2"
    test_file = "gzip_test_with_preprocess.tfrec"
    train_file = "gzip_train_with_preprocess.tfrec"
    model_args = {
        "in_channel": 3,
        "num_class": 100,
        "num_keypoints": 137,
        "dev": "cpu",
        "in_out_channels": None,
        "dropout": 0.3,
        "kernel_size": 9,
        "stride": 1,
    }
    epochs = 100
    batch_size = 32
    seed = 42


class CFG_TGCN_v1:
    """TGCN v1 class 100 config"""

    model_name = "tgcn_v1"
    test_file = "gzip_test_with_preprocess.tfrec"
    train_file = "gzip_train_with_preprocess.tfrec"
    model_args = {
        "in_channel": 137,
        "num_class": 100,
        "num_keypoints": 137,
        "dev": "cpu",
        "in_out_channels": None,
        "dropout": 0.3,
        "kernel_size": 9,
        "stride": 1,
    }
    epochs = 100
    batch_size = 32
    seed = 42


if __name__ == "__main__":
    import sys, os

    print(os.getcwd())
    sys.path.append(".")
