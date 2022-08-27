from slr.model.trainer import TorchTrainer
from slr.data.ksl.datapath import DataPath
from slr.model.configs.tgcn_config import CFG_TGCN_v1, CFG_TGCN_v2
from slr.model.tgcn import TGCN, TGCN_v2

from slr.data.datagenerator import (
    GraphDataGenerator,
    KeyDataGenerator,
    TFRecDataGenerator,
    TestDataGenerator,
)

from slr.static.const import TGCN_INOUT_CHANNELS_ver1
from slr.utils.utils import get_device

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchsummary import summary
except ImportError:
    pass

import time
from tqdm import tqdm
import sys

import numpy as np


def print_log(str, print_time=True):
    print(str)
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + " ] " + str
    print(str)
    with open("./log.txt", "a") as f:
        print(str, file=f)


def train_tgcn(class_lim=30, batch_size=8, epochs=500):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = CFG_TGCN_v1
    cfg.model_args["device"] = dev
    cfg.model_args["num_class"] = class_lim
    cfg.epochs = epochs
    cfg.batch_size = batch_size

    x_train, x_test, y_train, y_test = DataPath(class_lim).split_data

    train_generator = GraphDataGenerator(
        x_test, y_test, batch_size=cfg.batch_size, seq_len=150
    )
    # test_generator = GraphDataGenerator(
    #     x_test, y_test, batch_size=batch_size, seq_len=150
    # )
    # train_generator = TestDataGenerator((150,137,137),1,1)
    test_generator = TestDataGenerator((150, 137, 137), 1, 1)

    model = TGCN(**cfg.model_args).to(cfg.model_args["device"])
    summary(model, (150, 137, 137), device="cuda")

    criterion = nn.CrossEntropyLoss().float().to(cfg.model_args["device"])
    optimizer = optim.Adam

    trainer = TorchTrainer(
        model,
        epochs=cfg.epochs,
        train_loader=train_generator,
        test_loader=test_generator,
        optim=optimizer,
        criterion=criterion,
        name="TGCN_trainer",
        dev=cfg.model_args["device"],
        cfg=cfg,
    )

    history = trainer.train()


def summary_model(model, cfg):
    summary(
        model,
        (
            cfg.model_args["window_size"],
            cfg.model_args["num_keypoints"],
            cfg.model_args["channel"],
        ),
    )


def set_config_tgcn_v2():
    if sys.platform == "linux":
        cfg = CFG_TGCN_v2
        cfg.model_args["dev"] = get_device()
        cfg.batch_size = 32
        x_train, x_test, y_train, y_test = DataPath(10).split_data

    elif sys.platform == "linux":
        cfg = CFG_TGCN_v2
        cfg.model_args["dev"] = cfg.model_args["dev"] = get_device()
        cfg.batch_size = 2
        train_loader = TFRecDataGenerator(
            cfg.train_file, comp="GZIP", batch_size=cfg.batch_size
        )
        test_loader = TFRecDataGenerator(
            cfg.test_file, comp="GZIP", batch_size=cfg.batch_size
        )
    else:
        cfg = CFG_TGCN_v2
        cfg.model_args["dev"] = cfg.model_args["dev"] = get_device()
        cfg.batch_size = 2

    return cfg


def train_tgcn_v2():

    cfg = set_config_tgcn_v2()

    model = TGCN_v2(**cfg.model_args, cfg=cfg)
    summary(model, cfg)

    x_train, x_test, y_train, y_test = DataPath(cfg.model_args["num_class"]).split_data

    train_generator = KeyDataGenerator(
        x_test, y_test, batch_size=cfg.batch_size, seq_len=150
    )
    test_generator = TestDataGenerator((150, 137, 3), 1)
    # test_generator = KeyDataGenerator(
    #     x_test, y_test, batch_size=cfg.batch_size, seq_len=150
    # )

    criterion = nn.CrossEntropyLoss().float().to(cfg.model_args["device"])
    optimizer = optim.Adam

    trainer = TorchTrainer(
        model,
        epochs=cfg.epochs,
        train_loader=train_generator,
        test_loader=test_generator,
        optim=optimizer,
        criterion=criterion,
        name="TGCN_trainer",
        dev=cfg.model_args["device"],
        cfg=cfg,
    )
    history = trainer.train()


if __name__ == "__main__":

    # training tgcn_v1
    train_tgcn(class_lim=100, epochs=100, batch_size=2)

    train_tgcn_v2()
