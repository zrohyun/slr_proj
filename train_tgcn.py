from slr.data.datagenerator import GraphDataGenerator
from slr.data.ksl.datapath import DataPath
from slr.model.tgcn import TGCN
from slr.model.trainer import TorchTrainer

import torch
import torch.nn as nn
import torch.optim as optim
from slr.static.const import TGCN_INOUT_CHANNELS_ver1
from torchsummary import summary

import time
from tqdm import tqdm

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
    x_train, x_test, y_train, y_test = DataPath(10).split_data

    train_generator = GraphDataGenerator(
        x_train, y_train, batch_size=batch_size, seq_len=150
    )
    test_generator = GraphDataGenerator(
        x_test, y_test, batch_size=batch_size, seq_len=150
    )

    model = TGCN(
        in_channel=137,
        num_keypoints=137,
        in_out_channels=TGCN_INOUT_CHANNELS_ver1,
        num_class=class_lim,
        dev=dev,
    ).to(dev)
    summary(model, (150, 137, 137), device="cuda")

    criterion = nn.CrossEntropyLoss().float().to(dev)
    optimizer = optim.Adam

    trainer = TorchTrainer(
        model,
        epochs=epochs,
        train_loader=train_generator,
        test_loader=test_generator,
        optim=optimizer,
        criterion=criterion,
        name="TGCN_trainer",
    )

    history = trainer.train()


if __name__ == "__main__":
    epochs = 100
    for cls in [100, 30, 10]:
        train_tgcn(class_lim=cls, epochs=epochs)
