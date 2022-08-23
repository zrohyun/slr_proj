from matplotlib.pyplot import cla
from sklearn.model_selection import train_test_split
from slr.data.datagenerator import GraphDataGenerator
from slr.data.ksl.datapath import DataPath
import time
from slr.model.tgcn import TGCN
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary

import tensorflow as tf
import numpy as np
import random
import os


def fix_seed(my_seed=42):
    def my_seed_everywhere_torch(seed: int = 42):
        random.seed(seed)  # random
        np.random.seed(seed)  # numpy
        os.environ["PYTHONHASHSEED"] = str(seed)  # os
        # pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def my_seed_everywhere_tf(seed: int = 42):
        random.seed(seed)  # random
        np.random.seed(seed)  # np
        os.environ["PYTHONHASHSEED"] = str(seed)  # os
        tf.random.set_seed(seed)  # tensorflow

    my_seed_everywhere_torch(my_seed)
    my_seed_everywhere_tf(my_seed)


tgcn_inout_channels = [
    (256, 256, 1),
    (256, 256, 2),
    (256, 512, 1),
    (512, 512, 2),
    (512, 512, 1),
    (512, 256, 1),
]

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


def print_log(str, print_time=True):
    print(str)
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + " ] " + str
    print(str)
    with open("./log.txt", "a") as f:
        print(str, file=f)


def train_tgcn(class_lim=30, batch_size=8, epochs=500):

    x, y = DataPath(class_limit=class_lim).data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=66, test_size=0.3
    )
    train_generator = GraphDataGenerator(
        x_train, y_train, batch_size=batch_size, seq_len=150
    )
    test_generator = GraphDataGenerator(
        x_test, y_test, batch_size=batch_size, seq_len=150
    )

    gtcn_inout_channels_large = [
        (256, 256, 1),
        (256, 256, 2),
        (256, 512, 1),
        (512, 512, 2),
        (512, 512, 1),
        (512, 256, 1),
    ]
    gtcn_inout_channels = [
        (128, 128, 1),
        (128, 128, 2),
        (128, 256, 1),
        (256, 256, 2),
        (256, 256, 1),
        (256, 128, 1),
    ]
    model = TGCN(
        in_channel=137,
        num_keypoints=137,
        in_out_channels=gtcn_inout_channels,
        num_class=class_lim,
        dev=dev,
    ).to(dev)
    summary(model, (150, 137, 137), device="cuda")
    # x,y = train_generator[0]
    criterion = nn.CrossEntropyLoss().float().to(dev)
    optimizer = optim.Adam(model.parameters())
    entire_train_loss = []
    entire_val_loss = []
    entire_train_acc = []
    entire_val_acc = []

    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        val_acc = 0.0
        training_accuracy = 0.0
        val_loss = 0.0
        total = 0

        for i, data in tqdm(enumerate(train_generator)):
            model.train()
            stb = time.time()
            x, y = data
            x = torch.from_numpy(x).float().to(dev)
            y = torch.from_numpy(y).type(torch.LongTensor).to(dev)
            print(x.shape)
            print(y)
            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, outputs = torch.max(outputs, 1)
            training_accuracy += (outputs == y).sum().item() / len(y)
            # print(outputs,y)
            # print(training_accuracy)
            # 통계를 출력합니다.
            train_loss += loss.item()

            # print(f'1batch(16) elapse time:{int(time.time()-stb)}')
            # if i % 2000 == 1999:    # print every 2000 mini-batches
        train_loss = train_loss / len(train_generator)
        training_accuracy = training_accuracy / len(train_generator)
        print_log(
            f"[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.3f} acc: {training_accuracy:.3f}"
        )
        # print_log(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} acc: {training_accuracy/len(train_generator):.3f}')
        # print(f"training acc: {training_accuracy/len(train_generator)}")
        entire_train_acc.append(training_accuracy)
        entire_train_loss.append(train_loss)

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for i, data in tqdm(enumerate(test_generator)):
                inputs, outputs = data
                inputs = torch.from_numpy(inputs).float().to(dev)
                outputs = torch.from_numpy(outputs).type(torch.LongTensor).to(dev)

                predicted_outputs = model(inputs)
                loss = criterion(predicted_outputs, outputs)

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, 1)
                val_loss += loss.item()
                # total += outputs.size(0)
                val_acc += (predicted == outputs).sum().item() / outputs.size(0)

            # Calculate validation loss value
            val_loss = val_loss / len(test_generator)

            # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.
            val_acc = val_acc / len(test_generator)  # / total)
            print_log(f"val_loss: {val_loss}, val_accuracy: {val_acc}")

        entire_val_loss.append(val_loss)
        entire_val_acc.append(val_acc)

    return (
        np.array(entire_train_acc),
        np.array(entire_train_loss),
        np.array(entire_val_acc),
        np.array(entire_val_loss),
    )
    # print(f'1epoch elapse time:{int(time.time() - ste)}')


if __name__ == "__main__":
    epochs = 100
    for cls in [100, 30, 10]:
        fix_seed()
        st = time.time()
        ta, tl, va, vl = train_tgcn(class_lim=cls, epochs=epochs)
        print_log(f"whole time:{int(time.time()-st)}")
        for k, v in zip(["ta", "tl", "va", "vl"], [ta, tl, va, vl]):
            np.save(f"{cls}_{epochs}_{k}", v)
