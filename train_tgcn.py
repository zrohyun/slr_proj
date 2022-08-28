from pathlib import Path

from sklearn.model_selection import train_test_split
from slr.model.trainer import TorchTrainer
from slr.data.ksl.datapath import DataPath
from slr.model.configs.tgcn_config import CFG_TGCN_v1, CFG_TGCN_v2
from slr.model.tgcn import TGCN, TGCN_v2

from slr.data.datagenerator import (
    GraphDataGenerator,
    KSLTFRecDataGenerator,
    KeyDataGenerator,
    # TFRecDataGenerator,
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
    cfg.model_args["dev"] = dev
    cfg.model_args["num_class"] = class_lim
    cfg.epochs = epochs
    cfg.batch_size = batch_size

    x_train, x_test, y_train, y_test = DataPath(class_lim).split_data

    train_generator = GraphDataGenerator(
        x_train, y_train, batch_size=cfg.batch_size, seq_len=150
    )
    test_generator = GraphDataGenerator(
        x_test, y_test, batch_size=cfg.batch_size, seq_len=150
    )
    # train_generator = TestDataGenerator((150,137,137),1,1)
    # test_generator = TestDataGenerator((150, 137, 137), 1, 1)

    model = TGCN(**cfg.model_args).to(cfg.model_args["dev"])
    summary(model, (150, 137, 137), device="cuda")

    criterion = nn.CrossEntropyLoss().float().to(cfg.model_args["dev"])
    optimizer = optim.Adam

    trainer = TorchTrainer(
        model,
        epochs=cfg.epochs,
        train_loader=train_generator,
        test_loader=test_generator,
        optim=optimizer,
        criterion=criterion,
        name="TGCN_trainer_add_relu",
        dev=cfg.model_args["dev"],
        cfg=cfg,
    )

    history = trainer.train()

def train_tgcn_not_using_trainer(class_lim=30, batch_size=8, epochs=500):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = CFG_TGCN_v1
    cfg.model_args["dev"] = dev
    cfg.model_args["num_class"] = class_lim
    cfg.epochs = epochs
    cfg.batch_size = batch_size

    x_train, x_test, y_train, y_test = DataPath(class_lim).split_data

    train_generator = GraphDataGenerator(
        x_train, y_train, batch_size=cfg.batch_size, seq_len=150
    )
    test_generator = GraphDataGenerator(
        x_test, y_test, batch_size=cfg.batch_size, seq_len=150
    )
    # train_generator = TestDataGenerator((150,137,137),1,1)
    # test_generator = TestDataGenerator((150, 137, 137), 1, 1)

    model = TGCN(**cfg.model_args).to(cfg.model_args["dev"])
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
            # print(x.shape)
            # print(y)
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
        np.array(entire_val_loss),)



def summary_model(model, cfg):
    input_shape = (
        cfg.window_size,
        cfg.model_args["num_keypoints"],
        cfg.model_args["in_channel"],
    )
    summary(model, input_shape)


def set_config_tgcn_v2():
    if sys.platform == "win32":
        cfg = CFG_TGCN_v2
        cfg.model_args["dev"] = get_device()
        cfg.batch_size = 8

    elif sys.platform == "linux":
        cfg = CFG_TGCN_v2
        cfg.model_args["dev"] = cfg.model_args["dev"] = get_device()

    else:
        cfg = CFG_TGCN_v2
        cfg.model_args["dev"] = cfg.model_args["dev"] = get_device()
        cfg.batch_size = 2

    return cfg


def get_loader():
    cfg = CFG_TGCN_v2
    if sys.platform == "win32":
        cfg.model_args["dev"] = get_device()
        cfg.train_file = Path(
            r"C:\Users\user\Documents\GitHub\slr_proj\slr\data\gzip_train_with_preprocess.tfrec"
        )
        cfg.test_file = Path(
            r"C:\Users\user\Documents\GitHub\slr_proj\slr\data\gzip_test_with_preprocess.tfrec"
        )

        x_train, x_test, y_train, y_test = DataPath(
            cfg.model_args["num_class"]
        ).split_data
        # train_loader = KeyDataGenerator(
        #     x_train,
        #     y_train,
        #     batch_size=cfg.batch_size,
        #     seq_len=150,
        # )
        # test_loader = KeyDataGenerator(
        #     x_test,
        #     y_test,
        #     batch_size=cfg.batch_size,
        #     seq_len=150,
        # )
        train_loader = KSLTFRecDataGenerator(
            cfg.train_file,
            "GZIP",
            batch_size=cfg.batch_size,
            channel=cfg.model_args["in_channel"],
        )
        test_loader = KSLTFRecDataGenerator(
            cfg.test_file,
            "GZIP",
            batch_size=cfg.batch_size,
            channel=cfg.model_args["in_channel"],
        )

    elif sys.platform == "linux":
        train_loader = TFRecDataGenerator(
            cfg.train_file, comp="GZIP", batch_size=cfg.batch_size
        )
        test_loader = TFRecDataGenerator(
            cfg.test_file, comp="GZIP", batch_size=cfg.batch_size
        )
    else:
        input_shape = (
            cfg.window_size,
            cfg.model_args["num_keypoints"],
            cfg.model_args["in_channel"],
        )
        train_loader = TestDataGenerator((input_shape), 1)
        test_loader = TestDataGenerator((input_shape), 1)

    return train_loader, test_loader


def train_tgcn_v2():

    cfg = set_config_tgcn_v2()
    model = TGCN_v2(**cfg.model_args, cfg=cfg).to(cfg.model_args["dev"])

    summary_model(model, cfg)

    # train_loader, test_loader = get_loader()

    # just check working
    _, train_loader = get_loader()

    test_loader = TestDataGenerator((150, 137, 3), 1)

    criterion = nn.CrossEntropyLoss().float().to(cfg.model_args["dev"])
    optimizer = optim.Adam

    trainer = TorchTrainer(
        model,
        epochs=cfg.epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        optim=optimizer,
        criterion=criterion,
        name="TGCN_v2_trainer",
        dev=cfg.model_args["dev"],
        cfg=cfg,
    )
    history = trainer.train()

def train_tgcn_original(
   class_lim = 30,
   batch_size = 8,
   epochs = 500):
   
   
   if torch.cuda.is_available(): 
    dev = "cuda:0" 
   else: 
    dev = "cpu" 
    
   x,y = DataPath(class_limit= class_lim).data
   x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.3)
   train_generator = GraphDataGenerator(x_train,y_train,batch_size = batch_size,seq_len=150)
   test_generator = GraphDataGenerator(x_test,y_test,batch_size = batch_size, seq_len=150)

   gtcn_inout_channels_large = [
     (256,256,1),(256,256,2),(256,512,1),(512,512,2),(512,512,1),(512,256,1)
   ]
   gtcn_inout_channels = [
     (128,128,1),(128,128,2),(128,256,1),(256,256,2),(256,256,1),(256,128,1)
   ]
   model = TGCN(in_channel = 137,
               num_keypoints = 137,
               in_out_channels = gtcn_inout_channels,
               num_class=class_lim,
               dev = dev).to(dev)

   # x,y = train_generator[0]
   criterion = nn.CrossEntropyLoss().float().to(dev)
   optimizer = optim.Adam(model.parameters())
   entire_train_loss = []
   entire_val_loss = []
   entire_train_acc = []
   entire_val_acc = []

   for epoch in tqdm( range(epochs)):
     train_loss = 0.0 
     val_acc = 0.0
     training_accuracy = 0.0
     val_loss = 0.0 
     total = 0

     for i,data in tqdm(enumerate(train_generator)):
       model.train() 
       stb = time.time()
       x,y = data
       x = torch.from_numpy(x).float().to(dev)
       y = torch.from_numpy(y).type(torch.LongTensor).to(dev)

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
     print_log(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.3f} acc: {training_accuracy:.3f}')
     # print_log(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} acc: {training_accuracy/len(train_generator):.3f}')
     # print(f"training acc: {training_accuracy/len(train_generator)}")
     entire_train_acc.append(training_accuracy)
     entire_train_loss.append(train_loss)      

     # Validation Loop 
     with torch.no_grad(): 
       model.eval() 
       for i,data in tqdm(enumerate(test_generator)): 
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
       val_loss = val_loss/len(test_generator) 

       # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
       val_acc = val_acc / len(test_generator)# / total)
       print_log(f"val_loss: {val_loss}, val_accuracy: {val_acc}")


     entire_val_loss.append(val_loss)
     entire_val_acc.append(val_acc)

   return np.array(entire_train_acc), np.array(entire_train_loss), np.array(entire_val_acc), np.array(entire_val_loss)
     # print(f'1epoch elapse time:{int(time.time() - ste)}')
if __name__ == "__main__":

    # training tgcn_v1
    # train_tgcn(class_lim=100, epochs=100, batch_size=2)
    train_tgcn_not_using_trainer(class_lim=100, epochs=100, batch_size=8)
    # train_tgcn_original(100,epochs=100)
    # train_tgcn_v2()
