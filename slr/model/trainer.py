from functools import partial
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from itertools import tee
from typing import Iterator, List, Tuple
from tqdm import tqdm
import time


class TorchTrainer:
    """
    Torch model trainer class
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        train_loader: Iterator,
        test_loader: Iterator,
        optim,
        criterion,
        cfg,
        name: str = "torch_trainer",
        log: bool = True,
        log_file: Path = Path("./log.txt"),
        dev: torch.device = torch.device("cpu"),
        save_history: bool = True,
    ):

        self.model = model.to(dev)

        self.epochs = epochs

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optim = optim(self.model.parameters())
        self.criterion = criterion.to(dev)

        self.dev = dev
        self.cfg = cfg
        self.name = name

        self.log_file = log_file
        self.save_history = save_history

        if log:
            self._print_log(f"\n Training {self.name}\n")
            self._print_log(f"{self.model}", print_time=False)

    def _cvt_tensor(self, d):
        x, y = d
        x = torch.from_numpy(x).float().to(self.dev)
        y = torch.from_numpy(y).type(torch.LongTensor).to(self.dev)
        return x, y

    def train(self) -> dict:
        """
        training step
        1. run training step
        2. run validation step
        3. repeat for epochs
        return training_history
        """
        training_acc, training_loss = [], []
        val_acc, val_loss = [], []

        for epoch in tqdm(range(self.epochs)):

            train_loader, test_loader = self._copy_iter()

            # train step
            acc, loss, iteration = self.train_step(train_loader)

            training_acc.append(acc)
            training_loss.append(loss)

            self._print_log(
                f"[{epoch + 1}, {iteration :5d}] train_loss: {loss:.3f} train_acc: {acc:.3f}"
            )

            # validation step
            acc, loss, iteration = self.validate(test_loader)
            val_acc.append(acc)
            val_loss.append(loss)

            self._print_log(
                f"[{epoch + 1}, {iteration :5d}] val_loss: {loss:.3f} val_acc: {acc:.3f}"
            )

        history = {
            "training_acc": training_acc,
            "training_loss": training_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }
        if self.save_history:
            self._save_history(history)

        return history

    def train_step(self, train_loader) -> Tuple[int, int, int]:
        """validation step
        return (acc,loss,iter_size)
        """

        self.model.train()

        t_acc, t_loss = 0.0, 0.0

        loop = tqdm(enumerate(train_loader))

        for i, d in loop:
            x, y = self._cvt_tensor(d)

            t_acc, t_loss = self._inference_step(x, y, t_acc, t_loss)

        iteration = i + 1
        t_acc = t_acc / iteration
        t_loss = t_loss / iteration

        return t_acc, t_loss, iteration

    def validate(self, test_loader) -> Tuple[int, int, int]:
        """validation step
        return (acc,loss,iter_size)
        """
        v_acc, v_loss = 0.0, 0.0

        with torch.no_grad():
            self.model.eval()

            loop = tqdm(enumerate(test_loader))

            for i, d in loop:
                x, y = self._cvt_tensor(d)

                v_acc, v_loss = self._inference_step(x, y, v_acc, v_loss, train=False)

        # Calculate accuracy as the number of correct predictions in the batch
        # divided by the total number of predictions done.
        iteration = i + 1
        v_acc = v_acc / iteration
        v_loss = v_loss / iteration  # Calculate validation loss value

        return v_acc, v_loss, iteration

    def _inference_step(self, x, y, acc, loss, train=True) -> Tuple[float, float]:
        """
        inference step
        calcaulate accumulate acc, loss
        return (acc, loss)
        """
        if train:
            # 변화도(Gradient) 매개변수를 0으로 만들고
            self.optim.zero_grad()

        outputs = self.model(x)
        step_loss = self.criterion(outputs, y)

        if train:
            step_loss.backward()
            self.optim.step()

        # The label with the highest value will be our prediction
        _, preds = torch.max(outputs, 1)
        acc += (preds == y).sum().item() / len(y)
        loss += step_loss.item()

        return acc, loss

    def _print_log(self, message, print_time=True):
        """class print_func for logging"""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            message = "[ " + localtime + " ] " + message
        print(message)
        with open(self.log_file, "a") as f:
            print(message, file=f)

    def _save_history(self, history):
        """save acc, loss history"""
        file_name = Path(
            f'./{str(time.asctime(time.localtime(time.time(())))).replace(" ","_")}_{self.name}_history'
        )

        with open(file_name, "r") as f:
            f.write(str(history))

    def _copy_iter(self) -> Tuple[Iterator, Iterator]:
        """
        copy generator(iterator) using itertools.tee
        return train_generator, test_generator
        """
        train, self.train_loader = tee(self.train_loader)
        test, self.test_loader = tee(self.test_loader)
        return train, test

    def _get_iteration(self) -> Tuple[int, int]:
        """
        get iteration length
        return len(train_generator),len(test_generator)
        """
        train, test = self._copy_iter()
        test_len = len([1 for _ in train])
        train_len = len([1 for _ in test])

        return train_len, test_len


class STGCNTrainer(TorchTrainer):
    def __init__(
        self,
        model,
        epochs,
        train_loader,
        test_loader,
        optim,
        criterion,
        cfg,
        name="torch_trainer",
        log=True,
        log_file="./log.txt",
        dev="cpu",
    ):
        super().__init__(
            model,
            epochs,
            train_loader,
            test_loader,
            optim,
            criterion,
            cfg,
            name,
            log,
            log_file,
            dev,
        )

    def _cvt_tensor(self, d: Tuple[np.ndarray, np.ndarray]):
        x, y = d
        x = torch.from_numpy(x).unsqueeze(-1).float().to(self.dev)
        x = torch.einsum("ntvcm->nctvm", x)
        y = torch.from_numpy(y).type(torch.LongTensor).to(self.dev)
        return x, y


class TestDataGenerator:
    """Data Iterator(Generator) for Model debugging"""

    def __init__(self, input_shape: tuple, batch_size: int, iteration: int = 1):
        """using partial for fetching data when user needs"""
        self.x = [
            partial(np.random.random, (batch_size, *input_shape))
            for _ in range(iteration)
        ]
        self.x = iter(self.x)

        self.y = iter(
            [partial(np.random.randint, 10, size=batch_size) for _ in range(iteration)]
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.x)(), next(self.y)()


if __name__ == "__main__":
    import sys, os
    import torch.optim as optim

    print(os.getcwd())
    sys.path.append(".")
    from slr.model.tgcn import TGCN_v2
    from slr.static.const import TGCN_INOUT_CHANNELS_ver1
    from slr.model.configs.tgcn_config import CFG_100

    device = torch.device("cpu")
    CFG = CFG_100
    model = TGCN_v2(
        3,
        137,
        num_class=100,
        dev=device,
        cfg=CFG,
        in_out_channels=TGCN_INOUT_CHANNELS_ver1,
    ).to(torch.device("cpu"))
    test_trainer = TorchTrainer(
        model=model,
        epochs=CFG.epochs,
        train_loader=TestDataGenerator(
            (150, 137, 3), 16
        ),  # KSLTFRecDataGenerator(CFG.test_file,comp='GZIP',batch_size=CFG.batch_size,channel=CFG.model_args['channel']),
        test_loader=TestDataGenerator(
            (150, 137, 3), 16
        ),  # KSLTFRecDataGenerator(CFG.test_file,comp='GZIP',batch_size=CFG.batch_size,channel=CFG.model_args['channel']),
        optim=optim.Adam,
        criterion=nn.CrossEntropyLoss().float().to(device),
        name="test_trainer",
        dev=device,
        cfg=CFG,
    )
    test_trainer.train()
