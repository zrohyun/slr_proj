from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from pathlib import Path
from itertools import tee
from typing import Iterator, List, Tuple
from tqdm import tqdm
import time


class TorchTrainer:
    """
    Torch model trainer class

    optimizer input like 'optim = optim.Adam'
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
        device: torch.device = torch.device("cpu"),
        save_history: bool = True,
    ):

        self.model = model.to(device)

        self.epochs = epochs

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optim = optim(self.model.parameters())
        self.criterion = criterion.to(device)

        self.device = device
        self.cfg = cfg
        self.name = name

        self.log_file = log_file
        self.save_history = save_history

        if log:
            self._print_log(f"\n Training {self.name}\n")
            self._print_log(f"{self.model}", print_time=False)

    def _cvt_tensor(self, d) -> Tuple[Tensor, Tensor]:
        """convert x,y data to tensor and allocate to device"""
        x, y = d
        if isinstance(x, Tensor):
            x = x.clone().detach().float().to(self.device)
            y = y.clone().detach().type(torch.LongTensor).to(self.device)
        else:
            x = torch.from_numpy(np.array(x)).float().to(self.device)
            y = torch.from_numpy(np.array(y)).type(torch.LongTensor).to(self.device)

        return x, y

    def run(self, val_interval=10, validation=True) -> dict:
        """
        Trainer Sequence
        1. run training step
        2. run validation step( run at validation interval )
        3. repeat for epochs
        return training_history
        """
        training_acc, training_loss = [], []
        val_acc, val_loss = [], []

        run_time = time.time()
        for epoch in tqdm(range(self.epochs)):
            t_time = time.time()

            # train step
            train_loader, test_loader = self._copy_iter()
            acc, loss, iteration = self.train_step(train_loader)

            training_acc.append(acc)
            training_loss.append(loss)

            self._print_log(
                f"[{epoch + 1}, {iteration :5d}] train_loss: {loss:.3f} train_acc: {acc:.3f} "
                + f"train_time: {time.time()-t_time:.1f}"
            )

            # validation step
            if validation:
                # run At val_interval
                if (epoch % val_interval == val_interval - 1) or (
                    epoch == self.epochs - 1
                ):
                    v_time = time.time()
                    acc, loss, iteration = self.validate(test_loader)
                    val_acc.append(acc)
                    val_loss.append(loss)

                    self._print_log(
                        f"[{epoch + 1}, {iteration :5d}] val_loss: {loss:.3f} val_acc: {acc:.3f} "
                        + f"val_time: {time.time()-v_time:.1f}"
                    )

        history = {
            "training_acc": training_acc,
            "training_loss": training_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }

        if self.save_history:
            self._save_history(history)

        self._print_log(f"TOTAL TRAINING TIME: {time.time() - run_time:.1f}\n")

        return history

    def train_step(self, train_loader) -> Tuple[float, float, float]:
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

    def validate(self, test_loader) -> Tuple[float, float, float]:
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

        outputs = self.model(x.clone().detach())
        step_loss = self.criterion(outputs, y.clone().detach())

        if train:
            step_loss.backward()
            self.optim.step()

        # The label with the highest value will be our prediction
        _, preds = torch.max(outputs, 1)
        acc += (preds == y).sum().item() / len(y)
        loss += step_loss.item()

        return acc, loss

    def _print_log(self, message, print_time=True) -> None:
        """class print_func for logging"""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            message = "[ " + localtime + " ] " + message
        print(message)
        with open(self.log_file, "a") as f:
            print(message, file=f)

    def _save_history(self, history) -> None:
        """save acc, loss history"""
        localtime = str(time.asctime(time.localtime(time.time()))).replace(" ", "_")
        if sys.platform == "win32":
            localtime = localtime.replace(":", "_")
        file_name = Path(f"./{localtime}_{self.name}_history")

        with open(file_name, "w") as f:
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
