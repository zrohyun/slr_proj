import torch
from itertools import tee
from tqdm import tqdm
import time


class TorchTrainer:
    def __init__(
        self,
        model,
        epochs,
        train_loader,
        test_loader,
        optim,
        criterion,
        name="torch_trainer",
        log=True,
        log_file="./log.txt",
        dev="cpu",
    ):

        self.model = model.to(dev)

        self.epochs = epochs

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optim = optim
        self.criterion = criterion

        self.dev = dev

        self.log_file = log_file

        if log:
            self._print_log(f"\n Training {name}")

    def _cvt_tensor(self, d):
        x, y = d
        x = torch.from_numpy(x).unsqueeze(-1).float().to(self.dev)
        y = torch.from_numpy(y).type(torch.LongTensor).to(self.dev)
        return x, y

    def train(self):
        self.model.train()
        training_acc, training_loss = [], []
        val_acc, val_loss = [], []

        for epoch in tqdm(range(self.epochs)):

            # copy generator(iterator) using itertools.tee
            train_loader, self.train_loader = tee(self.train_loader)
            test_loader, self.test_loader = tee(self.test_loader)

            acc, loss, iteration = self.train_step(train_loader)

            training_acc.append(acc)
            training_loss.append(loss)

            self._print_log(
                f"[{epoch + 1}, {iteration :5d}] train_loss: {loss:.3f} train_acc: {acc:.3f}"
            )

            acc, loss, iteration = self.validate(test_loader)
            val_acc.append(acc)
            val_loss.append(loss)

            self._print_log(
                f"[{epoch + 1}, {iteration :5d}] val_loss: {loss:.3f} val_acc: {acc:.3f}"
            )

        return {
            "training_acc": training_acc,
            "training_loss": training_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }

    def train_step(self, train_loader):

        t_acc, t_loss = 0.0, 0.0

        loop = tqdm(enumerate(train_loader))

        for i, d in loop:
            x, y = self._cvt_tensor(d)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optim.step()

            _, preds = torch.max(outputs, 1)
            t_acc += (preds == y).sum().item() / len(y)
            t_loss += loss.item()

            iteration = i + 1

            t_acc = t_acc / iteration
            t_loss = t_loss / iteration

        return t_acc, t_loss, iteration

    def validate(self, test_loader):

        v_acc, v_loss = 0.0, 0.0

        with torch.no_grad():
            self.model.eval()

            loop = tqdm(enumerate(test_loader))
            for i, d in loop:
                x, y = self._cvt_tensor(d)
                outputs = self.model(x)

                loss = self.criterion(outputs, y)

                # The label with the highest value will be our prediction
                _, preds = torch.max(outputs, 1)
                v_acc += (preds == y).sum().item() / len(y)
                v_loss += loss.item()

        # Calculate accuracy as the number of correct predictions in the validation batch
        # divided by the total number of predictions done.
        iteration = i + 1
        v_acc = v_acc / iteration
        v_loss = v_loss / iteration  # Calculate validation loss value

        return v_acc, v_loss, iteration

    def _print_log(self, message, print_time=True):
        # print(str_)
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            message = "[ " + localtime + " ] " + message
        print(message)
        with open("./log.txt", "a") as f:
            print(message, file=f)


class STGCNTrainer(TorchTrainer):
    def __init__(
        self,
        model,
        epochs,
        train_loader,
        test_loader,
        optim,
        criterion,
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
            name,
            log,
            log_file,
            dev,
        )

    def _cvt_tensor(self, d):
        x, y = d
        x = torch.from_numpy(x).unsqueeze(-1).float().to(self.dev)
        x = torch.einsum("ntvcm->nctvm", x)
        y = torch.from_numpy(y).type(torch.LongTensor).to(self.dev)
        return x, y
