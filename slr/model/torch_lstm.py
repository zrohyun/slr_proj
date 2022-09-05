import sys, os

sys.path.append(".")

import torch
import torch.nn as nn

from typing import Tuple, List

from slr.utils.utils import get_device
from scipy.spatial.distance import cdist


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class GCLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dev, bias: bool = True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        A: torch.Tensor
            Adjacency Matrix()
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W_x = nn.Linear(input_dim, 4 * hidden_dim, bias=bias).to(dev)
        self.W_h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias).to(dev)
        self.dev = dev

    def forward(self, x, A, hidden) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = hidden
        h = h.to(self.dev)
        c = c.to(self.dev)
        gates = self.W_x(A @ x) + self.W_h(A @ h)

        gates = gates.squeeze()

        i, f, o, u = gates.chunk(4, -1)

        i = torch.sigmoid(i)  # input gate for update cell_state
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        u = torch.tanh(u)  # update cell_state with new X

        c = f * c + i * u

        h = o * torch.tanh(c)

        return (h, c)


class GraphLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, class_num, dev, dropout=None, **kwargs):
        super().__init__()
        self.layer = GCLSTMCell(in_dim, out_dim, dev)
        self.fc = nn.Linear(out_dim * 137, class_num)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.class_num = class_num
        self.dev = dev
        self.kwargs = kwargs

    def forward(self, x, A=None):
        B, T, V, F = x.shape
        h = torch.zeros((V, self.out_dim)).to(self.dev)
        s = torch.zeros((V, self.out_dim)).to(self.dev)
        A = A.to(self.dev) if A is not None else self._adj_mat(x)
        outs = []
        for t in range(T):
            h, s = self.layer(x[:, t], A[:, t], (h, s))
            outs.append(h)
        outs = torch.stack(outs, dim=1).to(self.dev)
        outs = outs.mean(dim=1)
        outs = nn.Flatten()(outs)
        return torch.sigmoid(self.fc(outs))

    def _adj_mat(self, X: torch.Tensor, delta=0.5):
        """get sequential Adjacency(affinity) Matrix"""
        batch_A = []
        batch_size, window_size = X.shape[:2]

        def _gaussian_kernel(X, delta=delta):
            """exp( dist(x,y) / 2 * (delta^2) )"""
            return torch.exp(
                torch.tensor(-cdist(X.cpu(), X.cpu()) / (2.0 * delta**2))
            )

        for b in range(batch_size):
            A = []
            for w in range(window_size):
                A.append(_gaussian_kernel(X[b, w]))
            batch_A.append(torch.stack(A))

        return torch.stack(batch_A).float().to(self.dev)


# simple LSTMCell
class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        A: torch.Tensor
            Adjacency Matrix()
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W_x = nn.Linear(input_dim, 4 * hidden_dim, bias=bias)
        self.W_h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)

    def forward(self, x, hidden) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = hidden
        A = Tensor(torch.ones(137, 137))

        gates = self.W_x(x) + self.W_h(h)
        gates = gates.squeeze()

        i, f, o, u = gates.chunk(4, -1)

        i = torch.sigmoid(i)  # input gate for update cell_state
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        u = torch.tanh(u)  # update cell_state with new X

        c = f * c + i * u

        h = o * torch.tanh(c)

        return (h, c)


def signlanguage_data_shape_test():
    # sign langauge data shape testing
    B, in_, out_, V = 32, 3, 8, 137
    key = torch.ones((B, V, in_))
    hid = (torch.ones((B, V, out_)), torch.ones((B, V, out_)))
    bias = torch.ones(out_ * 4)
    W = torch.ones((out_ * 4, in_))
    Wh = torch.ones((out_ * 4, out_))
    print(key.shape, W.shape)
    graph_lstm_cell(key, hid, W, Wh, bias, bias)[0].shape


# dev = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

dev = cfg.model_args["dev"]
GraphLSTM(3, 4, 100, dev=dev).to(dev)(
    torch.ones((32, 150, 137, 3)).to(dev)
).shape  # , torch.ones((32,150,137,137))).shape

keypoint_sample_input = torch.ones(
    (32, 150, 137, 3)
)  # Batch, temporal_length, keypoint_vertex, features
A = torch.ones((32, 150, 137, 137))  # affinity matrix

if __name__ == "__main__":
    device = get_device()
    in_dim, out_dim = 3, 4
    convlstmparmas = {
        "input_tensor": torch.ones((1, in_dim, 100, 100)),
        "cur_state": (
            torch.ones(1, out_dim, 100, 100),
            torch.ones(1, out_dim, 100, 100),
        ),
    }
    assert (ConvLSTMCell(in_dim, out_dim, (1, 1), True)(**convlstmparmas))[0].shape == (
        1,
        out_dim,
        100,
        100,
    )
    graphconvlstmparmas = {
        "x": torch.ones((1, 1, 137, in_dim)),
        "A": torch.ones((137, 137)),
        "hidden": (torch.ones(1, 1, 137, out_dim), torch.ones(1, 1, 137, out_dim)),
    }
    assert (GCLSTMCell(3, 4, dev=device)(**graphconvlstmparmas))[0].shape == (
        1,
        1,
        137,
        out_dim,
    )
