import torch
from torch import Tensor
import torch.nn as nn
from torch import Tensor

from typing import Tuple


class GraphLSTMCell(nn.RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)` or `(input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial hidden state
        - **c_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial cell state

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next hidden state
        - **c_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next cell state

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.LSTMCell(10, 20) # (input_size, hidden_size)
        >>> input = torch.randn(2, 3, 10) # (time_steps, batch, input_size)
        >>> hx = torch.randn(3, 20) # (batch, hidden_size)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(input.size()[0]):
                hx, cx = rnn(input[i], (hx, cx))
                output.append(hx)
        >>> output = torch.stack(output, dim=0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        is_batched = input.dim() == 3
        if not is_batched:
            input = input.unsqueeze(0)

        B, V, F = input.shape

        if hx is None:
            zeros = torch.zeros(
                input.size(0),
                V,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        # print(hx[0].shape,hx[1].shape)
        # ret = torch.ones((1,2,3))
        ret = graph_lstm_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret


def graph_lstm_cell(
    input: Tensor,
    hidden: Tuple[Tensor, Tensor],
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = torch.matmul(input, w_ih.t()) + torch.matmul(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, -1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def check_graphlstmcell():
    # lstm weight shape
    a = GraphLSTMCell(10, 20)  # work like same with simple lstm
    print(
        "lstm h,i weight:", a.weight_ih.t().shape, a.weight_hh.t().shape
    )  # input,hidden weight shape wi(in, out*4), wh(out, out*4)
    print("lstm bias shape", a.bias_ih.shape)
    print(
        "lstm shape", GraphLSTMCell(10, 20)(torch.ones((30, 10)))[0].shape
    )  # input shape (batch, features)

    # lstm weight shape test extened for graph
    a = GraphLSTMCell(3, 30)(
        torch.ones(32, 137, 3)
    )  # input shape (batch, vertex, features)
    print("graph lstm shape test", a[0].shape)


def check_torch_op():
    # matmul for 2d matrix
    q = torch.ones((30, 10))
    print("mm shape:", torch.mm(torch.ones((30, 10)), torch.ones(80, 10).T).shape)
    print(
        "matmul shape:",
        torch.matmul(torch.ones((30, 1, 10)), torch.ones((10, 80))).shape,
    )
    print(
        "bmm shape:", torch.bmm(torch.ones((30, 1, 10)), torch.ones((30, 10, 80))).shape
    )
    print(
        ".t(), .T comparision:",
        torch.equal(torch.ones((80, 10)).T, torch.ones((80, 10)).t()),
    )
    print(id(q.t()) == id(q.T), id(q.T) == id(q))


class GraphLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        class_num,
        device=None,
        dtype=None,
        dropout=None,
        **kwargs
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer = GraphLSTMCell(input_size, hidden_size, **factory_kwargs)

        # lstm module stacking version testing
        # self.layer = nn.ModuleList([GraphLSTMCell(input_size, hidden_size, **factory_kwargs) for i in range(9)])

        self.fc = nn.Linear(hidden_size * 137, class_num, **factory_kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.kwargs = kwargs

    def forward(self, X: Tensor, A=None):
        factory_kwargs = {"device": X.device, "dtype": X.dtype}
        B, T, V, F = X.shape
        zero = torch.zeros((B, V, self.hidden_size), **factory_kwargs)
        h, c = zero, zero
        # print(h.requires_grad, s.requires_grad)
        if A is not None:
            A = torch.tensor(A.clone().detach(), requires_grad=False, **factory_kwargs)
        else:
            A = self._adj_mat(X.detach().clone(), **factory_kwargs)
            # A = A.clone().detach()

        # for testing
        # A = torch.ones((B,T,V,V),,requires_grad=False, **factory_kwargs)

        outs = []
        for t in range(T):
            a = A[:, t]
            # print(a.shape)
            input_X = torch.bmm(a, X[:, t])

            if t % 50 == 49:
                h, c = zero, zero

            h = torch.bmm(a, h)  # a @ h
            h, c = self.layer(input_X, (h, c))
            outs.append(h)

        outs = torch.stack(outs, dim=1)

        outs = outs.mean(dim=1)

        outs = nn.Flatten()(outs)

        return self.fc(outs)

    def _adj_mat(self, X: Tensor, delta=0.5, device=None, dtype=None):
        """get sequential Adjacency(affinity) Matrix"""
        factory_kwargs = {"device": X.device, "dtype": X.dtype}
        batch_A = []
        batch_size, window_size = X.shape[:2]

        def _gaussian_kernel(X, delta=delta):
            """exp( dist(x,y) / 2 * (delta^2) )"""
            # return torch.exp(
            #     torch.tensor(-torch.cdist(X, X) / (2.0 * delta**2), **factory_kwargs)
            # )
            return torch.exp(-torch.cdist(X, X) / (2.0 * delta**2)).to(
                **factory_kwargs
            )

        for b in range(batch_size):
            A = []
            for w in range(window_size):
                A.append(_gaussian_kernel(X[b, w]))
            batch_A.append(torch.stack(A))

        return torch.stack(batch_A)


if __name__ == "__main__":
    pass
