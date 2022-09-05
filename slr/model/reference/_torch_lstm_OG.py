"""
pytorch LSTM paper Implementation Original version
from https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM

Refered for Implementing Graph LSTM
"""

import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, List, Tuple

# pytorch Implementation Original
class LSTMCell(nn.RNNCellBase):
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
        assert input.dim() in (
            1,
            2,
        ), f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = _VF.lstm_cell(
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


def lstm_cell(
    input: Tensor,
    hidden: Tuple[Tensor, Tensor],
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy
