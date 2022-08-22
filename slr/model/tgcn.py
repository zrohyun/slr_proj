import torch
import torch.nn as nn


class TGCN(nn.Module):
    def __init__(
        self,
        in_channel,
        num_keypoints,
        in_out_channels,
        num_class,
        dev,
        dropout=0.3,
        kernel_size=9,
        stride=1,
    ):
        super().__init__()
        self.k = kernel_size  # window size(temporal step)
        self.s = stride
        self.d = dev

        out_c = in_out_channels[0][0]
        self.gtcn0 = nn.Conv2d(
            in_channel,
            out_c,
            kernel_size=(self.k, 1),
            padding=(int((self.k - 1) / 2), 0),
            stride=(self.s, 1),
        ).to(dev)

        self.gtcn_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=(self.k, 1),
                    padding=(int((self.k - 1) / 2), 0),
                    stride=(s, 1),
                ).to(dev)
                for in_channel, out_channel, s in in_out_channels
            ]
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        gtcn_out_c = in_out_channels[-1][1]

        if num_class >= 100:
            self.fcl = nn.ModuleList([nn.Linear(gtcn_out_c, num_class)])
        else:
            self.fcl = nn.ModuleList(
                [nn.Linear(gtcn_out_c, 256), nn.Linear(256, num_class)]
            )

        self.fcn = nn.Conv1d(gtcn_out_c, num_class, kernel_size=1).to(dev)
        self._conv_init(self.fcn)

    def forward(self, G):
        B, C, F, V = G.shape
        # 패딩된 행렬은 제외하고 평균 인접행렬 생성
        # 배치 중에 길이가 200이상이라 패딩이 안된 데이터도 있음. 밑에는 안됨.

        A = torch.stack(
            [g[g.sum((1, 2)) != 0].view(-1, F, V).mean(0) for g in G], dim=0
        )

        G = G.permute(0, 2, 1, 3).contiguous()

        # first layer
        x = self.gtcn0(G)

        for layer in self.gtcn_layers:
            B, C, F, V = x.shape
            xa = x.view(B, -1, V).bmm(A).view(B, C, F, V)
            x = layer(xa)
            B, C, F, V = x.shape
            # 현재 F(feature)가 window size만큼을 나타내고 있기 때문에 축의 순서를 바꾸어 batnorm을 수행 행한다.
            x = nn.BatchNorm1d(C * V).to(self.d)(
                x.permute(0, 1, 3, 2).contiguous().view(B, -1, F)
            )
            x = x.view(B, C, V, F).permute(0, 1, 3, 2).contiguous()
            x = self.relu(x)

        x = torch.mean(x, (2, 3))

        for fc in self.fcl:
            x = fc(x)

        return x

    def _conv_init(self, module):
        # he_normal
        n = module.out_channels
        from math import sqrt

        for k in module.kernel_size:
            n *= k
        module.weight.data.normal_(0, sqrt(2.0 / n))
