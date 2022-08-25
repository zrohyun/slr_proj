import torch
import torch.nn as nn
from torchsummary import summary
import sys

from scipy.spatial.distance import cdist

sys.path.append(".")


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
            xa = x.view(B, -1, V).bmm(A).view(B, C, F, V).contiguous()
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


class TGCN_v2(TGCN):
    """
    Temporal GCN ver2
    reducing calculate cost
    """

    def __init__(
        self,
        in_channel,
        num_keypoints,
        in_out_channels,
        num_class,
        dev,
        cfg,
        dropout=0.3,
        kernel_size=9,
        stride=1,
    ):
        super().__init__(
            in_channel,
            num_keypoints,
            in_out_channels,
            num_class,
            dev,
            dropout,
            kernel_size,
            stride,
        )
        self.cfg = cfg
        self.data_bn = nn.BatchNorm1d(in_channel * num_keypoints)

    def forward(self, X: torch.Tensor):
        """
        B : batch, T: seq length(window_size), V(vertex, num_keypoints), C(cannel, features)
        """
        B, T, V, C = X.shape

        A = self._adj_mat(X)  # shape B,T,V,V

        # batch norm
        X = torch.einsum("btvc->bvct", X).contiguous().view(B, -1, T)
        X = self.data_bn(X)
        X = torch.einsum("bvct->bctv", X.view(B, V, C, T)).contiguous()  # shape B,C,T,V

        # first layer
        # rotate axis for considering F(feature) as T(Window_size)
        X = self.gtcn0(X)  # swap F and C axis

        stride = self.gtcn0.stride[0]
        for layer in self.gtcn_layers:
            if stride != 1:
                A = self._reduce_window_size(A, stride)
            X = torch.einsum("bctv->btcv", X).contiguous()
            # AX = (X @ A).einsum("btvc->bcvt").contiguous()  # shape = B, C, V, T
            AX = torch.einsum(
                "btcv->bctv", X @ A
            ).contiguous()  # shape B T C V -> B C T V
            X = layer(AX)
            # 현재 T가 window size만큼을 나타내고 있기 때문에 축의 순서를 바꾸어 batnorm을 수행 행한다.
            # x = nn.BatchNorm1d(T * V).to(self.d)(
            #     x.permute(0, 1, 3, 2).contiguous().view(B, -1, C)
            # )
            # x = x.view(B, T, V, C).permute(0, 1, 3, 2).contiguous()
            X = self.relu(X)
            stride = layer.stride[0]

        X = torch.mean(X, (2, 3))

        for fc in self.fcl:
            X = fc(X)

        return X

    def _adj_mat(self, X: torch.Tensor, delta=0.5):
        """get sequential Adjacency(affinity) Matrix"""
        batch_A = []

        def _gaussian_kernel(X, delta=delta):
            """exp( dist(x,y) / 2 * (delta^2) )"""
            return torch.exp(torch.tensor(-cdist(X, X) / (2.0 * delta**2)))

        for b in range(self.cfg.batch_size):
            A = []
            for w in range(self.cfg.model_args["window_size"]):
                A.append(_gaussian_kernel(X[b, w]))
            batch_A.append(torch.stack(A))

        return torch.stack(batch_A).float().to(self.d)

    def _reduce_window_size(self, A, stride):
        a = []
        for i in range(0, A.shape[1], stride):
            a.append(A[:, i : i + stride, :, :].mean(axis=1))
        A = torch.stack(a, axis=1)
        return A


class CFG:
    debug = True
    model_name = "st_gcn.Model"
    test_file = "gzip_test_with_preprocess.tfrec"
    train_file = "gzip_train_with_preprocess.tfrec"
    model_args = {
        "channel": 3,
        "num_class": 100,
        "window_size": 150,
        "num_point": 137,
    }
    epochs = 100
    batch_size = 2
    seed = 42
    T_max = 3
    lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-3


if CFG.debug:
    CFG.epochs = 1

if __name__ == "__main__":
    import slr.static.const as const

    model = TGCN(
        137, 137, const.TGCN_INOUT_CHANNELS_ver1, num_class=10, dev=torch.device("cpu")
    )

    model = TGCN_v2(
        3,
        137,
        const.TGCN_INOUT_CHANNELS_ver1,
        num_class=10,
        dev=torch.device("cpu"),
        cfg=CFG,
    )

    summary(model, (150, 137, 3))
