import torch as t
import numpy as np
from random import seed

from torch import nn

rot_random_seed = 1234
seed(rot_random_seed)

class TurboQuant(nn.Module):
    def __init__(self, hidden_size, embedding_dim=386):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embedding_dim = embedding_dim
        self.R = t.randn(hidden_size, hidden_size)

    def rotate(self, x):
        self.R, _ = t.linalg.qr(self.R)

        x_rot = t.matmul(x, self.R)
        xmin = x_rot.min(dim=-1, keepdim=True).values
        xmax = x_rot.max(dim=-1, keepdim=True).values
        return x_rot, xmin, xmax

    def quantize(self, x):
        x, xmin, xmax = self.rotate(x)

        scale = (xmax - xmin) / 15
        q = t.round((x - xmin) / scale)

        diff = t.abs(
            x.unsqueeze(-1) - q
        )
        quantized = t.argmin(diff, dim=-1)
        return quantized, levels

    def dequantize(self,x):
        quantized, levels = self.quantize(x)

        x_hat_rot = levels[quantized]
        x_hat = t.matmul(x_hat_rot, self.R.T)
        return x_hat




