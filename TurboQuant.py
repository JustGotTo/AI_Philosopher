import torch as t
import numpy as np
from random import seed

from torch import nn

rot_random_seed = 1234
seed(rot_random_seed)
#ROLE of TurboQuant - Perform the quantization of data in 4 bits, spreading it in 16 different levels. It dramatically reduces the KV cache consumption, which is vital for my SLM.
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

        scale = (xmax - xmin).clamp(min=1e-8) / 15
        quantized = t.round((x - xmin) / scale)
        quantized = quantized.clamp(0,15)
        return q.to(t.uint8), scale, xmin

    def dequantize(self,x):
        quantized, scale, xmin = self.quantize(x)

        x_hat_rot = quantized.float() * scale + xmin
        x_hat = t.matmul(x_hat_rot, self.R.T)
        return x_hat

def fwht(t):




