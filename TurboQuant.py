import torch as t
import numpy as np
from random import seed

from torch import nn

rot_random_seed = 1234
seed(rot_random_seed)
#ROLE of TurboQuant - Perform the quantization of data in 4 bits, spreading it in 16 different levels. It dramatically reduces the KV cache consumption, which is vital for my SLM.
class TurboQuant(nn.Module):
    def __init__(self, hidden_size, embedding_dim=512):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embedding_dim = embedding_dim
        self.signs = t.randint(
            0, 2,
            (hidden_size,)
        ).float() * 2 - 1
        self.quantized = t.randint(0, 16, (hidden_size,)).float()
        self.copy = t.randint(0, 1, (hidden_size,)).float()
        assert (
                       self.hidden_size &
                       (self.hidden_size - 1)
               ) == 0

    def rotate(self, x):
        x_rot = x * self.signs

        x_rot = self.fwht(x_rot)
        self.copy = x_rot.clone()

        amax = x_rot.abs().max(dim=-1, keepdim=True).values
        return x_rot, amax

    def quantize(self, x):
        xrot, amax = self.rotate(x)

        scale = (amax / 7).clamp(min=1e-8)
        quantized = t.round(xrot / scale)
        quantized = quantized.clamp(-7, 7)
        q_unsigned = quantized + 8
        q_unsigned = q_unsigned.to(t.int8)

        quantized_even = q_unsigned[..., 0::2]
        quantized_odd  = q_unsigned[..., 1::2]

        packed = ((quantized_even & 0xF) | ((quantized_odd & 0xF) << 4))

        return packed, scale, amax

    def dequantize(self,x):
        quantized, scale, xmin = self.quantize(x)

        x_hat_rot = quantized.float() * scale
        x_hat = t.matmul(x_hat_rot, self.fwht(quantized))
        return x_hat

    def fwht(self, x):
        h = 1

        while h < x.shape[-1]:
            x = x.reshape(
                *x.shape[:-1], #first dimension
                -1,             #residue dimensions
                h*2
            )

            l = x[..., :h]
            r = x[..., h:]

            x = t.cat([l+r, l-r], dim=-1)

            h *= 2
        x = x / np.sqrt(self.hidden_size)

        return x