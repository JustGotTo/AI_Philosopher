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
        self.register_buffer(
            "signs",
            t.randint(
                0, 2,
                (hidden_size,)
            ).float() * 2 - 1
        )
        assert (
                       self.hidden_size &
                       (self.hidden_size - 1)
               ) == 0, "hidden_size must be a power of two"
        self.packed, self.scale, self.amax = None, None, None

    def rotate(self, x):
        x_rot = x * self.signs
        x_rot = self.fwht(x_rot)
        amax = x_rot.abs().max(dim=-1, keepdim=True).values

        return x_rot, amax

    def quantize(self, x):
        xrot, self.amax = self.rotate(x)

        self.scale = (self.amax / 8).clamp(min=1e-8)
        quantized = t.round(xrot / self.scale)
        quantized = quantized.clamp(-8, 7)
        q_unsigned = quantized + 8
        q_unsigned = q_unsigned.to(t.uint8)

        quantized_even = q_unsigned[..., 0::2]
        quantized_odd  = q_unsigned[..., 1::2]

        self.packed = ((quantized_even & 0xF) | ((quantized_odd & 0xF) << 4))

        return self.packed, self.scale, self.amax

    def dequantize(self):

        low = self.packed & 0xF
        high = (self.packed >> 4) & 0xF

        unpacked = t.empty(
            *self.packed.shape[:-1],
            self.packed.shape[-1]*2,
            dtype=self.packed.dtype,
            device=self.packed.device
        )

        unpacked[..., 0::2] = low
        unpacked[..., 1::2] = high #now unpacked is a container for both even and odd entries, which we can use to reconstruct the original data.

        unpacked = unpacked.to(t.int32)
        unpacked = unpacked - 8 #return data to range between -8 and 7

        x_hat_rot = unpacked.float() * self.scale
        x_hat = self.fwht(x_hat_rot)
        x_hat = x_hat * self.signs
        return x_hat

    def fwht(self, x):
        shape = x.shape
        d = shape[-1]
        x = x.reshape(-1, d)
        h = 1
        while h < d:
            x = x.reshape(-1, d // (h * 2), 2, h)
            a = x[:, :, 0, :]
            b = x[:, :, 1, :]
            x = t.stack([a + b, a - b], dim=2)
            h *= 2
        return (x.reshape(shape) / np.sqrt(d))

if __name__ == "__main__":
    pass