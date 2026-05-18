from BytePairEncoder import BytePairEncoder

import torch.nn as nn
import torch as t
import numpy as np

class Embedding(nn.Module):
    def __init__(self, prompt, vocab_size=1000, embedding_dim=128, hidden_size=512, output_size=512):
        super().__init__()
        self.prompt = BytePairEncoder(prompt=prompt).forward(prompt)
        self.embedding_dim = embedding_dim #same as input_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, input):
        return self.embedding(input) #Simple embedding mechanism


class AddNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(t.ones(hidden_size))

    def forward(self, residual, x):
        x = residual + x
        rms = t.sqrt(t.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class LinearPostAttention(nn.Module):
    def __init__(self, hidden_size, output_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(t.ones(output_size)*0.9)
        self.bias = nn.Parameter(t.zeros(output_size))

    def forward(self, x):
        return self.weight * x + self.bias

