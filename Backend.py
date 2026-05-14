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
    def __init__(self, input):
        super().__init__()
        self.input = input

    def forward(self, input):
        input = np.array(self.input + input)
        rms = np.sqrt(np.mean(np.square(input)))

