from BytePairEncoder import BytePairEncoder

import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, prompt, vocab_size=1000, embedding_dim=64, hidden_size=384, output_size=384):
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

