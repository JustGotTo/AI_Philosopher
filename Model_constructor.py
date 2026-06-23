from Backend import Embedding, AddNorm, WordFeedForward
from Backend import LinearPostAttention, SentenceFeedForward, PhraseFeedForward, AdaptiveMultiheadMaskedAttention, BeliefsLayer

from PolarQuant import PolarQuant
from BytePairEncoder import BytePairEncoder

import torch.nn as nn
import torch as t
import numpy as np

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size, eps, prompt=""):
        super().__init__()
        self.BPE = BytePairEncoder(prompt=prompt, vocab_size=)
        self.embedding = Embedding(prompt=prompt, vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size)
        self.linear = LinearPostAttention(output_size=embedding_dim, eps=self.eps)
        self.addnorm = AddNorm(embedding_dim, eps=self.eps)
        self.feedforward = SentenceFeedForward(hidden_size=hidden_size, output_size=embedding_dim)
        self.wordfeed = WordFeedForward(hidden_size=hidden_size, output_size=embedding_dim, shrinked_size=embedding_dim//2)
        self.phrasefeed = PhraseFeedForward(hidden_size=hidden_size, output_size=embedding_dim, shrinked_size=embedding_dim//2)
        self.polar = PolarQuant(hidden_size=hidden_size)
        #Attention will be called individually in the forward pass, in order to adjust the mask window size and batch size.
        #Beliefs layer is applied internally so no need to call it.

        self.prompt = prompt
        self.vocab_size = vocab_size
        self.eps = eps
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """Upon entering the forward pass, """
        self.vocab_size = x.shape[1]
        x = self.embedding(x) #Creating word embeddings
        x = self.polar.quantize(x)
        x = self.addnorm.forward(x, x) #Normalisation before attention layer


