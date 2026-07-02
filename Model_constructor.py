from Backend import Embedding, AddNorm, WordFeedForward
from Backend import LinearPostAttention, SentenceFeedForward, PhraseFeedForward, AdaptiveMultiheadMaskedAttention
from BytePairEncoder import BytePairEncoder

from PolarQuant import PolarQuant

import torch.nn as nn
import torch as t

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim, eps, vocab_size=25000, prompt=""):
        super().__init__()
        self.embedding = Embedding(prompt=prompt, vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.linear = LinearPostAttention(output_size=embedding_dim, eps=self.eps)
        self.addnorm = AddNorm(embedding_dim, eps=self.eps)
        self.feedforward = SentenceFeedForward(hidden_size=hidden_size, output_size=hidden_size)
        self.wordfeed = WordFeedForward(hidden_size=hidden_size, output_size=embedding_dim)
        self.phrasefeed = PhraseFeedForward(hidden_size=hidden_size, output_size=embedding_dim)
        #Attention will be called individually in the forward pass, in order to adjust the mask window size and batch size to create hierarchial style attention.
        #Beliefs layer is applied internally so no need to call it.

        self.prompt = prompt
        self.vocab_size = vocab_size
        self.eps = eps
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """Upon entering the forward pass, x is a list of encoded tokens with embeddings"""
        #3 level Hierarchial attention

        #First applying sentence-level attention:
        x = self.addnorm.forward(x,x)  # Normalisation before attention layer
        x = AdaptiveMultiheadMaskedAttention(batch_size=128, full_size=x.shape[1], mask_window_size=25, embedding_size=self.embedding_dim, prompt=self.prompt).forward(x)
        x = self.linear.forward(x)
        x = self.feedforward.forward(x)
        #Phrase-level attention
        x = self.addnorm.forward(x, x)
        x = AdaptiveMultiheadMaskedAttention(batch_size=128, full_size=x.shape[1], mask_window_size=5, embedding_size=self.embedding_dim, prompt=self.prompt).forward(x)
        x = self.linear.forward(x)
        x = self.phrasefeed.forward(x)
        #Word-level attention
        x = self.addnorm.forward(x, x)
        x = AdaptiveMultiheadMaskedAttention(batch_size=128, full_size=x.shape[1], mask_window_size=1, embedding_size=self.embedding_dim, prompt=self.prompt).forward(x)
        x = self.linear.forward(x)
        x = self.wordfeed.forward(x)

        return x

class SLModel(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size=25000, prompt=""):
        super().__init__()
        self.eps = 1e-6
        self.encoder = BytePairEncoder(prompt=prompt, vocab_size=vocab_size, input_size=embedding_dim, hidden_size=hidden_size, output_size=hidden_size)
        self.embedding = Embedding(prompt=prompt, vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.quant = PolarQuant(hidden_size=hidden_size)
        self.model = nn.ModuleList([Decoder(hidden_size=hidden_size, embedding_dim=embedding_dim, eps=self.eps) for _ in range(5)])

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.embedding.forward(x)
        x = self.quant.quantize(x)
        x = self.model(x)
        x = self.quant.dequantize()

        return x



