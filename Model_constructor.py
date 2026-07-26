from Backend import Embedding, AddNorm, WordFeedForward
from Backend import LinearPostAttention, SentenceFeedForward, PhraseFeedForward, AdaptiveMultiheadMaskedAttention
from BytePairEncoder import BytePairEncoder

from PolarQuant import PolarQuant

import torch.nn as nn
import torch as t
import re

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim, eps=1e-6, vocab_size=25000, prompt=""):
        super().__init__()
        self.prompt = prompt
        self.vocab_size = vocab_size
        self.eps = eps
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.sentences = [s.strip() for s in re.findall(r'[^.!?]*\.', self.prompt)]

        self.embedding = Embedding(prompt=prompt, vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.linear = LinearPostAttention(output_size=embedding_dim)
        self.addnorm = AddNorm(embedding_dim, eps=self.eps)
        self.feedforward = SentenceFeedForward(hidden_size=hidden_size, output_size=hidden_size)
        self.wordfeed = WordFeedForward(hidden_size=hidden_size, output_size=embedding_dim)
        self.phrasefeed = PhraseFeedForward(hidden_size=hidden_size, output_size=embedding_dim)
        #Attention will be called individually in the forward pass, in order to adjust the mask window size and batch size to create hierarchial style attention.
        #Beliefs layer is applied internally so no need to call it.


    def forward(self, x):
        """Upon entering the forward pass, x is a list of encoded tokens with embeddings"""
        #3 level Hierarchial attention
        #First applying sentence-level attention:
        #DO: complete the adaptive step

        x = self.addnorm.forward(x,x)  # Normalisation before attention layer
        x = AdaptiveMultiheadMaskedAttention(batch_size=128, full_size=x.shape[1], mask_window_size=((2**len(self.sentences))%33), embedding_size=self.embedding_dim, prompt=self.prompt).forward(x)
        x = self.linear.forward(x)
        x = self.feedforward.forward(x)
        #Phrase-level attention
        x = self.addnorm.forward(x, x)
        x = AdaptiveMultiheadMaskedAttention(batch_size=128, full_size=x.shape[1], mask_window_size=(2**len(self.sentences))%15, embedding_size=self.embedding_dim, prompt=self.prompt).forward(x)
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
        self.model = nn.ModuleList([Decoder(hidden_size=hidden_size, embedding_dim=embedding_dim, eps=self.eps) for _ in range(6)])


    def forward(self, x, device=None):
        if device is None:
            device = next(self.parameters()).device
        # We feed in the prompt, which is then converted to tokens by the system
        x = self.encoder.forward(x)
        x = self.embedding.forward(x)
        x = self.quant.quantize(x)

        for elem in self.model:
            x = elem(x)

        x = self.quant.dequantize()

        x = nn.Linear(int(self.embedding_dim), int(self.hidden_size))

        return x

    @t.no_grad()
    def generate(self, input_ids, device=None):
        if device is None:
            input_ids = input_ids.device

        vocab = BytePairEncoder.vocab
        output = self.forward(input_ids)
        text_output = ""
        for id in output.shape[0]:
            text_output += vocab.ito[id] + " "

        return text_output