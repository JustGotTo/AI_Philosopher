from BytePairEncoder import BytePairEncoder

import torch.nn as nn
import torch as t
import numpy as np
import re


class Embedding(nn.Module):
    def __init__(self, prompt, vocab_size=1000, embedding_dim=512, hidden_size=512, output_size=512):
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
    def __init__(self, output_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(t.ones(output_size)*0.9)
        self.bias = nn.Parameter(t.zeros(output_size))

    def forward(self, x):
        return self.weight * x + self.bias

class SentenceFeedForward(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.norm = AddNorm(hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x, x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class PhraseFeedForward(nn.Module):
    def __init__(self, hidden_size, output_size, shrinked_size):
        super().__init__()
        self.shrinked_size = hidden_size//8
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(hidden_size, shrinked_size)
        self.norm = AddNorm(output_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x, x)
        x = self.act(x)
        x = self.norm(x, x)
        x = self.dropout(x)
        return x

class WordFeedForward(nn.Module):
    def __init__(self, hidden_size, output_size, shrinked_size):
        super().__init__()
        self.shrinked_size = shrinked_size
        self.linear1 = nn.Linear(shrinked_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.norm = AddNorm(output_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.norm(x, x)
        x = self.dropout(x)
        return x

class AdaptiveMultiheadMaskedAttention(nn.Module):
    def __init__(self, batch_size:int, full_size, mask_window_size, embedding_size=512, eps=1e-8, prompt = None): #full_size - size of prompt before splitting the tokens into batches
        super().__init__()
        self.eps = eps
        self.mask_window_size = mask_window_size if mask_window_size < batch_size else batch_size//2
        self.num_heads = t.floor((mask_window_size + full_size)/(embedding_size + 1))
        self.batch_size = batch_size
        self.t_beliefs = BeliefsLayer(full_size, embedding_size, window_size=self.mask_window_size)
        self.t_context = AddNorm(self.embedding_size) #tensor for context
        self.mask = self.createMask() #creates a mask of the batch_size x batch_size matrix
        self.prompt = prompt
        self.dph = self.embedding_size//self.num_heads

    def createMask(self):
        mask = t.ones(self.batch_size, self.batch_size)

        window_size = min(self.mask_window_size, self.batch_size)

        for i in range(self.batch_size):
            for j in range(window_size):
                column = (i + j) % self.batch_size
                mask[i, column] = 0 #blocks chunks of sentences from seeing each other

        return mask

    def split_batch(self, x, prompt):
        chunk_size = 256
        sliding_window = 64
        self.batch_size = chunk_size+sliding_window
        chunks = t.empty(prompt//chunk_size)
        i=0
        while i < prompt.length:
            if i == 0:
                chunks[0] = x[0:chunk_size]
            else:
                try:
                    chunks[i] = x[i+(chunk_size*i)-sliding_window:i+(chunk_size*i)+sliding_window]
                except:
                    chunks[i] = x[i+(chunk_size*i)-sliding_window:]
            i += 1

        return chunks



    def getMeanSentenceLength(self, prompt):
        sentences = [s.strip() for s in re.findall(r'[^.!?]*\.', prompt)]
        return np.mean([len(sentence) for sentence in sentences])


    def forward(self, x):
        x = self.split_batch(x, self.prompt) #Splitting prompt into chunks
        num_heads = self.num_heads
        dph = self.dph #dims per head
        w_q = t.tensor((self.embedding_size, dph))
        w_k = t.tensor((self.embedding_size, dph))
        w_v = t.tensor((self.embedding_size, dph))
        w_b = t.tensor((self.embedding_size, dph))
        result = t.zeros((self.batch_size, self.embedding_size))

        for chunk in x:
            chunk_beliefs = self.t_beliefs(chunk) #beliefs in the chunk
            Q = t.matmul(chunk, w_q)
            K = t.matmul(chunk, w_k)
            V = t.matmul(chunk, w_v)
            B = t.matmul(chunk_beliefs, w_b)
            t_context = self.t_context()
            chunk_context = t.matmul(self.t_context, w_b)
            chunk_context = chunk_context.repeat(num_heads, 1, 1)
            chunk_context = chunk_context.view(self.batch_size, num_heads, dph)
            chunk_context = chunk_context.permute(1, 0, 2)
            chunk_context = chunk_context.repeat(self.batch_size, 1, 1)
            chunk_context = chunk_context.view(self.batch_size, num_heads, dph)







class BeliefsLayer(nn.Module):
    #Beliefs layer is a simple attention mechanism that is used to calculate beliefs of the agent.
    def __init__(self, hidden_size, output_size, window_size=1, embedding_size=512, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.window_size = window_size
        self.attention = nn.MultiheadAttention

    def forward(self, x):
        #Plan - run the entire prompt through standard attention and then apply Top-k algorithm to achieve the most significant neurons.
        #Idea - beliefs = attention(x, x, x)
        #         beliefs = 0.99*beliefs - (1-0.99)*topk(fed_tensor)
        x = self.attention(x, x, x) #a starting tensor
        def topk(x, k):
            #x - tensor to be sorted
            #k - number of top elements to be returned
            values, indices = t.topk(x, k, dim=-1)

            output = t.zeros_like(x) #Clears the tensor from leftover falues.
            output.scatter_(1, indices, values) #Fills the tensor with Top-k values.

            return output


        a = 0.99
        beliefs = t.mul(a,x) - (1-a)*topk(x, self.window_size) #Significance of window_size is determined by the size of the prompt
        return beliefs
