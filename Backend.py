from BytePairEncoder import BytePairEncoder

import torch.nn as nn
import torch as t
import numpy as np
import re


class Embedding(nn.Module):
    def __init__(self, prompt, vocab_size=25000, embedding_dim=512):
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
    def __init__(self, output_size):
        super().__init__()
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
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
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
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
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
    def __init__(self, batch_size:int, full_size, mask_window_size, embedding_size=512, prompt = None): #full_size - size of prompt before splitting the tokens into batches
        super().__init__()
        self.mask_window_size = mask_window_size if mask_window_size < batch_size else batch_size//2
        self.embedding_size = embedding_size
        self.num_heads = int(t.floor(t.tensor((mask_window_size + full_size)/(embedding_size + 1))))
        if self.num_heads == 0: self.num_heads = 1
        self.batch_size = batch_size
        self.t_beliefs = BeliefsLayer(full_size, embedding_size, window_size=self.mask_window_size, embedding_size=embedding_size)
        self.mask = self.createMask() #creates a mask of the batch_size x batch_size matrix
        self.prompt = prompt

        self.Q = t.randn((self.hidden_size, self.embedding_size))
        self.K = t.randn((self.hidden_size, self.embedding_size))
        self.V = t.randn((self.hidden_size, self.embedding_size))

    def createMask(self):
        mask = t.ones(self.batch_size, self.batch_size)

        window_size = min(self.mask_window_size, self.batch_size)

        for i in range(self.batch_size):
            for j in range(window_size):
                column = (i + j) % self.batch_size
                mask[i, column] = 0 #blocks chunks of sentences from seeing each other

        return mask

    def split_batch(self, x, chunk_size=256, sliding_window=64):
        # x shape: (seq_len, embedding_size) or (batch, seq_len, embedding_size)
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        
        seq_len = x.shape[0]
        self.batch_size = chunk_size+sliding_window
        num_chunks = int(np.ceil(seq_len / chunk_size))
        chunks = []
        i=0
        while i < num_chunks:
            start = i * chunk_size
            if i == 0:
                chunks.append(x[start:start + chunk_size + sliding_window])
            else:
                chunks.append(x[max(0, start - sliding_window):start + chunk_size + sliding_window])
            i += 1

        return chunks

    def getMeanSentenceLength(self, prompt):
        sentences = [s.strip() for s in re.findall(r'[^.!?]*\.', prompt)]
        return np.mean([len(sentence) for sentence in sentences])

    def split_heads(self, x):
        self.Q, self.K, self.V, self.t_beliefs = nn.Linear(x.shape[1], x.shape[1]*3).chunk(3, dim=-1) #Splitting x into 3 chunks of equal size
        self.Q = nn.Parameter(self.Q.reshape((self.hidden_size, self.embedding_size)))
        self.K = nn.Parameter(self.K.reshape((self.hidden_size, self.embedding_size)))
        self.V = nn.Parameter(self.V.reshape((self.hidden_size, self.embedding_size)))

    def forward(self,x):
        x_chunks = self.split_batch(x) #Splitting prompt into chunks
        num_heads = self.num_heads
        dph = self.embedding_size//num_heads #dims per head

        device = "cuda"
        #Problem of redefining the Q,K,V and B weights was solved by passing the existing weights into forward pass
        all_results = []

        for chunk in x_chunks:
            if chunk.dim() == 3:
                chunk = chunk.squeeze(0)
            
            B = self.t_beliefs(chunk.unsqueeze(0)).squeeze(0) # beliefs in the chunk
            self.split_heads(chunk)
            Q = self.Q
            K = self.K
            V = self.V

            mask = self.mask
            if mask.shape[0] < V.shape[0]:
                mask = t.ones(V.shape[0], V.shape[0], device=device) # Fallback
            mask = mask[:V.shape[0], :V.shape[0]]

            seg = 0
            context = t.empty(0, dph, device=device)
            while seg < V.shape[0]:
                end_seg = min(seg + dph, V.shape[0])
                attention = t.matmul(Q[seg:end_seg], K.transpose(-2, -1)) / t.sqrt(t.tensor(dph, dtype=t.float32))

                curr_B = B[seg:end_seg] # (dph, dph)
                curr_mask = mask[seg:end_seg, :]
                
                attention = attention + t.matmul(curr_B, K.transpose(-2, -1)) + curr_mask * -1e9
                attention = t.softmax(attention, dim=-1)
                temp_context = t.matmul(attention, V)
                
                context = t.concat([context, temp_context], dim=0)
                seg += dph

            all_results.append(context)

        final_result = t.cat(all_results, dim=0)
        return final_result

class BeliefsLayer(nn.Module):
    #Beliefs layer is a simple attention mechanism that is used to calculate beliefs of the agent.
    def __init__(self, hidden_size, output_size, window_size=1, embedding_size=512, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, batch_first=True)

    def forward(self, x):
        #Plan - run the entire prompt through standard attention and then apply Top-k algorithm to achieve the most significant neurons.
        #Idea - beliefs = attention(x, x, x)
        #         beliefs = 0.99*beliefs - (1-0.99)*topk(fed_tensor)
        x, _ = self.attention(x, x, x) #a starting tensor
        def topk(x, k):
            #x - tensor to be sorted: (Batch, Seq, Dim)
            #k - number of top elements to be returned
            k = min(k, x.shape[-1])
            values, indices = t.topk(x, k, dim=-1)

            output = t.zeros_like(x) #Clears the tensor from leftover values.
            output.scatter_(-1, indices, values) #Fills the tensor with Top-k values.

            return output


        a = 0.99
        beliefs = t.mul(a,x) - (1-a)*topk(x, self.window_size) #Significance of window_size is determined by the size of the prompt
        return beliefs

