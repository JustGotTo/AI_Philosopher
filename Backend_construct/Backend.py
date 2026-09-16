from Backend_construct.BytePairEncoder import BytePairEncoder

import torch.nn as nn
import torch as t
import numpy as np


class Embedding(nn.Module):
    def __init__(self, prompt, vocab_size=100000, embedding_dim=512):
        super().__init__()
        self.prompt = BytePairEncoder(prompt=prompt).forward(prompt)  # MAY CRASH: if BytePairEncoder.forward expects token IDs/tensors instead of raw string; side-effectful init may also be heavy
        self.embedding_dim = embedding_dim #same as input_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)  # NOTE: input to forward must be Long tensor of token IDs; weights remain float

    def forward(self, input):
        return self.embedding(input) # Simple embedding mechanism; REQUIRES input dtype=torch.long and values in [0, vocab_size) — otherwise Embedding will throw an index/type error


class AddNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(t.ones(hidden_size))

    def forward(self, residual, x):  # CRASH RISK: residual and x must have same shape; dtype/shape mismatch will raise runtime error
        x = x + residual
        rms = t.sqrt(t.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class LinearPostAttention(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.weight = nn.Parameter(t.ones(output_size)*0.9)
        self.bias = nn.Parameter(t.zeros(output_size))

    def forward(self, x):
        return self.weight * x + self.bias  # CRASH RISK: last dim of x must equal output_size; otherwise broadcast/shape mismatch occurs

class SentenceFeedForward(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.norm = AddNorm(hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)  # CRASH RISK: last dim of x must equal hidden_size for nn.Linear(hidden_size, hidden_size)
        x = self.norm(x, x)  # ASSUMPTION: residual shape matches x; misuse can raise shape mismatch error
        x = self.act(x)
        x = self.linear2(x)  # CRASH RISK: expects last dim == hidden_size; otherwise Linear will throw
        return x

class PhraseFeedForward(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.norm = AddNorm(output_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear1(x)  # CRASH RISK: expects last dim == hidden_size; passing embedding_dim here will raise
        x = self.norm(x, x)  # CRASH RISK: AddNorm was constructed with output_size; shapes must match x
        x = self.act(x)
        #x = self.norm(x, x)  # POTENTIAL LOGIC BUG: applying AddNorm twice consecutively; not a crash by itself but suspicious
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
        x = self.linear1(x)  # CRASH RISK: last dim of x must equal hidden_size
        x = self.act(x)
        x = self.linear2(x)  # CRASH RISK: expects last dim == hidden_size
        x = self.norm(x, x)  # CRASH RISK: AddNorm constructed with output_size; must match x's last dim after linear2
        x = self.dropout(x)
        return x

class AdaptiveMultiheadMaskedAttention(nn.Module):
    def __init__(self, batch_size:int, full_size:int, mask_window_size, embedding_size=512, prompt = None, device="cuda"): #full_size - size of prompt before splitting the tokens into batches
        super().__init__()

        self.mask_window_size = mask_window_size if mask_window_size < batch_size else batch_size//2

        self.prompt = prompt
        self.full_size = full_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.device = device

        self.num_heads = int(t.floor(t.tensor((mask_window_size + full_size)/(embedding_size + 1))))  # MAY CRASH/HANG: if num_heads > embedding_size then dph=embedding_size//num_heads becomes 0, leading to an infinite loop and/or invalid math later
        if self.num_heads == 0: self.num_heads = 1
        self.batch_size = batch_size
        self.t_beliefs = BeliefsLayer(full_size, embedding_size, window_size=self.mask_window_size, embedding_size=embedding_size)  # NOTE: passing 'embedding_size' positionally as output_size, and also as a named arg; works but is confusing
        self.mask = self.create_mask(self.device)  # BUG: missing required 'device' argument; this call will raise TypeError at init time. Also, mask may be created on CPU and later used with GPU tensors -> device mismatch. # creates a mask of the batch_size x batch_size matrix
        self.splitter = nn.Linear(self.embedding_size, self.embedding_size*3)

    def create_mask(self, device):  # API footgun: 'device' is required by callers; 'batch_size' param is unused inside (self.batch_size is used instead) — missing/incorrect args here will crash
        mask = t.ones(self.batch_size, self.batch_size, device=device, dtype=t.float32)

        window_size = min(self.mask_window_size, self.batch_size)

        for i in range(self.batch_size):
            for j in range(window_size):
                column = (i + j) % self.batch_size
                mask[i, column] = 0 #blocks chunks of sentences from seeing each other

        return mask

    def split_batch(self, x, prompt=None, chunk_size=None, sliding_window=64):  # CRASH RISK: chunk_size must be an int; if None, arithmetic below (chunk_size + sliding_window, ceil division) will raise TypeError
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


    def split_heads(self, x):
        qkv = self.splitter(x)  # CRASH RISK: x's last dim must equal embedding_size; otherwise Linear will raise. Splitting x into 3 chunks of equal size
        Q, K, V = t.chunk(qkv, chunks=3, dim=-1)  # ASSUMPTION: splitter.out_features divisible by 3; else chunk will throw
        return Q, K, V

    def forward(self,x, mask_window_size=1):
        self.mask_window_size = mask_window_size if mask_window_size < self.batch_size else self.batch_size%33  # CRASH RISK: if result is 0, later dph calc may be 0; also changing batch_size elsewhere can desync mask dimensions
        x_chunks = self.split_batch(x, chunk_size=256, sliding_window=64) #Splitting prompt into chunks
        num_heads = self.num_heads
        dph = self.embedding_size//num_heads # dims per head; CRASH/HANG RISK: if num_heads > embedding_size then dph==0 → division by zero below and infinite while-loop (seg += dph)

        device = x.device
        #Problem of redefining the Q,K,V and B weights was solved by passing the existing weights into forward pass
        all_results = []

        for chunk in x_chunks:
            if chunk.dim() == 3:
                chunk = chunk.squeeze(0)
            
            B = self.t_beliefs(chunk.unsqueeze(0)).squeeze(0) # beliefs in the chunk
            Q, K, V = self.split_heads(chunk)

            mask = self.mask  # CRASH RISK: mask may have been created on a different device (e.g., CPU) in __init__; adding it to GPU tensors below will raise a device mismatch unless recreated/moved
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

