from datasets import load_dataset
import pyarrow as pa
import pandas as pd
import torch as t
from tensorboard.compat.tensorflow_stub.errors import OutOfRangeError
from Backend import AdaptiveMultiheadMaskedAttention

from Model_constructor import SLModel

data = load_dataset("HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True)


hidden_size = 512
embedding_dim = 512
output_size = 512
input_size = 256
vocab_size = 100000
eps = 1e-4
loss = 0

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = SLModel(hidden_size, embedding_dim, vocab_size=100000, prompt="").to(device)

fn_loss = t.nn.CrossEntropyLoss()
optimizer = t.optim.AdamW(model.parameters(), lr=eps)


def create_mask(batch_size: int, mask_window_size):
    mask = t.ones(batch_size, batch_size)

    window_size = min(mask_window_size, batch_size)

    for i in range(batch_size):
        for j in range(window_size):
            column = (i + j) % batch_size
            mask[i, column] = 0  # blocks chunks of sentences from seeing each other

    return mask

for sample in data:

    phrase = sample["text"]

    #tokenizing the text sample
    tokens = model.encoder.tokenize(phrase)
    tokens = t.tensor([tokens for i in range(tokens)], dtype=t.long).to(device) #Creates tensor of size (len(tokens), len(tokens)), effectively making a matrix
    for mask_size in (20, 5, 1):
        mask = create_mask(tokens.shape[0], mask_size)
        for i in range(0, len(tokens), mask_size):
            try:
                token
            except OutOfRangeError:
                break

    loss = fn_loss(model(token_ids), tokens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss: {loss.item()}")
    t.save(model.state_dict(), "model.pt")