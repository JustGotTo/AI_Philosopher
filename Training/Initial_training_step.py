from datasets import load_dataset
import torch as t
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:10000")

from tensorboard.compat.tensorflow_stub.errors import OutOfRangeError

from Backend_construct.Model_constructor import SLModel

data = load_dataset("HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True)


hidden_size = 512
embedding_dim = 512
output_size = 512
input_size = 256
batch_size = 256
vocab_size = 100000
eps = 1e-4
loss = 0

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = SLModel(hidden_size, embedding_dim, vocab_size=25000, prompt="").to(device)

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


if __name__ == "__main__":
    print("Training started...")

for sample in data:

    phrase = sample["text"]

    #tokenizing the text sample
    tokens = model.encoder.tokenize(phrase)
    token_ids = t.tensor(tokens, device=device, dtype=t.long).to(device)
    for i in range(0, token_ids.shape[0], batch_size):
        for mask_size in (20, 5, 1):
            print(f"mask_size={mask_size}, batch_no={i}")
            mask = create_mask(batch_size, mask_size).to(device)
            batch = token_ids[i:i + batch_size]

            for j in range(0, len(tokens), mask_size):

                token_curr = batch.clone()
                mask_positions = t.zeros(
                    batch.shape,
                    dtype=t.bool,
                    device=device
                )

                try:
                    token_curr[j:j + mask_size] = 0  # Making a token list with some masked tokens
                    # Masked output = token_curr, full output = token_ids.
                    mask_positions[j:j + mask_size] = True

                except OutOfRangeError:
                    print(f"mask_size={mask_size} is too large for the current sequence length.")
                    token_curr[j:] = 0
                    mask_positions[j:] = True
                    break

                prediction = model(token_curr)
                loss = fn_loss(prediction[mask_positions], batch[mask_positions])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"mask_size={mask_size} loss: {loss.item()}")

    print(f"loss: {loss.item()}")
    t.save(model.state_dict(), "model.pt")
