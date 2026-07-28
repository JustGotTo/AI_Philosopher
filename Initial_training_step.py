from huggingface_hub import snapshot_download
import pyarrow as pa
import pandas as pd
import torch as t

from Model_constructor import SLModel

data = snapshot_download( #Loading the data for training
                "HuggingFaceFW/fineweb",
                repo_type="dataset",
                local_dir="./fineweb/",
                allow_patterns="sample/10BT/*")

data = data.to_pandas_dataframe().dropna()[:100000]

phrases = data["text"]
sample = phrases.head()

hidden_size = 512
embedding_dim = 512
output_size = 512
input_size = 256
vocab_size = 25000
eps = 1e-6

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = SLModel(hidden_size, embedding_dim, vocab_size=25000, prompt="")

for line in sample:
    line = line.to(model.device)
    line = model.generate(line)
