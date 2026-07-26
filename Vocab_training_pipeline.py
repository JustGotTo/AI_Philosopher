import json

from BytePairEncoder import BytePairEncoder

from datasets import load_dataset

if __name__ == "__main__":

    en = load_dataset("allenai/c4", "en", streaming=True)


    encoder = BytePairEncoder(prompt=en["text"])

