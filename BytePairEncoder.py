import torch.nn as nn
import re


class BytePairEncoder(nn.Module):
    def __init__(self, prompt="", vocab_size=1000, input_size=128, hidden_size=384, output_size=384):
        super().__init__()
        self.prompt = prompt
        self.vocab_size = vocab_size
        self.vocab = {}
        self.frequency = {}
        self.merges = []

    def get_pairs(self, words):
        frequency = {}

        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                frequency[pair] = frequency.get(pair, 0) + 1

        return frequency

    def merge_pair_in_word(self, word, merge):
        new_word = []
        i = 0
        left, right = merge
        merged_token = left + right

        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        return new_word

    def apply_merge(self, words, merge):
        new_words = []
        changed = False

        for word in words:
            new_word = self.merge_pair_in_word(word, merge)

            if new_word != word:
                changed = True

            new_words.append(new_word)

        return new_words, changed

    def add_token_to_vocab(self, token):
        if token not in self.vocab and len(self.vocab) < self.vocab_size:
            self.vocab[token] = len(self.vocab)

    def train_vocab(self, prompt):
        raw_words = re.findall(r"\w+", prompt)

        words = [list(word) for word in raw_words]

        # Add individual characters first.
        for word in words:
            for token in word:
                self.add_token_to_vocab(token)

        while len(self.vocab) < self.vocab_size:
            self.frequency = self.get_pairs(words)

            if not self.frequency:
                break

            best_merge = max(self.frequency, key=self.frequency.get)
            merged_token = best_merge[0] + best_merge[1]

            new_words, changed = self.apply_merge(words, best_merge)

            if not changed:
                break

            if merged_token in self.vocab:
                del self.frequency[best_merge]

                if not self.frequency:
                    break

                continue

            self.merges.append(best_merge)
            self.add_token_to_vocab(merged_token)

            words = new_words

        return words

    def encode_word(self, word):
        tokens = list(word)

        for merge in self.merges:
            tokens = self.merge_pair_in_word(tokens, merge)

        return tokens

    def tokenize(self, prompt):
        self.vocab = {}
        self.frequency = {}
        self.merges = []

        self.train_vocab(prompt)

        raw_words = re.findall(r"\w+", prompt)
        result = []

        for word in raw_words:
            tokens = self.encode_word(word)

            for token in tokens:
                if token in self.vocab:
                    result.append(self.vocab[token])

        return result

    def forward(self, prompt):
        return self.tokenize(prompt)


class SLModel(nn.Module):
    def __init__(self, prompt, enable_encoder=True, layers=3, vocab_size=50000):
        super().__init__()
        self.prompt = prompt
        self.vocab_size = vocab_size
        self.layers = layers
        self.enable_encoder = enable_encoder
        self.encoder = BytePairEncoder(vocab_size=self.vocab_size, prompt="")