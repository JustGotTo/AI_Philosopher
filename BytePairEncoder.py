import torch as t
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn as nn
import re, collections

# TODO: FINAL GOAL: Make small language model, with a handcrafted encoder and optimizer to allow for at-home training
# TODO: FIRST GOAL: Make a byte-pair encoder.
# TODO: SECOND GOAL: Introduce hierarchical attention, hard-code the algorithm, find the dimensions that will work best
# TODO: THIRD GOAL: Create an updated standard attention mechanism for the "word clouds"


class BytePairEncoder(nn.Module):
    def __init__(self, prompt, vocab_size, input_size=128, hidden_size=384, output_size=384):
        super().__init__()
        self.prompt = prompt
        self.vocab_size = vocab_size
        self.vocab = {}   #maps token string → integer ID
        self.frequency = {}

    def get_pairs(self, wordList):
        chars = ["<"] + wordList + [">"]
        for char in range(len(chars)-1):
            pair = chars[char] + chars[char+1]
            if pair in self.frequency:
               self.frequency[pair] += 1
            else:
                self.frequency[pair] = 1
                #self.vocab[pair] = len(self.vocab)

    def next_step_merge(self, wordList, merge):
        chars = ["<"] + wordList + [">"]
        if list(self.vocab[merge]) in chars:
            index = chars.index(merge[0])
            chars = list(filter(lambda c: c not in merge, chars))
            chars.insert(index, merge[0] + merge[1])
            #joining all the merges in the word(from the ones we took into vocab)
        return chars

    def merge_most_frequent(self):
        largest = 0
        merg = ""
        for merge in self.frequency.keys():
            if self.frequency[merge] > largest:
                largest = self.frequency[merge]
                merg = merge
        self.vocab[merg] = len(self.vocab)



    def tokenize(self, prompt):
        words = re.findall(r'\w+', prompt)
        self.frequency = {}
        for word in words:
            word = list(word)
            self.get_pairs(word) #first we find all pairs in all words and get frequency for each

        while len(self.vocab) < self.vocab_size:
            if not self.frequency:
                break

            self.merge_most_frequent()
            best_merge = list(self.vocab.keys())[-1]

            new_words = []
            for word in words:
                new_words.append(self.next_step_merge(word, best_merge))
            words = new_words

            self.frequency = {}
            for word in words:
                self.get_pairs(word)



        return [self.vocab[pair] for pair in pairs.keys()]




class SLModel(nn.Module):
    def __init__(self, prompt, enable_encoder=True, layers=3, vocab_size=50000):
        super().__init__()
        self.prompt = prompt
        self.vocab_size = vocab_size
        self.layers = layers
        self.enable_encoder = enable_encoder
        self.encoder = BytePairEncoder(vocab_size=self.vocab_size, prompt="")

