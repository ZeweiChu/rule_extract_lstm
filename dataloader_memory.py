import torch
import torch.nn as nn
from torch.nn import Parameter 
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys
import numpy as np
import json
import nltk
from collections import Counter
import pickle

class Reader():
    def __init__(self, args):
        super(Reader, self).__init__()
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.batch_size = args.batch_size
        self.vocab_size = args.vocab_size
        self.last_line = None

    def build_dict(self):
        word_dict = Counter()
        for line in self.train_data:
            for token in line[0]:
                word_dict[token] += 1

        word_dict = word_dict.most_common(self.vocab_size)
        word_dict = {w[0]: index + 1 for (index, w) in enumerate(word_dict)}
        word_dict["UNK"] = 0
        return word_dict

    def load_data(self):
        self.fp_train = open(self.train_file, "r")
        self.train_data = []
        for line in self.fp_train:
            # line = line.strip().split("\t")
            # text = nltk.word_tokenize(line[0])
            # polarity = int(line[1])
            obj = json.loads(line)
            text = obj["text"]
            text = nltk.word_tokenize(text)
            stars = int(obj["stars"])
            if stars >= 3:
                polarity = 1
            else: 
                polarity = 0
            self.train_data.append([text, polarity]) 

        self.fp_dev = open(self.dev_file, "r")
        self.dev_data = []
        for line in self.fp_dev:
            obj = json.loads(line)
            text = obj["text"]
            text = nltk.word_tokenize(text)
            stars = int(obj["stars"])
            if stars >= 3:
                polarity = 1
            else: 
                polarity = 0
            self.dev_data.append([text, polarity]) 

    def load_test_data(self):
        self.fp_test = open(self.test_file, "r")
        self.test_data = []
        for line in self.fp_test:

            obj = json.loads(line)
            text = obj["text"]
            text = nltk.word_tokenize(text)
            stars = int(obj["stars"])
            if stars >= 3:
                polarity = 1
            else: 
                polarity = 0
            self.test_data.append([text, polarity]) 

    def encode(self, vocab, sort_by_len=True):
        for idx, line in enumerate(self.train_data):
            line[0] = [vocab[w] if w in vocab else 0 for w in line[0]]
        for idx, line in enumerate(self.dev_data):
            line[0] = [vocab[w] if w in vocab else 0 for w in line[0]]

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x][0]))

        if sort_by_len:
            sorted_index = len_argsort(self.train_data)
            self.train_data = [self.train_data[i] for i in sorted_index]

    def encode_test(self, vocab, sort_by_len=True):
        for idx, line in enumerate(self.test_data):
            line[0] = [vocab[w] if w in vocab else 0 for w in line[0]]
        

    def get_minibatches(self, n, shuffle=True):
        idx_list = np.arange(0, n, self.batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx + self.batch_size, n)))
        return minibatches

    def prepare_data(self, data):
        lengths = [len(line[0]) for line in data]
        n_samples = len(data)

        max_len = np.max(lengths)
        x = np.zeros((n_samples, max_len)).astype("int32")
        x_mask = np.zeros((n_samples, max_len)).astype('float32')
        y = np.zeros((n_samples)).astype("int32")
        for idx, d in enumerate(data):
            x[idx, :lengths[idx]] = d[0]
            x_mask[idx, :lengths[idx]] = 1.
            y[idx] = d[1]
        return x, x_mask, y

    def gen_examples(self, data):
        minibatches = self.get_minibatches(len(data), self.batch_size)
        all_ex = []
        for minibatch in minibatches:
            mb_sentences = [data[t] for t in minibatch]
            x, x_mask, y = self.prepare_data(mb_sentences)
            all_ex.append((x, x_mask, y))
        return all_ex
