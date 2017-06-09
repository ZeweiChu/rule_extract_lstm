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

    def init_train(self):
        self.fp_train = open(self.train_file, "r")

    def reset_train(self):
        self.fp_train.seek(0)

    def close_train(self):
        self.fp_train.close()

    def init_dev(self):
        self.fp_dev = open(self.dev_file, "r")

    def reset_dev(self):
        self.fp_dev.seek(0)

    def close_dev(self):
        self.fp_dev.close()

    def init_test(self):
        self.fp_test = open(self.test_file, "r")

    def reset_test(self):
        self.fp_test.seek(0)

    def close_test(self):
        self.fp_test.close()


    def build_dict(self):
        word_dict = Counter()
        for line in self.fp_train:
            text = json.loads(line)["text"]
            for token in nltk.word_tokenize(text):
                word_dict[token] += 1

        word_dict = word_dict.most_common(self.vocab_size)
        word_dict = {w[0]: index + 1 for (index, w) in enumerate(word_dict)}
        word_dict["UNK"] = 0
        return word_dict
        # code.interact(local=locals())

    def encode_batch(self, data, vocab):
        for d in data:
            d[0] = [vocab[w] if w in vocab else 0 for w in d[0]]
            
        lengths = [len(d[0]) for d in data]
        batch_size = len(data)
        max_len = np.max(lengths)
        x = np.zeros((batch_size, max_len)).astype("int32")
        x_mask = np.zeros((batch_size, max_len)).astype("float32")
        y = np.zeros((batch_size)).astype("int32")
        for idx, d in enumerate(data):
            x[idx, :lengths[idx]] = d[0]
            x_mask[idx, :lengths[idx]] = 1.
            y[idx] = d[1]
        return (x, x_mask, y)

    def get_batch(self, data_split="train", vocab={}):
        data = []
        if self.last_line is None:
            self.last_line = getattr(self, "fp_" + data_split).readline()
        last_batch = False
        for i in range(self.batch_size):
            line = getattr(self, "fp_" + data_split).readline()
            # print(self.last_line)
            obj = json.loads(self.last_line)
            text = obj["text"]
            text = nltk.word_tokenize(text)
            stars = int(obj["stars"])
            if stars >= 3:
                polarity = 1
            else: 
                polarity = 0
            data.append([text, polarity]) 
            # code.interact(local=locals())
            if not line:
                self.fp_train.seek(0)
                last_batch = True
                self.last_line = None
                break
            else:
                self.last_line = line
            
        return self.encode_batch(data, vocab), last_batch
