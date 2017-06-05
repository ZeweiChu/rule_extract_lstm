import torch
import torch.nn as nn
from torch.nn import Parameter 
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        num_directions = 2 if bidirectional else 1

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                gate_size = 4 * hidden_size

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))

                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                setattr(self, weights[0], w_ih)
                setattr(self, weights[1], w_hh)
                if bias:
                    setattr(self, weights[2], b_ih)
                    setattr(self, weights[3], b_hh)
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        B, T, input_encoding_size = input.size()

        forgetgates = []
        hiddens = []
        cellgates = []
        outgates = []

        hx, cx = hx
        for i in range(T):
            gates = F.linear(input[:, i, :], self.weight_ih_l0, self.bias_ih_l0) \
                        + F.linear(hx, self.weight_hh_l0, self.bias_hh_l0)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            forgetgates.append(forgetgate)
            cellgate = F.tanh(cellgate)
            cellgates.append(cellgate.unsqueeze(1))
            outgate = F.sigmoid(outgate)
            outgates.append(outgate)

            cx = (forgetgate * cx) + (ingate * cellgate)
            hx = outgate * F.tanh(cx)
            hiddens.append(hx.unsqueeze(1))

        hiddens = torch.cat(hiddens, 1)
        cellgates = torch.cat(cellgates, 1)
        return forgetgates, hiddens, cellgates, outgate

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.lstm = LSTM(self.embedding_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 2)

        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hx = Variable(weight.new(bsz, self.hidden_size).zero_())
        cx = Variable(weight.new(bsz, self.hidden_size).zero_())
        return (hx, cx)


    def forward(self, x, x_mask):
        B, T = x.size()
        x = self.embed(x)
        forgetgates, hiddens, cellgates, outgate = self.lstm(x, hx=self.init_hidden(B))
        lengths = x_mask.sum(1).long()
        lengths = lengths.unsqueeze(2).expand(B, 1, self.hidden_size) - 1
        
        x = torch.gather(hiddens, 1, lengths).squeeze(1) # batch * length * dimension
        x = self.linear(x)
        return x #torch.exp(F.log_softmax(x))
        # code.interact(local=locals())
        


