#
# Attention mechanisms
#
#
# Developed environment:
# Python                       3.11.x
# torch                        2.x (MPS support: macOS Apple Silicon / CPU fallback)
# numpy                        1.26.x
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import torch
import torch.nn as nn

"""
Bahdanau Attention, a.k.a. Additive attention, Multi-Layer perceptron
"""


class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, query, values):
        # query : (batch, hidden_units)
        # values: (batch, seq_len, hidden_units)
        query_with_time_axis = query.unsqueeze(1)
        # score: (batch, seq_len, 1)
        score = self.V(torch.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        # attention_weights: (batch, seq_len, 1)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(dim=1)

        return context_vector, attention_weights


"""
Luong attention, a.k.a. Bilinear Attention, General Attention.
"""


class LuongAttention(nn.Module):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(units, units)

    def forward(self, query, values):
        query_with_time_axis = query.unsqueeze(1)

        # equivalent to tf.matmul(query_with_time_axis, self.W(values), transpose_b=True)
        # query_with_time_axis: (batch, 1,       hidden_units)
        # self.W(values)      : (batch, seq_len, hidden_units)
        # .transpose(1, 2)    : (batch, hidden_units, seq_len)
        # matmul result       : (batch, 1, seq_len)
        # .transpose(1, 2)    : (batch, seq_len, 1)
        score = torch.matmul(
            query_with_time_axis, self.W(values).transpose(1, 2)
        ).transpose(1, 2)

        # attention_weights: (batch, seq_len, 1)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(dim=1)

        return context_vector, attention_weights
