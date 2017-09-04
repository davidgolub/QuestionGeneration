import math
import torch
import random
import unittest
import itertools
import contextlib
from copy import deepcopy
from itertools import repeat, product
from functools import wraps

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as dp
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.nn import Parameter

lengths = [10, 10, 6, 2, 2, 1, 1]
lengths_tensor = Variable(torch.LongTensor(lengths))
max_length = lengths[0]
x = Variable(torch.randn(max_length, len(lengths), 3), requires_grad=True)
lstm = nn.LSTM(3, 4, bidirectional=True, num_layers=2, batch_first=False)

packed = rnn_utils.pack_padded_sequence(x, lengths)
packed_out, packed_hidden = lstm(packed)
unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)

def sort_sequence(tensor, lengths, batch_first=False):
    """
    Sorts sequence in descending order
    tensor: Padded tensor of variable length stuff (Torch tensor)
    lengths: Lengths of padded tensor (Torch LongTensor)
    batch_first: Boolean, whether tensor is batch_first or not  
    """
    idx = None
    if batch_first:
        idx = 0
    else:
        idx = 1 

    sorted_lengths, indices = torch.sort(lengths, dim=0, descending=True)
    new_tensor = torch.index_select(tensor, idx, indices)
    return new_tensor, sorted_lengths, indices

def unsort_sequence(tensor, indices, batch_first=False):
    """
    Unsort a tensor according to indices and idx
    """
    if batch_first:
        idx = 0
    else:
        idx = 1 
    unsorted_tensor = torch.index_select(tensor, idx, indices)
    return unsorted_tensor

def pack_forward(rnn, tensor, lengths, batch_first=False):
    """
    Forwards a padded tensor with lengths lengths thru rnn 
    rnn: Cell to forward through 
    tensor: Tensor to use 
    lengths: Lengths to use 
    batch_first: Whether tensor is batch first or not
    """

    sorted_tensor, sorted_lengths, sorted_indices = sort_sequence(tensor, lengths, batch_first)
    packed = rnn_utils.pack_padded_sequence(sorted_tensor, sorted_lengths.data.numpy())
    packed_out, packed_hidden = lstm(packed)
    unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)
    unsorted_out = unsort_sequence(unpacked, sorted_indices, batch_first=False)
    unsorted_hidden = list(map(lambda idx: unsort_sequence(packed_hidden[idx], sorted_indices, batch_first=False), [0, 1]))
    return unsorted_out, unsorted_hidden

sorted_tensor, sorted_indices, sorted_idx = sort_sequence(x, lengths_tensor, batch_first=False)
unsorted_tensor = unsort_sequence(sorted_tensor, sorted_idx)

unsorted_out, unsorted_hidden = pack_forward(lstm, x, lengths_tensor, )
print(packed_out[0].size())
print(unsorted_out[0].size())

