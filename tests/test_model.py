import numpy as np
import torch 
import torch.nn as nn 
import torch.optim
from torch.autograd import variable

from models.card_model import CardModel 

config = {}
config['vocab_size'] = 52
config['embedding_size'] = 23 

model = CardModel(config)

emb1 = nn.Embedding(config['vocab_size'], config['embedding_size'])

desired = variable.Variable(torch.randn(3, 23))
tmp = variable.Variable(torch.LongTensor([1,2,3]))
tmp1 = emb1(tmp)
tmp2 = emb1(tmp)

criterion = nn.MSELoss()
loss = criterion(tmp1 + tmp2, desired)
loss.backward()