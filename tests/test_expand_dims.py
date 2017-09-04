import numpy as np 
import torch
from torch.autograd import variable 

x = torch.Tensor([[1], [2], [3]])
print(x.size())
torch.Size([3, 1])
print(x.expand(3, 1, 1))