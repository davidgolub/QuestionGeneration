import torch
input = torch.LongTensor([[1, 2], [3, 4], [5,6]])
dim = 0 
index = torch.LongTensor([1, 2])
res = torch.gather(input, dim, index)
print(res)