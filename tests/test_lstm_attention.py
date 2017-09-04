from dnn_units.lstm_attention import LSTMAttentionDot, LSTMAttention
import torch 
from torch import nn
from torch.autograd import variable 
from torch import optim

batch_size = 25
input_size = 125
input_length = 25
hidden_size = 250
ctx_length = 230

net = LSTMAttentionDot(input_size=input_size, 
    hidden_size=hidden_size, 
    batch_first=False).cuda()

inputs = variable.Variable(torch.randn(input_length, batch_size,  input_size)).cuda()
hidden = variable.Variable(torch.randn(batch_size, hidden_size)).cuda()
cell = variable.Variable(torch.randn(batch_size, hidden_size)).cuda()
context = variable.Variable(torch.randn(ctx_length, batch_size, hidden_size)).cuda()
desired = variable.Variable(torch.randn(batch_size, hidden_size)).cuda()

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=3e-2)

for i in range(0, 1000):
    print(i)
    optimizer.zero_grad()
    out, h = net.forward(inputs, [hidden, cell], context)
    loss = criterion(h[0], desired)
    loss.backward()
    optimizer.step()
