import torch 
import numpy as np 
from torch.autograd import variable 
from models.language_model import TextFieldPredictor, SoftmaxPredictor

config = {}
config['vocab_size'] = 12
config['embedding_size'] = 20 
config['hidden_size'] = 50
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = True 

# First test text field predictor
inp = variable.Variable(torch.LongTensor([[1, 2, 3], [4, 5, 6]]))
hidden = variable.Variable(torch.randn(2, config['hidden_size']))
predictor = TextFieldPredictor(config)
lstm_embeddings = predictor.forward_prepro(inp)
h_tilde, attentions, inp = predictor.forward_similarity(hidden)

inp1 = variable.Variable(torch.LongTensor(2, config['vocab_size'] - 3).zero_())
inp2 = variable.Variable(torch.zeros(2, config['vocab_size'] - 3))
stacked_inps = torch.cat((inp, inp1), 1)
stacked_attentions = torch.cat((attentions, inp2), 1)

# Second test softma predictor
softmax_predictor = SoftmaxPredictor(config)
softmax_logits = softmax_predictor.forward(hidden)

res = variable.Variable(torch.zeros(2, config['vocab_size']))
res.scatter_(1, stacked_inps, stacked_attentions)

tmp = softmax_logits + res

print(tmp)




