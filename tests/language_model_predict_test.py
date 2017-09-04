from models.language_model import LanguageModel 
import torch 
from torch import nn
from torch import optim
from torch.autograd import variable 
from helpers import torch_utils 

config = {}
config['vocab_size'] = 25 
config['hidden_size'] = 50
config['embedding_size'] = 10 
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False

language_model = LanguageModel(config)
language_model.cuda()
# contexts: context_length x batch_size
# inputs: input_length x batch_size
# desired_inputs: input_length x batch_size

input_token = variable.Variable(torch.LongTensor([[1]]))
context_tokens = variable.Variable(torch.LongTensor([[2], [3], [4], [5], [6], [7], [8]]))
language_model.predict(input_token, context_tokens, torch.LongTensor([[1]]))