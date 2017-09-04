from models.language_model import LanguageModel 
import torch 
from torch import nn
from torch import optim
from torch.autograd import variable 
from helpers import torch_utils 

config = {}
config['vocab_size'] = 110000 
config['hidden_size'] = 150
config['embedding_size'] = 100
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False
config['use_pretrained_embeddings'] = False 
config['finetune_embeddings'] = True

language_model = LanguageModel(config).cuda()

# contexts: context_length x batch_size
# inputs: input_length x batch_size
# desired_inputs: input_length x batch_size


optimizer = optim.Adam(language_model.parameters(), lr=3e-2)
criterion = nn.NLLLoss()

for i in range(0, 1000):
    optimizer.zero_grad()
    inputs = variable.Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7]] * 100)).cuda()
    contexts = variable.Variable(torch.LongTensor([[4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10]])).cuda()
    context_masks = variable.Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).cuda()
    desired_inputs = variable.Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7]] * 100)).cuda()
    input_masks = variable.Variable(torch.FloatTensor([[1, 1, 1, 1, 1, 1, 1]] * 100)).cuda()
    answer_features = variable.Variable(torch.LongTensor([[4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10]])).cuda()
    print("On index %s" % i)
    
    optimizer.zero_grad()
    language_probs = language_model.forward(inputs, contexts, context_masks, answer_features=None)
    reshaped_inputs = desired_inputs.view(-1)
    reshaped_language_probs = language_probs.view(-1, config['vocab_size'])
    loss = criterion(reshaped_language_probs, reshaped_inputs)
    loss.backward()
    optimizer.step()



