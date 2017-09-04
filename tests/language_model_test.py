import torch 
from torch import nn
from torch import optim
from torch.autograd import variable 
from helpers import torch_utils 
from models.language_model import LanguageModel 

config = {}
config['vocab_size'] = 25 
config['hidden_size'] = 50
config['embedding_size'] = 10 
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False
config['use_pretrained_embeddings'] = False
config['gpu_mode'] = True 

language_model = LanguageModel(config)

# contexts: context_length x batch_size
# inputs: input_length x batch_size
# desired_inputs: input_length x batch_size

inputs = variable.Variable(torch.LongTensor([[1, 2, 3], [4,5,6]])).cuda()
contexts = variable.Variable(torch.LongTensor([[4, 5, 6], [7, 8, 9], [4, 5, 6], [7, 8, 9]])).cuda()
desired_inputs = variable.Variable(torch.LongTensor([[2, 3, 4], [5, 6, 7]])).cuda()

optimizer = optim.Adam(language_model.parameters(), lr=3e-2)
criterion = nn.NLLLoss()
language_model.cuda()

for i in range(0, 100):
    optimizer.zero_grad()
    language_probs = language_model.forward(inputs, contexts, context_masks=None, answer_features=contexts.float())
    reshaped_inputs = desired_inputs.view(-1)
    reshaped_language_probs = language_probs.view(-1, config['vocab_size'])

    max_likelihoods, best_indices = torch.max(language_probs, 2)
    diff = torch.eq(torch.squeeze(best_indices).data,desired_inputs.data)
    accuracy = (diff.sum()) / torch_utils.num_elements(diff)

    loss = criterion(reshaped_language_probs, reshaped_inputs)
    loss.backward()
    optimizer.step()

    print(loss)
    print(accuracy)



