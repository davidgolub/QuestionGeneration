from models.language_model import LanguageModel
from helpers import torch_utils 

config = {}
config['vocab_size'] = 12
config['embedding_size'] = 20 
config['hidden_size'] = 50
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = True 

model = LanguageModel(config)

torch_utils.save_model(model, path='test.model')
model = torch_utils.load_model(path='test.model')