from data_loaders.language_model_loader_truncate import LanguageModelLoaderTruncate
from models.language_model import LanguageModel
from models.language_trainer import LanguageTrainer
from models.language_wrapper import LanguageWrapper 
from helpers import constants, torch_utils
import torch 
from torch.autograd import variable

base_path = 'datasets/squad_expanded_vocab'


language_model_loader = LanguageModelLoaderTruncate(base_path, tokenizer_type=constants.TOKENIZER_TAB)

config = {}
config['vocab_size'] = language_model_loader.get_vocab().size()
config['hidden_size'] = 100
config['embedding_size'] = 300 
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False
config['batch_size'] = 20
config['learning_rate'] = 1e-3
config['log_path'] = 'logs.txt'
config['save_directory'] = 'logs/squad_saved_data_truncated_expanded_vocab'
config['use_pretrained_embeddings'] = True
config['pretrained_embeddings_path'] = 'datasets/squad_expanded_vocab/word_embeddings.npy'
config['finetune_embeddings'] = False 
config['load_model'] = False
config['beam_size'] = 5
config['load_path'] = 'logs/squad_saved_data_truncated/model_0.pyt7' # CHANGE THIS TO ONE OF THE SAVED MODEL PATHS

language_model = LanguageModel(config)
if config['load_model']:
    language_model = torch_utils.load_model(config['load_path'])

language_model.cuda() 
language_wrapper = LanguageWrapper(language_model, language_model_loader.get_vocab())
language_trainer = LanguageTrainer(config, language_wrapper, language_model_loader)

for i in range(0, 10):
    loss, accuracy, predictions = language_trainer.train(epoch_num=i)
    
    if i % 2 == 0:
        predictions = language_trainer.predict(dataset_type=constants.DATASET_TEST, 
            epoch_num=10, max_length=20)
        language_trainer.save(i)
        language_trainer.save_predictions(i, predictions)




