from data_loaders.language_model_loader import LanguageModelLoader
from models.language_model import LanguageModel 
from helpers import constants

base_path = 'datasets/newsqa'
language_model_loader = LanguageModelLoader(base_path, tokenizer_type=constants.TOKENIZER_NLTK)
language_model_loader.reset_indices()
batch = language_model_loader.get_batch(dataset_type=constants.DATASET_TRAIN, batch_size=10)

config = {}
config['vocab_size'] = language_model_loader.get_vocab().size()
config['hidden_size'] = 100
config['embedding_size'] = 300 
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False
config['batch_size'] = 24
config['learning_rate'] = 1e-3
config['log_path'] = 'logs.txt'
config['save_directory'] = 'logs/squad_saved_data'
config['use_pretrained_embeddings'] = True
config['pretrained_embeddings_path'] = 'datasets/squad/word_embeddings.npy'
config['finetune_embeddings'] = False 
config['load_model'] = True 
config['load_path'] = 'logs/squad_saved_data/model_7_old.pyt7'

language_model = LanguageModel(config)

