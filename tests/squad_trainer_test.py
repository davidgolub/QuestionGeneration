import torch 
from torch.autograd import variable
from data_loaders.language_model_loader import LanguageModelLoader 
from models.language_model import LanguageModel
from models.language_trainer import LanguageTrainer
from models.language_wrapper import LanguageWrapper 
from helpers import constants, torch_utils, io_utils

torch.cuda.set_device(1)

base_path = 'datasets/squad/'
language_model_loader = LanguageModelLoader(base_path, tokenizer_type=constants.TOKENIZER_TAB)

config = {}
config['vocab_size'] = language_model_loader.get_vocab().size()
config['hidden_size'] = 100
config['embedding_size'] = 300 
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False
config['batch_size'] = 24
config['learning_rate'] = 1e-3
config['beam_size'] = 5
config['log_path'] = 'logs.txt'
config['save_directory'] = 'logs/squad_saved_data'
config['use_pretrained_embeddings'] = True
config['pretrained_embeddings_path'] = 'datasets/squad/word_embeddings.npy'
config['finetune_embeddings'] = False 
config['load_model'] = False
config['gpu_mode'] = True
config['load_path'] = 'logs/squad_saved_data/model_6.pyt7' # CHANGE THIS TO WHATEVER PATH YOU WANT

io_utils.check_dir('logs/squad_saved_data')

language_model = LanguageModel(config)
if config['load_model']:
    language_model = torch_utils.load_model(config['load_path'])

language_model.cuda() 
language_wrapper = LanguageWrapper(language_model, language_model_loader.get_vocab())
language_trainer = LanguageTrainer(config, language_wrapper, language_model_loader)

for i in range(0, 15):
    loss, accuracy, predictions = language_trainer.train(epoch_num=i)
    
    if i % 3 == 2:
        predictions = language_trainer.predict(dataset_type=constants.DATASET_TEST, 
            epoch_num=10, max_length=20, beam_size=config['beam_size'])
        language_trainer.save(i)
        language_trainer.save_predictions(i, predictions)




