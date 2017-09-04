import torch 
from torch.autograd import variable
from data_loaders.language_model_loader import LanguageModelLoader 
from models.language_model import LanguageModel
from models.language_trainer import LanguageTrainer
from models.language_wrapper import LanguageWrapper 
from helpers import constants, torch_utils

base_path = 'datasets/newsqa_train'

language_model_loader = LanguageModelLoader(base_path, tokenizer_type=constants.TOKENIZER_TAB)

config = {}
config['vocab_size'] = language_model_loader.get_vocab().size()
config['hidden_size'] = 100
config['embedding_size'] = 300 
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False
config['batch_size'] = 10
config['learning_rate'] = 1e-3
config['log_path'] = 'logs.txt'
config['save_directory'] = 'logs/newsqa_train_saved_data'
config['use_pretrained_embeddings'] = True
config['pretrained_embeddings_path'] = 'datasets/newsqa_train/word_embeddings.npy'
config['finetune_embeddings'] = False 
config['load_model'] = True
config['saved_epoch'] = 1
config['load_path'] = 'logs/newsqa_train_saved_data/model_1.pyt7'

language_model = LanguageModel(config)
if config['load_model']:
    language_model = torch_utils.load_model(config['load_path'])

language_model.cuda() 
language_wrapper = LanguageWrapper(language_model, language_model_loader.get_vocab())
language_trainer = LanguageTrainer(config, language_wrapper, language_model_loader)

for i in range(0, 100):
    loss, accuracy, predictions = language_trainer.train(epoch_num=i)
    
    if i % 3 == 1:
        predictions = language_trainer.predict(dataset_type=constants.DATASET_TEST, 
            epoch_num=10, max_length=20)
        language_trainer.save(i + config['saved_epoch'])
        language_trainer.save_predictions(i + config['saved_epoch'], predictions)




