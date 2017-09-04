import torch 
from torch.autograd import variable
from data_loaders.language_model_loader import LanguageModelLoader 
from models.language_model import LanguageModel
from models.language_trainer import LanguageTrainer
from models.language_wrapper import LanguageWrapper 
from helpers import constants, torch_utils


base_path = 'datasets/question_generator'


language_model_loader = LanguageModelLoader(base_path)

config = {}
config['vocab_size'] = language_model_loader.get_vocab().size()
config['hidden_size'] = 100
config['embedding_size'] = 300 
config['num_layers'] = 1
config['dropout'] = 0.0
config['batch_first'] = False
config['batch_size'] = 3 
config['learning_rate'] = 1e-3
config['log_path'] = 'logs.txt'
config['save_directory'] = 'logs/saved_data'
config['use_pretrained_embeddings'] = True
config['pretrained_embeddings_path'] = 'datasets/question_generator/word_embeddings.npy'
config['finetune_embeddings'] = True 
config['gpu_mode'] = True

language_model = LanguageModel(config).cuda()
language_wrapper = LanguageWrapper(language_model, language_model_loader.get_vocab())
language_trainer = LanguageTrainer(config, language_wrapper, language_model_loader)

for i in range(0, 100):
    loss, accuracy, predictions = language_trainer.train(epoch_num=i)
    
    if i % 10 == 0:
        predictions = language_trainer.predict(dataset_type=constants.DATASET_TEST,epoch_num=10, max_length=20)
        language_trainer.save_predictions(i, predictions)
        language_trainer.save(i)



