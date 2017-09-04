from data_loaders.language_model_loader import LanguageModelLoader 
from models.language_model import LanguageModel
from models.language_discriminator_trainer import LanguageDiscriminatorTrainer
from models.language_wrapper import LanguageWrapper 
from helpers import constants 
import torch 
from helpers import torch_utils, utils
from torch.autograd import variable

dataset_path = 'datasets/squad'
load_path = 'logs/squad_saved_data/model_6.pyt7'

language_model_loader = LanguageModelLoader(dataset_path, tokenizer_type=constants.TOKENIZER_TAB)
language_model = torch_utils.load_model(load_path).cuda()
language_model.config['save_directory'] = 'logs/newsqa_saved_data'

language_wrapper = LanguageWrapper(language_model, language_model_loader.get_vocab())
language_trainer = LanguageDiscriminatorTrainer(language_model.config, language_wrapper, language_model_loader)

for i in range(0, 100):
    language_trainer.predict(dataset_type=constants.DATASET_TRAIN, epoch_num=1, max_length=20)




