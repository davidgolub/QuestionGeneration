from data_loaders.language_model_loader import LanguageModelLoader
from models.pointer_network import PointerNetwork
from helpers import constants
import torch
from torch.autograd import variable 
from torch import optim 
from torch import nn

base_path = 'datasets/newsqa_train'  
language_model_loader = LanguageModelLoader(base_path, tokenizer_type=constants.TOKENIZER_NLTK)
language_model_loader.mix_indices()

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

pointer_network = PointerNetwork(config).cuda()


criterion1 = nn.CrossEntropyLoss().cuda()
criterion2 = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(pointer_network.parameters(), 1e-2)


batch = language_model_loader.get_batch(dataset_type=constants.DATASET_TRAIN, batch_size=config['batch_size'])

large_negative_number = -1.e-10
while batch is not None:
    optimizer.zero_grad()
    input_lengths = variable.Variable(torch.from_numpy(batch['context_lengths'])).cuda()
    input_vals = variable.Variable(torch.from_numpy(batch['context_tokens'])).cuda()
    answer_starts = variable.Variable(torch.from_numpy(batch['answer_starts'])).cuda()
    answer_ends = variable.Variable(torch.from_numpy(batch['answer_ends'])).cuda()
    masks = variable.Variable(torch.from_numpy(batch['context_masks'].T).float()).cuda()

    p_start, p_end = pointer_network.forward(input_vals, input_lengths, masks)

    # Batch first
    loss = criterion1(p_start, answer_starts) + \
        criterion2(p_end, answer_ends)

    print(loss)
    loss.backward()
    optimizer.step()
    batch = language_model_loader.get_batch(dataset_type=constants.DATASET_TRAIN, batch_size=config['batch_size'])


