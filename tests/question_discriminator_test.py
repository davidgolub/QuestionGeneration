from models.language_model import TextFieldPredictor, LanguageModel, LanguageDiscriminator
from dnn_units.lstm_attention import LSTMAttentionDot
from torch import nn 
from torch import optim
from helpers import torch_utils
import torch 
from torch.autograd import variable

load_path = 'logs/squad_saved_data/model_6.pyt7'
language_model = torch_utils.load_model(load_path)
language_model = language_model.cuda()

batch_size = 3

embeddings = language_model.embedder 
text_field_predictor = language_model.text_field_predictor 
base_lstm = language_model.base_lstm 

discriminator = LanguageDiscriminator(language_model.config, 
    embeddings, text_field_predictor, base_lstm).cuda()

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=3e-2)
discriminator_criterion = nn.BCELoss()

contexts = variable.Variable(torch.LongTensor([[1, 2, 3], [2, 3, 4], [4, 5, 6]])).cuda()
answer_features = variable.Variable(torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])).cuda()
inputs = variable.Variable(torch.LongTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])).cuda()

desired_indices = variable.Variable(torch.FloatTensor([1, 1, 1])).cuda()

for i in range(0, 100):
    discriminator_optimizer.zero_grad()
    pred = discriminator.forward(inputs, contexts, answer_features)
    bce_loss = discriminator_criterion(pred, desired_indices)
    bce_loss.backward()

    print(bce_loss)
    discriminator_optimizer.step()





