from models.language_model import LanguageModel, LanguageDiscriminator
from helpers import vocab 
import numpy as np 
import torch
from torch.autograd import variable

class LanguageWrapper(object):
    def __init__(self, model, vocab):
        self.model = model 
        self.vocab = vocab
        self.create_discriminator()

    def create_discriminator(self):
        # Hack to create discriminator: note 
        # Violates interface design pattern
        embeddings = self.model.embedder 
        text_field_predictor = self.model.text_field_predictor 
        base_lstm = self.model.base_lstm 

        self.discriminator = LanguageDiscriminator(self.model.config, 
        embeddings, text_field_predictor, base_lstm).cuda()

    def get_discriminator(self):
        return self.discriminator 

    def get_model(self):
        return self.model 

    def predict(self, context_tokens, answer_features, max_length, pad=False):
        input_token = variable.Variable(torch.LongTensor([[self.vocab.start_index]])).cuda()
        end_token = torch.LongTensor([[self.vocab.end_index]]).cuda()
        context_tokens = variable.Variable(torch.LongTensor(context_tokens)).cuda()
        answer_features = variable.Variable(torch.from_numpy(answer_features)).cuda()
        
        predictions = self.model.predict(input_token=input_token, 
            context_tokens=context_tokens, 
            end_token=end_token,
            answer_features=answer_features,
            max_length=max_length)

        if pad:
            pad_token = variable.Variable(torch.LongTensor([self.vocab.pad_index]).cuda())
            while len(predictions) < max_length:
                predictions.append(pad_token)

        stacked_predictions = torch.stack(predictions, 0)
        tokens = self.get_tokens_single(stacked_predictions.cpu())
        sentence = " ".join(tokens)
        return sentence, stacked_predictions

    def get_tokens_single(self, predictions):
        numpy_predictions = torch.squeeze(predictions).data.numpy()
        tokens = self.vocab.tokens(numpy_predictions)
        return tokens 

    def get_tokens(self, predictions):
        numpy_predictions = torch.squeeze(predictions).data.numpy()
        tokens = self.vocab.tokens_list(numpy_predictions)
        return tokens