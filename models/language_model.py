import numpy as np
import torch 
import torch.nn as nn 
import torch.optim
from torch.autograd import variable
from helpers import constants, torch_utils, utils
from dnn_units.lstm_attention import LSTMAttentionDot, SoftDotAttention

class Encoder(nn.Module):
    def __init__(self, config, use_features=True):
        super(Encoder, self).__init__()
        self.config = config
        if use_features:
            input_size = config['embedding_size'] + 1
        else:
            input_size = config['embedding_size']
        hidden_size = int(config['hidden_size'] / 2)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=config['num_layers'], dropout=config['dropout'],
                        bidirectional=True, batch_first=False)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config['num_layers'] * 2, batch_size, int(self.config['hidden_size'] / 2)
        h0 = variable.Variable(inputs.data.new(*state_shape).zero_(), requires_grad=False).cuda()
        c0 = variable.Variable(inputs.data.new(*state_shape).zero_(), requires_grad=False).cuda()

        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return outputs, ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class LanguageDiscriminator(nn.Module):
    def __init__(self, config, 
        embeddings, 
        text_field_predictor, 
        base_lstm):
        super(LanguageDiscriminator, self).__init__()
        self.config = config
        self.embeddings = embeddings 
        self.text_field_predictor = text_field_predictor 
        self.base_lstm = base_lstm
        self.predictor = self.get_predictor()

    def get_predictor(self):
        fc = nn.Sequential(
                nn.Linear(self.config['hidden_size'], 1),
                nn.Sigmoid())
        return fc 

    def forward(self, inputs, contexts, answer_features):
        context_embeddings = self.text_field_predictor.forward_prepro(contexts, input_masks=None, answer_features=answer_features)
        input_embeddings = self.embeddings(inputs)
        batch_size = inputs.size(1)
        state_shape = (batch_size, self.config['hidden_size'])
        h0 = c0 = variable.Variable(input_embeddings.data.new(*state_shape).zero_()).cuda()
        cur_states = (h0, c0)

        out, hidden = self.base_lstm.forward(input_embeddings, cur_states, context_embeddings)

        h = hidden[0]
        pred = torch.squeeze(self.predictor(h))
        return pred 

class TextFieldPredictor(nn.Module):
    def __init__(self, config, embeddings):
        super(TextFieldPredictor, self).__init__()
        self.config = config
        self.embeddings = embeddings
        self.encoder = Encoder(config)
        self.attention = SoftDotAttention(config['hidden_size'])

    def forward_prepro(self, input, input_masks, answer_features=None):
        self.input = input
        self.input_masks = input_masks
        input_embeddings = self.embeddings(input)

        if answer_features is not None:
            unsqueezed_answer_features = torch.unsqueeze(answer_features, 2)
            self.input_embeddings = torch.cat((input_embeddings, unsqueezed_answer_features), 2)
        else:
            self.input_embeddings = input_embeddings

        self.lstm_embeddings, _ = self.encoder(self.input_embeddings) # Assumes NOT batch first

        # Get batch first lstm embeddings
        self.batch_first_lstm_embeddings = self.lstm_embeddings.transpose(0, 1)
        return self.lstm_embeddings 

    def forward_similarity(self, hidden_state):
        # print(self.batch_first_lstm_embeddings.size())
        # print(hidden_state.size())
        h_tilde, attentions = self.attention(hidden_state, 
            self.batch_first_lstm_embeddings, self.input_masks)
        return h_tilde, torch.log(attentions), self.input


class SoftmaxPredictor(nn.Module):
    def __init__(self, config):
        super(SoftmaxPredictor, self).__init__()
        self.config = config
        self.projection = nn.Linear(config['hidden_size'], config['vocab_size'])
        self.log_softmax = nn.LogSoftmax()

    def forward(self, hidden_state):
        return self.log_softmax(self.projection(hidden_state))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config
        self.model_type = constants.MODEL_TYPE_LANGUAGE_MODEL
        self.init_predictors()
        self.init_base_lstm()

    def get_type(self):
        return self.model_type 

    def init_predictors(self):
        self.embedder = self.get_embedder()
        self.combiner = self.get_combiner()
        self.text_field_predictor = TextFieldPredictor(self.config, self.embedder)
        self.softmax_predictor = SoftmaxPredictor(self.config)

    def get_embedder(self):
        embedder = nn.Embedding(self.config['vocab_size'], self.config['embedding_size'])
        if self.config['use_pretrained_embeddings']:
            embeddings = utils.load_matrix(self.config['pretrained_embeddings_path'])
            embedder.weight.data.copy_(torch.from_numpy(embeddings))
        return embedder

    def get_combiner(self):
        combiner = nn.Sequential(
            nn.Linear(self.config['hidden_size'], self.config['hidden_size']),
            nn.Tanh(),
            nn.Linear(self.config['hidden_size'], 2),
            nn.LogSoftmax())
        return combiner 

    def init_base_lstm(self):
        self.base_lstm = LSTMAttentionDot(input_size=self.config['embedding_size'], 
            hidden_size=self.config['hidden_size'], 
            batch_first=self.config['batch_first'])

    def predict(self, input_token, 
        context_tokens, 
        end_token, 
        answer_features,
        max_length=20,
        min_length=3):
        """
        input_token: Input token to start with 
        context_tokens: Context tokens to use
        Do greedy decoding using input token and context tokens
        """

        predicted_tokens = []
        total_loss = 0.0 

        batch_first_context_tokens = context_tokens.transpose(0, 1)
        context_embeddings = self.text_field_predictor.forward_prepro(context_tokens, input_masks=None, 
            answer_features=answer_features)

        state_shape = (1, self.config['hidden_size'])
        h0 = c0 = variable.Variable(context_embeddings.data.new(*state_shape).zero_())
        cur_states = (h0, c0)

        def step(input_token, states):
            cur_input_embedding = self.embedder(input_token)
            hidden_states, new_states = self.base_lstm.forward(cur_input_embedding, \
                states, context_embeddings)

            reshaped_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            predictor_probs = self.combiner(reshaped_hidden_states)

            language_probs = self.softmax_predictor(reshaped_hidden_states)
            reshaped_language_probs = language_probs.view(-1, language_probs.size(-1))
            
            _, attentions, inputs = self.text_field_predictor.forward_similarity(hidden_states)

            combined_predictions = self.combine_predictions_single(\
                context_tokens=batch_first_context_tokens, 
                predictor_probs=predictor_probs,
                attentions=attentions, 
                language_probs=reshaped_language_probs)

            loss, token = torch.max(combined_predictions, 1)
            return loss, token, new_states 

        loss, new_token, new_states = step(input_token, cur_states)

        while (not torch_utils.to_bool(new_token.data == end_token) or 
            len(predicted_tokens) < min_length) and len(predicted_tokens) < max_length:
            predicted_tokens.append(new_token)
            loss, new_token, new_states = step(new_token, new_states)
        return predicted_tokens


    def forward(self, input_tokens, context_tokens, context_masks, answer_features):
        self.batch_first_context_tokens = context_tokens.transpose(0, 1)
        self.context_embeddings = self.text_field_predictor.forward_prepro(context_tokens, context_masks, answer_features)
        self.input_embeddings = self.embedder(input_tokens)
        
        batch_size = input_tokens.size(1)
        token_length = input_tokens.size(0)

        state_shape = (batch_size, self.config['hidden_size'])
        h0 = c0 = variable.Variable(self.input_embeddings.data.new(*state_shape).zero_(), requires_grad=False)


        hidden_states, res = self.base_lstm.forward(self.input_embeddings, \
            (h0, c0), \
            self.context_embeddings)

        reshaped_hidden_states = hidden_states.view(batch_size * token_length, -1)
        predictor_probs = self.combiner(reshaped_hidden_states)
        reshaped_predictor_probs = predictor_probs.view(token_length, batch_size, predictor_probs.size(-1))

        language_probs = self.softmax_predictor(reshaped_hidden_states)
        reshaped_language_probs = language_probs.view(token_length, batch_size, language_probs.size(-1))
        
        attentions_list = []
        for i in range(0, token_length):
            _, attentions, inputs = self.text_field_predictor.forward_similarity(hidden_states[i, :, :])
            attentions_list.append(attentions)
        attentions_sequence = torch.stack(attentions_list, 0)

        combined_predictions = self.combine_predictions(context_tokens=self.batch_first_context_tokens, 
            predictor_probs=reshaped_predictor_probs,
            attentions=attentions_sequence, 
            language_probs=reshaped_language_probs)

        #return reshaped_language_probs
        return combined_predictions

    def combine_predictions_single(self, context_tokens,
        predictor_probs,
        attentions,
        language_probs):

        max_attention_length = attentions.size(1)
        pad_size = self.config['vocab_size'] - max_attention_length
        batch_size = attentions.size(0)

        context_tokens_padding = variable.Variable(torch.LongTensor(batch_size, pad_size).zero_()).cuda()
        attentions_padding = variable.Variable(torch.zeros(batch_size, pad_size)).cuda() + -1e10
        stacked_context_tokens = torch.cat((context_tokens, context_tokens_padding), 1)

        softmax_probs = predictor_probs[:, 0]
        text_field_probs = predictor_probs[:, 1]

        replicated_softmax_probs = softmax_probs.unsqueeze(1)
        replicated_text_field_probs = text_field_probs.unsqueeze(1)

        dims = replicated_softmax_probs.size()
        dims1 = replicated_text_field_probs.size()

        expanded_softmax_probs = replicated_softmax_probs.expand(dims[0],  self.config['vocab_size'])
        expanded_text_field_probs = replicated_text_field_probs.expand(dims[0], max_attention_length)

        stacked_attentions = torch.cat((attentions, attentions_padding), 1)            
        attention_results = variable.Variable(torch.zeros(batch_size, self.config['vocab_size'])).cuda() + -1e10

        attention_results.scatter_(1, stacked_context_tokens, stacked_attentions)
        use_softmax_predictor = softmax_probs > text_field_probs
        if torch_utils.to_bool(use_softmax_predictor.data):
            return language_probs
        else:
            return attention_results


    def combine_predictions(self, context_tokens, 
        predictor_probs, 
        attentions, 
        language_probs):

        max_attention_length = attentions.size(2)
        pad_size = self.config['vocab_size'] - max_attention_length
        batch_size = attentions.size(1)
        seq_size = attentions.size(0)

        context_tokens_padding = variable.Variable(torch.LongTensor(batch_size, pad_size).zero_(), requires_grad=False).cuda()
        attentions_padding = variable.Variable(torch.zeros(batch_size, pad_size) + -1e10, requires_grad=False).cuda()
        stacked_context_tokens = torch.cat((context_tokens, context_tokens_padding), 1)

        total_attention_results = []
        softmax_probs = predictor_probs[:, :, 0]
        text_field_probs = predictor_probs[:, :, 1]

        replicated_softmax_probs = softmax_probs.unsqueeze(2)
        replicated_text_field_probs = text_field_probs.unsqueeze(2)

        dims = replicated_softmax_probs.size()
        dims1 = replicated_text_field_probs.size()

        expanded_softmax_probs = replicated_softmax_probs.expand(dims[0], dims[1], self.config['vocab_size'])
        expanded_text_field_probs = replicated_text_field_probs.expand(dims[0], dims[1], max_attention_length)

        for i in range(0, seq_size):
            selected_text_field_probs = expanded_text_field_probs[i, :, :]
            selected_attention = attentions[i, :, :] + selected_text_field_probs
            stacked_attentions = torch.cat((selected_attention, attentions_padding), 1)            

            attention_results = variable.Variable(torch.zeros(batch_size, self.config['vocab_size']) + -1e10).cuda()
            attention_results.scatter_(1, stacked_context_tokens, stacked_attentions)
            total_attention_results.append(attention_results)

        concated_attention_results = torch.stack(total_attention_results, 0)    
        final_probs = torch.log(torch.exp(concated_attention_results) + torch.exp(language_probs + expanded_softmax_probs))

        return final_probs




