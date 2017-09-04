from models.language_model import LanguageModel
from models.language_wrapper import LanguageWrapper
import torch 
from torch import nn
from torch import optim
from torch.autograd import variable 
from helpers import constants, utils, torch_utils, logger
import gc 

class LanguageTrainer(object):
    def __init__(self, config, language_wrapper, data_loader):
        self.config = config
        self.logger = logger.FileLogger(config['log_path'])
        self.language_wrapper = language_wrapper
        self.language_model = language_wrapper.get_model()
        self.data_loader = data_loader

        self.init_trainer()

    def init_trainer(self):
        self.optimizer = optim.Adam(self.language_model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.NLLLoss()

    def get_path(self, epoch_num):
        save_dir = self.config['save_directory']
        save_path = '%s/%s' % (save_dir, 'model_%s.pyt7' % epoch_num)
        return save_path 

    def save_predictions(self, epoch_num, predictions):
        save_dir = self.config['save_directory']
        save_path = '%s/%s' % (save_dir, 'predictions_%s' % epoch_num)
        utils.save_lines(predictions, save_path)

    def save(self, epoch_num):
        save_path = self.get_path(epoch_num)
        #additional_args = {}
        #additional_args['vocab'] = self.data_loader.get_vocab()
        torch_utils.save_model(self.language_model, save_path)

    def load(self, epoch_num):
        load_path = self.get_path(epoch_num)
        self.language_model = torch_utils.load_model(load_path).cuda()
        self.init_trainer() # Need to reset optimizer 

    def predict(self, dataset_type, epoch_num, max_length):
        self.data_loader.reset_indices()
        batch_size = 1
        batch = self.data_loader.get_batch(dataset_type=dataset_type,
                batch_size=batch_size)

        total_predictions = []
        num_examples = 0
        while batch is not None: #and num_examples < 200:
            num_examples += batch_size
            if num_examples % 100 == 0:
                print("On example %s" % num_examples)

            prediction, _ = self.language_wrapper.predict(batch['context_tokens'], batch['answer_features'], max_length)
            total_predictions.append(prediction)
            batch = self.data_loader.get_batch(dataset_type=dataset_type,
                batch_size=batch_size)

        return total_predictions

    def train(self, epoch_num):
        self.data_loader.mix_indices()
        batch = self.data_loader.get_batch(dataset_type=constants.DATASET_TRAIN,
                batch_size=self.config['batch_size'])
        batch_size = len(batch['input_tokens'][0])

        total_loss = 0.0
        total_accuracy = 0.0
        num_examples = 0
        total_predictions = []

        num_steps = 0
        while batch is not None:
            num_steps = num_steps + 1
            if num_steps % 20 == 0:
                gc.collect()
            batch_size = len(batch['input_tokens'][0])
            loss = self.step(batch)
            num_examples += batch_size
            #total_loss += batch_size * loss.cpu().data.numpy()
            #total_accuracy += batch_size * accuracy.numpy()
            #total_predictions.extend(predictions)
            batch = self.data_loader.get_batch(dataset_type=constants.DATASET_TRAIN,
                batch_size=self.config['batch_size'])

            msg = "Loss %s num examples %s" % (loss.cpu().data.numpy(), num_examples)
            print(msg)
            self.logger.write(msg)

        average_loss = total_loss / num_examples
        average_accuracy = total_accuracy / num_examples 
        return average_loss, average_accuracy, total_predictions

    def step(self, batch, train=True):
        inputs = variable.Variable(torch.from_numpy(batch['input_tokens'])).cuda()
        desired_inputs = variable.Variable(torch.from_numpy(batch['desired_input_tokens'])).cuda()
        desired_input_masks = variable.Variable(torch.from_numpy(batch['desired_input_masks'])).cuda()
        contexts = variable.Variable(torch.from_numpy(batch['context_tokens'])).cuda()
        context_masks = variable.Variable(torch.from_numpy(batch['context_masks'])).cuda()
        answer_features = variable.Variable(torch.from_numpy(batch['answer_features'])).cuda()

        language_probs = self.language_model.forward(inputs, contexts, context_masks, answer_features)
        reshaped_inputs = desired_inputs.contiguous().view(-1)
        reshaped_language_probs = language_probs.view(-1, self.config['vocab_size'])

        max_likelihoods, best_indices = torch.max(language_probs, 2)
        #accuracy = torch_utils.average_accuracy(torch.squeeze(best_indices).data, desired_inputs.data)

        #predictions = self.language_wrapper.get_tokens(best_indices.cpu())
        #predictions_text = utils.transpose_join(predictions, " ")
        loss = 0 
        select_indices = torch_utils.get_index_select(desired_input_masks).cuda()
        gathered_indices = torch.index_select(reshaped_inputs, 0, select_indices)
        gathered_probs = torch.index_select(reshaped_language_probs, 0, select_indices)

        if train:
            self.optimizer.zero_grad()
            if not self.config['finetune_embeddings']:
                inputs.detach()
                contexts.detach()
                answer_features.detach()
            """
            batch_size = language_probs.size(1)
            for i in range(0, language_probs.size(1)):
                cur_language_probs = language_probs[:, i, :]
                cur_desired_inputs = desired_inputs[:, i]
                cur_lengths = batch['desired_input_lengths'][i]

                truncated_language_probs = cur_language_probs[0:cur_lengths, :]
                truncated_desired_inputs = cur_desired_inputs[0:cur_lengths]

                loss = self.criterion(truncated_language_probs, truncated_desired_inputs)
                if i == batch_size - 1:
                    loss.backward()
                else:
                    loss.backward(retain_variables=True)
            """

            loss = self.criterion(gathered_probs, gathered_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.language_model.parameters(), 5)
            self.optimizer.step()


        return loss#, accuracy, predictions_text
