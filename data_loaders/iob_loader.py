from helpers import utils
from helpers import tokenizer 
from helpers import vocab 
from helpers import constants 
import numpy as np 

class IOBLoader(object):
    def __init__(self, base_path, input_max_length=300, tokenizer_type=constants.TOKENIZER_NLTK):
        self.base_path = base_path
        self.load_vocabulary()
        self.train_dataset = self.load_dataset(dataset_directory='%s/train' % base_path, tokenizer_type=tokenizer_type)
        self.validation_dataset = self.load_dataset(dataset_directory='%s/validation' % base_path, tokenizer_type=tokenizer_type)
        self.test_dataset = self.load_dataset(dataset_directory='%s/test' % base_path, tokenizer_type=tokenizer_type)

        self.train_indices = range(len(self.train_dataset['inputs']))
        self.validation_indices = range(len(self.validation_dataset['inputs']))
        self.test_indices = range(len(self.test_dataset['inputs']))

        dataset_input_max_length = np.max([self.train_dataset['input_max_length'], \
            self.validation_dataset['input_max_length'], \
            self.test_dataset['input_max_length']])
        self.input_max_length = int(np.min([input_max_length, dataset_input_max_length]))

    def get_vocab(self):
        return self.vocab

    def get_batch(self, dataset_type, batch_size):
        """ Returns a batch of size batch_size of 
        questions, entities, predicates to train/eval/test on"""

        if dataset_type == constants.DATASET_TRAIN:
            self.cur_train_index, batch = self.get_batch_from_dataset(\
                self.train_dataset, \
                self.train_indices, \
                self.cur_train_index, \
                batch_size)
        elif dataset_type == constants.DATASET_VALIDATION:
            self.cur_validation_index, batch = self.get_batch_from_dataset(\
                self.validation_dataset, \
                self.validation_indices, \
                self.cur_validation_index, \
                batch_size)
        elif dataset_type == constants.DATASET_TEST:
            self.cur_test_index, batch = self.get_batch_from_dataset(\
                self.test_dataset, \
                self.test_indices, \
                self.cur_test_index, \
                batch_size)
        else:
            raise Exception("Invalid dataset type given %s" % dataset_type)
        return batch


    def get_batch_from_dataset(self, dataset, indices, \
        cur_index, batch_size):
        """ Gets specific batch from dataset 
        TODO: Add sampling of negatives of current context
        """

        raw_inputs = [] 
        raw_labels = []

        labels = np.zeros((batch_size, self.input_max_length), dtype=np.int32)
        inputs = np.zeros((batch_size, self.input_max_length), dtype=np.int32)
        input_lengths = np.zeros((batch_size), dtype=np.int32)
        input_masks = np.zeros((batch_size, self.input_max_length), dtype=np.int32)

        # Filler array for question inputs
        max_index = np.min([cur_index + batch_size, dataset['size']])
        if max_index <= cur_index or max_index == dataset['size']: return None, None

        for i in range(cur_index, max_index):
            #print(i)
            cur_batch_index = i - cur_index
            cur_input_index = indices[i]

            cur_inputs = dataset['inputs'][cur_input_index]
            cur_labels = dataset['labels'][cur_input_index]

            raw_inputs.append(cur_inputs)
            raw_labels.append(cur_labels)

            cur_input_tokens = dataset['input_tokens'][cur_input_index]
            cur_label_tokens = dataset['label_tokens'][cur_input_index]
            cur_input_lengths = np.min([len(cur_input_tokens), self.input_max_length])
            

            for j in range(0, cur_input_lengths):
                inputs[cur_batch_index][j] = cur_input_tokens[j]
                labels[cur_batch_index][j] = cur_label_tokens[j]

            input_masks[cur_batch_index][0:cur_input_lengths] = 1 
            input_lengths[cur_batch_index] = cur_input_lengths

        batch = {}
        batch['input_lengths'] = input_lengths
        batch['input_tokens'] = inputs
        batch['input_masks'] = input_masks
        batch['label_tokens'] = labels
        batch['labels'] = raw_labels 
        batch['inputs'] = raw_inputs

        return max_index, batch

    def reset_indices(self):
        self.cur_train_index = 0
        self.train_indices = range(0, len(self.train_indices))

        self.cur_validation_index = 0
        self.validation_indices = range(0, len(self.validation_indices))

        self.cur_test_index = 0
        self.test_indices = range(0, len(self.test_indices))

    def mix_indices(self):
        self.cur_train_index = 0
        self.train_indices = np.random.permutation(self.train_indices)

        self.cur_validation_index = 0
        self.validation_indices = np.random.permutation(self.validation_indices)

        self.cur_test_index = 0
        self.test_indices = np.random.permutation(self.test_indices)


    def load_vocabulary(self):
        base_path = self.base_path
        vocab_path = '%s/vocab.txt' % base_path
        label_vocab_path = '%s/label_vocab.txt' % base_path

        self.vocab = vocab.Vocab(vocab_type=constants.WORD_LEVEL)
        self.vocab.init_from_path(path=vocab_path)

        self.label_vocab = vocab.Vocab(vocab_type=constants.WORD_LEVEL, add_start_end=False)
        self.label_vocab.init_from_path(path=label_vocab_path)

    def load_dataset(self, dataset_directory, tokenizer_type):
        input_lines = utils.read_lines('%s/%s' % (dataset_directory, 'inputs.txt'))
        label_lines = utils.read_lines('%s/%s' % (dataset_directory, 'labels.txt'))

        input_tokens = list(map(lambda l: tokenizer.tokenize_sentence(l, 
            vocab=self.vocab,
            tokenizer_type=tokenizer_type,
            add_start_end=False), input_lines))
        label_tokens = list(map(lambda l: tokenizer.tokenize_sentence(l,
            vocab=self.label_vocab,
            tokenizer_type=tokenizer_type,
            add_start_end=False), label_lines))

        dataset = {}
        dataset['input_tokens'] = input_tokens
        dataset['label_tokens'] = label_tokens
        dataset['input_max_length'] = np.max(list(map(lambda l: len(l), input_tokens)))
        dataset['inputs'] = input_lines 
        dataset['labels'] = label_lines
        dataset['size'] = len(input_lines)
        return dataset

