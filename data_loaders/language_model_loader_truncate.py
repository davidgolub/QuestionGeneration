from helpers import utils
from helpers import tokenizer 
from helpers import vocab 
from helpers import constants 
import numpy as np 

class LanguageModelLoaderTruncate(object):
    def __init__(self, base_path, tokenizer_type=constants.TOKENIZER_NLTK):
        self.base_path = base_path
        if not utils.check_file('%s/vocab.txt' % self.base_path):
            self.create_vocabulary()

        self.load_vocabulary()
        self.train_dataset = self.load_dataset(dataset_directory='%s/train' % base_path, tokenizer_type=tokenizer_type)
        self.validation_dataset = self.load_dataset(dataset_directory='%s/validation' % base_path, tokenizer_type=tokenizer_type)
        self.test_dataset = self.load_dataset(dataset_directory='%s/test' % base_path, tokenizer_type=tokenizer_type)

        self.train_indices = range(len(self.train_dataset['inputs']))
        self.validation_indices = range(len(self.validation_dataset['inputs']))
        self.test_indices = range(len(self.test_dataset['inputs']))

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

    def truncate_context(self, context, answer_start, answer_end):
        period_index = self.vocab.index(".")
        period_indices = utils.get_indices(context, period_index)

        smaller_periods = list(filter(lambda idx: idx <= answer_start, period_indices))
        larger_periods = list(filter(lambda idx: idx >= answer_end, period_indices))

        start_idx = 1 
        end_idx = len(context)

        if len(smaller_periods) > 0:
            smaller_period = -1
            if len(smaller_periods) > 1:
                smaller_period = smaller_periods[-2]
            else:
                smaller_period = smaller_periods[-1]
            start_idx = smaller_period + 1

        if len(larger_periods) > 0:
            larger_period = larger_periods[0]
            end_idx = larger_period + 1


        truncated_context = context[start_idx:end_idx] + [self.vocab.end_index]
        if start_idx != 0: 
            # Make sure to add <START> token 
            truncated_context = [self.vocab.start_index] + truncated_context
        return start_idx - 1, truncated_context

    def get_batch_from_dataset(self, dataset, indices, \
        cur_index, batch_size):
        """ Gets specific batch from dataset 
        TODO: Add sampling of negatives of current context
        """

        # Filler array for question inputs
        max_index = np.min([cur_index + batch_size, dataset['size']])
        if max_index <= cur_index or max_index == dataset['size']: return None, None

        contexts = list(map(lambda idx: dataset['context_tokens'][dataset['indices'][indices[idx]]], \
            range(cur_index, max_index)))
        answer_starts = list(map(lambda idx: dataset['answer_starts'][indices[idx]], \
            range(cur_index, max_index)))
        answer_ends = list(map(lambda idx: dataset['answer_ends'][indices[idx]], \
            range(cur_index, max_index)))

        # Truncate conexts 
        start_indices = []
        truncated_contexts = []
        truncated_answer_starts = []
        truncated_answer_ends = []

        for ans_start, ans_end, context in zip(answer_starts, answer_ends, contexts):
            start_idx, truncated_ctx = self.truncate_context(context, ans_start, ans_end)
            start_indices.append(start_idx)
            truncated_contexts.append(truncated_ctx)
            truncated_answer_starts.append(ans_start - start_idx)
            truncated_answer_ends.append(ans_end - start_idx)

        # Truncate contexts
        input_lengths = list(map(lambda idx: len(dataset['input_tokens'][indices[idx]]), \
            range(cur_index, max_index)))
        context_lengths = list(map(lambda ctx: len(ctx), truncated_contexts))
        input_max_length = np.max(input_lengths)
        context_max_length = np.max(context_lengths)

        inputs = np.zeros((batch_size, input_max_length), dtype=np.int)
        input_lengths = np.zeros((batch_size), dtype=np.int)
        input_masks = np.ones((batch_size, input_max_length), dtype=np.int)

        desired_inputs = np.ones((batch_size, input_max_length), \
            dtype=np.int)

        context_lengths = np.zeros((batch_size), dtype=np.int)
        contexts = np.zeros((batch_size, context_max_length), dtype=np.int)
        context_masks = np.ones((batch_size, context_max_length), dtype=np.int)

        answer_features = np.zeros((batch_size, context_max_length), dtype=np.float32)

        raw_inputs = []
        raw_contexts = []
        raw_desired_inputs = []

        for i in range(cur_index, max_index):
            #print(i)
            cur_batch_index = i - cur_index
            cur_input_index = indices[i]

            cur_inputs = dataset['inputs'][cur_input_index]
            cur_contexts = dataset['contexts'][dataset['indices'][cur_input_index]]
            untruncated_cur_contexts = dataset['context_tokens'][dataset['indices'][cur_input_index]] 
            cur_desired_inputs = dataset['desired_inputs'][cur_input_index]

            raw_inputs.append(cur_inputs)
            raw_contexts.append(cur_contexts)
            raw_desired_inputs.append(cur_desired_inputs)

            cur_input_tokens = dataset['input_tokens'][cur_input_index]
            cur_desired_input_tokens = dataset['desired_input_tokens'][cur_input_index]
            cur_context_tokens = truncated_contexts[cur_batch_index]

            cur_answer_start = dataset['answer_starts'][cur_input_index]
            cur_answer_end = dataset['answer_ends'][cur_input_index]

            for j in range(0, len(cur_input_tokens)):
                inputs[cur_batch_index][j] = cur_input_tokens[j]
                desired_inputs[cur_batch_index][j] = cur_desired_input_tokens[j]

            input_lengths[cur_batch_index] = len(cur_input_tokens)
            input_masks[cur_batch_index][0:len(cur_input_tokens)] = 0

            context_lengths[cur_batch_index] = len(cur_context_tokens)
            context_masks[cur_batch_index][0:len(cur_context_tokens)] = 0

            for j in range(0, len(cur_context_tokens)):
                contexts[cur_batch_index][j] = cur_context_tokens[j]

            if 'answer_starts' in dataset:
                truncated_answer_start = truncated_answer_starts[cur_batch_index]
                truncated_answer_end = truncated_answer_ends[cur_batch_index]

                answer_features[cur_batch_index][truncated_answer_start:truncated_answer_end] = 1.0
                mapped_tokens = self.vocab.tokens(cur_context_tokens)
                original_answer_features = self.vocab.tokens(untruncated_cur_contexts[cur_answer_start:cur_answer_end])
                truncated_answer_features = mapped_tokens[truncated_answer_start:truncated_answer_end]
                
                """
                print("TESTTSOIDFJOISDFJODSJOIF")
                print(cur_inputs)
                print(original_answer_features)
                print(truncated_answer_features)
                print(mapped_tokens)
                print("TEST123")
                print(truncated_answer_start)
                print(truncated_answer_end)
                print("TEST12345")
                print(cur_answer_start)
                print(cur_answer_end)
                print(cur_context_tokens)
                """


        # Transpose so sequence length first
        length_first_inputs = inputs.T
        length_first_input_masks = input_masks.T 
        length_first_desired_inputs = desired_inputs.T
        length_first_desired_masks = input_masks.T
        length_first_contexts = contexts.T
        length_first_context_masks = context_masks.T 
        length_first_answer_features = answer_features.T

        batch = {}
        batch['answer_features'] = length_first_answer_features
        batch['context_lengths'] = context_lengths
        batch['context_masks'] = length_first_context_masks
        batch['context_tokens'] = length_first_contexts
        batch['input_lengths'] = input_lengths
        batch['input_tokens'] = length_first_inputs
        batch['input_masks'] = length_first_input_masks
        batch['desired_input_lengths'] = input_lengths
        batch['desired_input_tokens'] = length_first_desired_inputs
        batch['desired_input_masks'] = length_first_input_masks
        batch['contexts'] = raw_contexts 
        batch['inputs'] = raw_inputs 
        batch['desired_inputs'] = raw_desired_inputs 

        return max_index, batch

    def create_vocabulary(self):
        base_path = self.base_path
        tokens_list = []
        train_dataset = self.read_raw_lines('%s/train' % base_path)
        validation_dataset = self.read_raw_lines('%s/validation' % base_path)
        test_dataset = self.read_raw_lines('%s/test' % base_path)

        for dataset in [train_dataset, validation_dataset, test_dataset]:
            inputs = dataset['inputs']
            outputs = dataset['outputs']
            for inp in inputs:
                tokens = tokenizer.split_sentence(inp, tokenizer_type=constants.TOKENIZER_NLTK)
                tokens_list.extend(tokens)
            
            for out in outputs:
                tokens = tokenizer.split_sentence(out, tokenizer_type=constants.TOKENIZER_NLTK)
                tokens_list.extend(tokens)

        vocab = utils.unique_vals(tokens_list, min_num_occurences=0)
        utils.save_lines(vocab, '%s/vocab.txt' % base_path)

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

    def read_raw_lines(self, base_path):
        inputs_path = '%s/inputs.txt' % base_path 
        outputs_path = '%s/outputs.txt' % base_path

        inputs = utils.read_lines(inputs_path)
        outputs = utils.read_lines(outputs_path)

        dataset = {}
        dataset['inputs'] = inputs
        dataset['outputs'] = outputs
        return dataset

    def load_vocabulary(self):
        base_path = self.base_path
        vocab_path = '%s/vocab.txt' % base_path

        self.vocab = vocab.Vocab(vocab_type=constants.WORD_LEVEL)
        self.vocab.init_from_path(path=vocab_path)

    def load_dataset(self, dataset_directory, tokenizer_type):
        input_lines = utils.read_lines('%s/%s' % (dataset_directory, 'inputs.txt'))
        output_lines = utils.read_lines('%s/%s' % (dataset_directory, 'outputs.txt'))
        indices_lines = utils.read_lines('%s/%s' % (dataset_directory, 'indices.txt'))
        
        context_tokens = list(map(lambda l: tokenizer.tokenize_sentence(l, 
            vocab=self.vocab,
            tokenizer_type=constants.TOKENIZER_SPECIAL_DELIMITER,
            add_start_end=True), input_lines))
        output_tokens = list(map(lambda l: tokenizer.tokenize_sentence(l,
            vocab=self.vocab,
            tokenizer_type=tokenizer_type,
            add_start_end=True), output_lines))
        input_tokens = list(map(lambda tokens_list: tokens_list[0:-1], output_tokens))
        desired_input_tokens = list(map(lambda tokens_list: tokens_list[1:], output_tokens))

        indices_vals = list(map(lambda l: int(l), indices_lines))

        answer_starts_path = '%s/%s' % (dataset_directory, 'answer_starts.txt')
        answer_ends_path = '%s/%s' % (dataset_directory, 'answer_ends.txt')

        dataset = {}
        if utils.check_file(answer_starts_path):
            answer_starts = utils.read_lines(answer_starts_path)
            answer_starts = list(map(lambda l: int(l) + 1, answer_starts)) # We're adding <START>
            dataset['answer_starts'] = answer_starts
        
        if utils.check_file(answer_ends_path):
            answer_ends = utils.read_lines(answer_ends_path)
            answer_ends = list(map(lambda l: int(l) + 1, answer_ends)) # We're adding <END>
            dataset['answer_ends'] = answer_ends
      
        dataset['indices'] = indices_vals
        dataset['input_tokens'] = input_tokens
        dataset['desired_input_tokens'] = desired_input_tokens
        dataset['context_tokens'] = context_tokens
        dataset['contexts'] = input_lines
        dataset['inputs'] = output_lines
        dataset['desired_inputs'] = output_lines
        dataset['size'] = len(output_lines)
        return dataset

