import os
import os.path
import urllib
import mimetypes
import time
import random
import numpy
import collections
from helpers import utils
from helpers import constants
import pickle
import numpy as np
from helpers import io_utils
from helpers import tokenizer

class Vocab(object):
	""" A Vocab object maps tokens to integer indices and back """
	def __init__(self, vocab_type, use_unk_token=True, add_start_end=True):
		self.name = 'Vocab'
		self.vocab_type = vocab_type
		self.use_unk_token = use_unk_token
		self.add_start_end = add_start_end
		self.weights = None
		self.embeddings = None
		self.init_structs()

	def save(self, save_path):
		# Saves vocab into specified path
		io_utils.pickle_save(data=self, save_path=save_path)

	@staticmethod
	def load(load_path):
		print("Loading vocab from path %s" % load_path)
		vocab = io_utils.pickle_load(load_path)
		return vocab

	@staticmethod
	def load_pretrained_vocab():
		""" Loads pretrained word embeddings 
		(currently dependency embeddings) """
		pretrained_vocab = Vocab(vocab_type=constants.WORD_LEVEL)
		pretrained_vocab.init_from_path(path=constants.PRETRAINED_VOCAB_PATH)
		pretrained_vocab.init_embeddings(embeddings_path=constants.PRETRAINED_EMBEDDINGS_PATH)
		return pretrained_vocab

	@staticmethod
	# Loads part of speech vocab
	def load_pos_vocab():
		""" Loads part-of-speech vocab
		(currently dependency embeddings) """
		pos_vocab = Vocab(vocab_type=constants.WORD_LEVEL)
		pos_vocab.init_from_path(path=constants.POS_VOCAB_PATH)
		return pos_vocab

	def create_embeddings(self, save_path, initializer=constants.INITIALIZER_ZERO):
		"""
		Creates an embedding array and saves to specified 
		save_embeddings_path loads pretrained vocab and 
		populates all parts of array to
		"""
		print("Creating and saving embedding to %s" % save_path)
		assert self.array is not None
		
		pretrained_vocab = Vocab.load_pretrained_vocab()
		embedding_size = pretrained_vocab.embed_dim
		new_embed_array = None

		if initializer == constants.INITIALIZER_ZERO:
			new_embed_array = np.zeros((self.size(), embedding_size))
		elif initializer == constants.INITIALIZER_UNIFORM_RANDOM:
			new_embed_array = (np.random.rand(self.size(), embedding_size) - 0.5) / 100.0
		else: 
			raise Exception("Unknown initializer given %s" % initializer)

		num_added = 0
		for i in range(0, self.size()):
			if i % 100 == 0:
				print("On index %s" % i)
			cur_token = self.token(i)
			if pretrained_vocab.contains(cur_token):
				cur_idx = pretrained_vocab.index(cur_token)
				cur_embedding = pretrained_vocab.embeddings[cur_idx]
				new_embed_array[i] = cur_embedding
				num_added = num_added + 1

		utils.save_matrix(new_embed_array, save_path)
		print("Number added %s" % num_added)

	def init_embeddings(self, embeddings_path, path_type=constants.PATH_NPY_ARRAY):
		print("Initializing embeddings from %s" % embeddings_path)
		self.embeddings = utils.load_matrix(path=embeddings_path, path_type=path_type)
		self.embed_dim = np.size(self.embeddings, 1)
		print("Done initializing embeddings from %s" % embeddings_path)

	def get_embeddings(self):
		return self.embeddings

	def embedding_size(self):
		print(self.embed_dim)
		return self.embed_dim

	def init_from_path(self, path):
		"""
		Read lines (one vocab token per line) into counters
		"""
		self.path = path
		lines = utils.read_lines(path)
		self.init_from_array(lines)

	def string_to_tokens(self, text, delimiter, add_start_end, use_tokenizer=False):
		tokens = self.tokenize_text(text=text, delimiter=delimiter, use_tokenizer=use_tokenizer)

		if self.vocab_type == constants.WORD_CHAR_LEVEL:
			int_tokens = self.map_list(tokens, add_start_end)
			tmp = [[self.start_index]]
			tmp.extend(int_tokens)
			tmp.append([self.end_index])
			return tmp
		elif self.vocab_type == constants.CHAR_LEVEL:
			int_tokens = self.map(tokens, add_start_end)
		elif self.vocab_type == constants.WORD_LEVEL:
			int_tokens = self.map(tokens, add_start_end)
		return int_tokens

	def tokens_to_string(self, tokens, delimiter=" "):
		if self.vocab_type == constants.WORD_CHAR_LEVEL:
			char_tokens = self.tokens_list(tokens)
			original_string = \
			delimiter.join(map(lambda x: ''.join(x), char_tokens))
		elif self.vocab_type == constants.CHAR_LEVEL:
			original_string = self.tokens(tokens)
		elif self.vocab_type == constants.WORD_LEVEL:
			original_string = self.tokens(tokens)
		return original_string

	def tokenize_text(self, text, delimiter=" ", use_tokenizer=False):
		""" Tokenizes text using a tokenizer """
		if self.vocab_type == constants.WORD_LEVEL or self.vocab_type == constants.WORD_CHAR_LEVEL:
			if use_tokenizer:
				tokens = tokenizer.split_sentence(text)
			else:
				tokens = text.split(delimiter)
		elif self.vocab_type == constants.CHAR_LEVEL:
			tokens = text
		else:
			tokens = text
		return tokens

	def init_from_array(self, array):
		""" Inits vocab from file_path. 
			file_path: Contains vocab tokens, one per line
		"""
		self.array = array
		self.init_structs()
		self.add_init_tokens(self.use_unk_token, self.add_start_end)
		for token in self.array:
			self.add_token(token)

	def init_from_counter(self, counter, min_count):
		""" Inits vocab from file_path. 
			file_path: Contains vocab tokens, one per line
		"""
		self.counter = counter
		self.min_count = min_count
		self.init_structs()
		self.add_init_tokens(self.use_unk_token, self.add_start_end)
		self.add_all_tokens(counter, min_count)

	def size(self):
		return self.cur_idx + 1

	def contains(self, token):
		has_token = token in self.token_to_idx
		return has_token

	def index(self, token):
		if self.contains(token):
			return self.token_to_idx[token]
		else:
			return self.unk_index

	def token(self, index):
		token = self.idx_to_token[index]
		return token

	def tokens(self, indices):
		tokens = []
		for i in range(0, len(indices)):
			cur_index = indices[i]
			cur_token = self.token(cur_index)
			tokens.append(cur_token)
		return tokens

	def map(self, tokens, add_start_end):
		indices = []
		if add_start_end:
			indices.append(self.start_index)

		for i in range(0, len(tokens)):
			cur_token = tokens[i]
			if self.contains(cur_token):
				cur_index = self.index(cur_token)
				indices.append(cur_index)
			elif self.use_unk_token:
				indices.append(self.unk_index)

		if add_start_end:
			indices.append(self.end_index)
		return indices

    # Maps over a list of list of tokens. Namely 
    # Could be sentences, or characters. I.e.
    # [[the cat ran], [the dog ran]] etc.
	def map_list(self, tokens_list, add_start_end):
		items = map(lambda tokens: self.map(tokens, add_start_end),\
		 tokens_list)
		return items

    # Converts a list of list of indices back to tokens. Namely 
    # Could be sentences, or characters. I.e.
    # [[1 2 3], [1, 4, 3]] -> [[the cat ran], [the dog ran]] etc.
	def tokens_list(self, indices_list):
		items = list(map(lambda indices: self.tokens(indices), indices_list))
		return items

	def add_all_tokens(self, counter, min_count):
		for token, count in counter.iteritems():
			if count >= min_count:
				self.add_token(token)
			else:
				print("Token count not over limit:") + token

	def add_init_tokens(self, use_unk_token, add_start_end):
		self.add_token(self.gradient_masking_token)

		if use_unk_token:
			self.unk_index = self.add_token(self.unk_token)

		if add_start_end:
			self.start_index = self.add_token(self.start_token)
			self.end_index = self.add_token(self.end_token)
			self.pad_index = self.add_token(self.pad_token)
			self.dummy_index = self.add_token(self.dummy_token)

	def init_structs(self):
		self.gradient_masking_token = "<GRADIENT_MASKING_TOKEN>" # Used for gradient masking
		self.unk_token = "<UNK>"
		self.start_token = "<S>"
		self.end_token = "</S>"
		self.pad_token = "<PAD>"
		self.dummy_token = "<DUMMY>"

		self.cur_idx = -1
		self.idx_to_token = dict()
		self.token_to_idx = dict()

	def add_token(self, token):
		if token in self.token_to_idx:
			return self.token_to_idx[token]
		else:
			self.cur_idx = self.cur_idx + 1
			self.token_to_idx[token] = self.cur_idx
			self.idx_to_token[self.cur_idx] = token 
			return self.cur_idx
