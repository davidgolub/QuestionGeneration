import os
import os.path
import urllib
import mimetypes
import time
import random
import numpy
import collections
from helpers import constants
import json
from tqdm import tqdm
from collections import Counter

stopwords_list = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

verb_list = ["die", "write", "born", "does"]

class ConfigWrapper(object):
	def __init__(self):
		self.dict = {}

	def __setitem__(self, key, val):
		self.dict[key] = val

	def __getitem__(self, key):
		return self.dict[key]

def get_indices(tokens_arr, val):
	"""
	Function to get all indices in tokens_arr 
	where tokens_arr[idx] = val
	"""

	indices = [i for i, j in enumerate(tokens_arr) if j == val]
	return indices



def transpose_join(arr, delimiter=" "):
	t_arr = numpy.array(arr).T 
	lines = list(map(lambda r: delimiter.join(r), t_arr))
	return lines

def as_class(dic):
	new_dict = ConfigWrapper()
	for k, v in dic.iteritems():
		new_dict.__dict__[k] = v 
		new_dict[k] = v 
	return new_dict

def check_file(path):
	return os.path.isfile(path)

def count_lines(f):
	num_items = 0
	for line in open(f):
		num_items = num_items + 1
	return num_items

def create_mask(lengths, max_length):
	batch_size = numpy.shape(lengths)[0]
	mask_matrix = numpy.zeros((batch_size, max_length))
	for i in range(0, len(lengths)):
		mask_matrix[i, 0:lengths[i]] = 1
	return mask_matrix

def sparse_to_dense(indices, vocabulary_size):
	"""
	Converts a sparse array of one hot vectors to an array 
	where array[indices[i]] = 1, zeros everywhere else
	"""

	num_items = len(indices)
	dense_array = np.zeros((num_items, vocabulary_size))

	for i in range(0, num_items):
		cur_index = indices[i]
		dense_array[i][cur_index] = 1

	return dense_array

def save_matrix(matrix, path):
	""" Saves matrix to path """
	arr = numpy.array(matrix)
	numpy.save(path, matrix)

def load_matrix(path, path_type=constants.PATH_NPY_ARRAY):
	numpy_data = None
	if path_type == constants.PATH_NPY_ARRAY:
		numpy_data = numpy.load(path)
	elif path_type == constants.PATH_TEXT_ARRAY:
		numpy_data = numpy.loadtxt(path)
	else:
		raise Exception("Invalid path type given %s" % path_type)
	matrix = numpy.array(numpy_data)
	return matrix

def change_json_key(key, value, path):
	json_data = load_json(path)
	json_data[key] = value
	save_json(json_data, path)

def save_json(data, path):
	with open(path, 'w') as fp:
		json.dump(data, fp, indent=4)

def load_jsonl(path):
	data = jsonl.load(open(path, 'r'))
	return data 

def load_json(path):
	data = json.load(open(path, 'r'))
	return data

def clean_token(token, stopwords_list):
	if token in stopwords_list or token == "s" or token in verb_list:
		return ""
	else:
		return token

def remove_stopwords(sentence):
	sentence = sentence.replace("?", "")
	items = sentence.split(' ')
	cleaned_string = map(lambda item:clean_token(item, stopwords_list), items)
	string_w_spaces = ' '.join(cleaned_string)
	raw_tokens = string_w_spaces.split()
	single_space_string = ' '.join(raw_tokens)

	if len(raw_tokens) < 1:
		return sentence
	else:
		return single_space_string

def sample_negatives(np_list, num_samples, item_to_avoid, supplementary_list=None):
	""" Samples num_samples negatives from np_list. Where each
	sample is not equal to item_to_avoid 
	np_list: list to sample from 
	num_samples: num_samples to sample 
	item_to_avoid: item to avoid 
	supplementary_list: supplementary_list
	"""

	num_items = len(np_list)
	sample_list = []
	supplementary_sample_list = []
	while len(sample_list) < num_samples:
		new_sample_index = numpy.random.randint(num_items)
		new_item = np_list[new_sample_index]
		if not numpy.array_equal(new_item, item_to_avoid):
			sample_list.append(new_item)
			if supplementary_sample_list is not None:
				new_supplementary_item = supplementary_list[new_sample_index]
				supplementary_sample_list.append(new_supplementary_item)

	if supplementary_sample_list is not None:
		return sample_list, supplementary_sample_list
	else:
		return sample_list

def read_lines_into_counter(path):
	""" Reads lines from path into counter object.
		path: \n separated list of tokens.
	"""
	lines = read_lines(path)
	counter = Counter()

	map(lambda line:\
		update_counter(counter, line, constants.WORD_LEVEL),\
		lines)
	return counter
	

def update_counter(counter, token, token_type):
	""" Updates counter object to increase count for token
	For specified token type. Token is an array of characters.
	CHAR_LEVEL == array of chars. WORD_LEVEL == single word.
	"""
	if token_type == constants.CHAR_LEVEL:
		counter.update(token)
	elif token_type == constants.WORD_LEVEL:
		counter.update([token])
	else:
		raise Exception("Invalid token type given " + str(token_type))

def unique_vals(arr, min_num_occurences=1):
	""" 
	Gets unique values from list 
	arr: arr that contains value
	min_num_occurences: these values must occur at least X times
	"""
	counter_vals = Counter(arr)
	unique_words = [w for w, c in counter_vals.items() if c >= min_num_occurences]
	return unique_words

def create_dictionary(words):
	"""Creates word_to_idx and idx_to_word dictionary for labels
	words: list of words to use for dictionary
	"""
	word_to_idx = {}
	idx_to_word = {}
	cur_idx = 0
	for idx, word in enumerate(words):
		if not word in word_to_idx:
		 	word_to_idx[word] = cur_idx
		 	idx_to_word[cur_idx] = word
		 	cur_idx = cur_idx + 1

	return word_to_idx, idx_to_word

def remove_non_ascii(text):
	return ''.join(i for i in text if ord(i)<128)

def unicode_encode(text):
	try:
		text = unicode(text, 'utf-8')
		return text
	except TypeError:
		print("Error encoding %s" % text)
		return text

def resize_array(array, insert_index):
	""" Resizes array to contain insert index num elements 
		if insert index is out of bounds
		array: array to resize 
		insert_index: index we want to insert into 
	"""

	if len(array) <= insert_index:
		while insert_index >= len(array):
			array.append("NEED_TO_INSERT")

def read_lines_with_func(func, path):
	""" Reads lines with specified function from specified 
	path with function func to apply on each line """

	print("Reading lines with func from path %s" % path)
	mapped_content = []
	num_lines = count_lines(path)
	with open(path) as lines:
		for line in tqdm(lines, total=num_lines):
			mapped_content.append(func(line))
	return mapped_content

def read_lines(path, decode=True):
	""" Reads lines from path and does preprocessing """
	print("Readling lines from path %s" % path)
	def trim_func(s):
		if decode:
			return s.replace("\n", "")
		else:
			return s.replace("\n", "")
	with open(path) as f:
	    content = f.readlines()
	content = list(map(lambda s: trim_func(s), content))
	return content

def check_dir(save_dir):
	""" Creates dir if not exists"""
	if not os.path.exists(save_dir):
		print("Directory %s does not exist, making it now" % save_dir)
		os.makedirs(save_dir)
		return False
	else:
		print("Directory %s exists, all good" % save_dir)
		return True

def save_lines(array, path, encode=True):
	""" Writes lines to path from array 
		Assumes dir exists already
	"""
	print("Saving lines into path %s" % path)
	# Make sure dir exists
	check_dir(os.path.dirname(path)) ## directory of file)

	# Open a file in witre mode
	with open(path, "w") as fo:
		for i in range(0, len(array)):
			cur_el = array[i]
			fo.write(cur_el)

			# Make sure to not write newline in end of arr
			if i != len(array) - 1:
				fo.write("\n")
	return True

def save_tabbed_lines(array, path, encode=True):
	""" Writes lines from array to path 
		array: assumes to be an array of arrays
		path: path to write to
	"""
	check_dir(os.path.dirname(path))

	concatenated_array= list(map(lambda s: "\t".join(s), array))

	# Open a file in write mode
	with open(path, "w") as fo:
		success = save_lines(concatenated_array, path, encode=encode)

def read_tabbed_lines(path, delimiter='\t', append=False):
	""" Reads tabbed lines separated by delimiter from path 
		path: path to read from
		delimiter: delimiter to split by
		append: flag, either appends all items to array has an array of flags
	"""
	with open(path) as f:
	    content = f.readlines()
	content = map(lambda s: s.replace("\n", ""), content)

	final_content = []
	if append:
		list(map(lambda s: final_content.extend(s.split(delimiter)), content))
	else:
		final_content = list(map(lambda s: s.split(delimiter), content))

	return final_content

def split_array_with_labels(array, train_val_test_ratio):
	""" Splits array so that each class in the array is properly partitioned with the ratio
		array: array to split
		train_val_test_ratio: ratio to split by
	"""

	# First create new list where each index is grouped 
	idx_to_classes = {}


	values = set(array)
	sorted_values = [[idx for idx, y in enumerate(array) if y==x] for x in values]
	train_idx = []
	val_idx = []
	test_idx = []

	for value in sorted_values:
		# Need to convert to numpy array so we can do indexing
		nped_array = numpy.array(value)
		cur_train_idx, cur_val_idx, cur_test_idx = split_array(value, train_val_test_ratio)

		cur_train_vals = nped_array[cur_train_idx].tolist()
		cur_val_vals = nped_array[cur_val_idx].tolist()
		cur_test_vals = nped_array[cur_test_idx].tolist()

		train_idx.extend(cur_train_vals)
		val_idx.extend(cur_val_vals)
		test_idx.extend(cur_test_vals)

	return train_idx, val_idx, test_idx

def split_array(array, train_val_test_ratio):
	""" Splits array and returns indices
		array: array to split
		train_val_test_ratio: ratio to split (train to val + test (val/test are split evenly))
	"""
	size = len(array)
	indices = numpy.random.permutation(len(array))

	# Calculate partition index
	partition_idx = int(train_val_test_ratio * size)
	val_end_idx = int(partition_idx + size * (1 - train_val_test_ratio) * 0.5)

	print("Current partition index %d" % partition_idx)
	# Get train, val and test indices
	train_idx, val_idx, test_idx = indices[:partition_idx], indices[partition_idx:val_end_idx], indices[val_end_idx:]
	
	return train_idx, val_idx, test_idx

def convert_dict_to_array(d):
	""" Converts a dictionary into an array inserting items in increasing order by key
		d: Dictionary to convert to array 
		Returns: array
	"""
	arr = []
	od = collections.OrderedDict(sorted(d.items()))
	for k, v in od.iteritems():
		print(k)
		arr.append(v)
	return arr

def aggregate_files(base_dir, save_dir):
	""" Aggregates all lines from files 
		and saves it into a new file of the same name 
	"""
	# Keep track of file name to values
	idx_to_file = {}
	sub_dirs = [x[0] for x in os.walk(base_dir)]
	for sub_dir in sub_dirs:
		if sub_dir == base_dir: continue
		for cur_file in os.listdir(sub_dir):
			abs_file_path = os.path.join(sub_dir, cur_file)
			print("Aggregating file sub_dir %s " % abs_file_path)
			if cur_file not in idx_to_file:
				idx_to_file[cur_file] = []
			
			lines = read_lines(abs_file_path)
			idx_to_file[cur_file].extend(lines)

	# Check if directory exists
	check_dir(save_dir)

	# Save directory
	for key, value in idx_to_file.iteritems():
		save_path = os.path.join(save_dir, key)
		save_lines(value, save_path)