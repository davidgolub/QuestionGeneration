from helpers import utils
from helpers import constants
from helpers.vocab import Vocab

import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--path", "--path", dest="path",
                  help="Path to save words from")
parser.add_option("--save_word_path", "--save_word_path", dest="save_word_path",
                  help="Where to save vocab")
parser.add_option("--save_embeddings_path", "--save_embeddings_path", dest="save_embeddings_path",
                  default = 'save_embeddings_path', help="Where to save embeddings to")
(options, args) = parser.parse_args()

path = options.path
save_word_path = options.save_word_path
save_embeddings_path = options.save_embeddings_path
save_embeddings_file = open(save_embeddings_path, 'w')

# Test loading it into vocab
vocab = Vocab(vocab_type=constants.WORD_LEVEL, add_start_end=True)
vocab.init_from_path(path=save_word_path)

token_to_embedding = dict()
num_items = []
original_item = []
original_embedding = []
cur_index = 0
def read_line(line):
    items = line.strip().split(' ')
    embed_size = len(items) - 1 # First item is word
    cur_index = len(num_items)

    word = items[0]
    embedding_vector = items[1:]
    if cur_index % 100 == 0:
        print(cur_index)
    if len(original_item) == 0:
        original_item.append(word)
        original_embedding.append(map(lambda vec: float(vec), embedding_vector))
    num_items.append(embed_size)

    if word in vocab.token_to_idx:
        token_to_embedding[word] = map(lambda vec: float(vec), embedding_vector)

print("Reading embeddings")
# Read in raw embeddings
utils.read_lines_with_func(func=read_line, path=path)

print("Done reading embeddings, now creating save matrix")
original_embedding_size = num_items[0]
word_embedding_matrix = []
num_items_saved = 0
for i in range(0, vocab.size()):
    if i % 100 == 0:
        print("On index %s from %s" % (i, vocab.size()))
    cur_token = vocab.token(i)
    embedding_vector = np.zeros(original_embedding_size)
    if cur_token in token_to_embedding:
        embedding_vector = token_to_embedding[cur_token]
        num_items_saved = num_items_saved + 1
    word_embedding_matrix.append(embedding_vector)

utils.save_matrix(matrix=word_embedding_matrix, path=save_embeddings_path)
vocab.init_embeddings(embeddings_path=save_embeddings_path, path_type=constants.PATH_NPY_ARRAY)

print("Saved %s of %s tokens" % (num_items_saved, vocab.size()))
print("Testing embeddings for original token %s" % original_item)
index = vocab.index(original_item[0])
embedding = vocab.embeddings[index]

print(index)
print(embedding)

tmp = np.array(original_embedding[0])
diff = embedding - tmp
print(diff)
