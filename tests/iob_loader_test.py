from data_loaders.iob_loader import IOBLoader
from helpers import constants 

base_directory = 'datasets/iob_test'

tmp = IOBLoader(base_directory, tokenizer_type=constants.TOKENIZER_NLTK)
tmp.mix_indices()

batch = tmp.get_batch(constants.DATASET_TRAIN, 2)

print(batch)