from data_loaders.iob_loader import IOBLoader
from models.iob.iob_model import IOBModel
from helpers import constants, utils
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

base_directory = 'datasets/squad_iob'

data_loader = IOBLoader(base_directory, tokenizer_type=constants.TOKENIZER_SPACE)
data_loader.mix_indices()

config = {
    'input_max_length': data_loader.input_max_length,
    'vocab_size': data_loader.vocab.size(),
    'embeddings_size': 300,
    'hidden_size': 150,
    'out_size': 100,
    'num_classes': data_loader.label_vocab.size(),
    'batch_size': 25,
    'learning_rate': 1e-2,
    'save_path': 'iob/logs'}

embeddings = utils.load_matrix('%s/word_embeddings.npy' % base_directory)
config_path = 'iob/logs/squad/config.json'
params_path = 'iob/logs/squad/model_params_%s.ckpt'

model = IOBModel(config, embeddings=embeddings)
model.save(config_path, params_path)
model.restore(params_path)

batch = data_loader.get_batch(constants.DATASET_TRAIN, config['batch_size'])

num_steps = 0

for i in range(0, 100):
    while batch is not None:
      loss, predictions = model.forward(batch)  
      batch = data_loader.get_batch(constants.DATASET_TRAIN, config['batch_size'])
      num_steps += config['batch_size']

      print(num_steps)
      print(loss)

    if i % 3 == 0:
      model.save(config_path, params_path % i)
      data_loader.reset_indices()
      total_predictions = []
      while True:
        batch = data_loader.get_batch(constants.DATASET_TEST, config['batch_size'])
        if batch is None:
          break
        predictions = model.predict(batch)
        texts = data_loader.label_vocab.tokens_list(predictions)
        for i in range(0, len(texts)):
          cur_input_length = batch['input_lengths'][i]
          cur_text = texts[i]
          text_str = " ".join(cur_text[0:cur_input_length])
          total_predictions.append(text_str)
      utils.save_lines(total_predictions, \
        '%s/predictions_test_%s.txt' % (config['save_path'], i))

    data_loader.mix_indices()
    batch = data_loader.get_batch(constants.DATASET_TRAIN, config['batch_size'])
  





