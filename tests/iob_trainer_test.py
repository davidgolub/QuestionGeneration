from data_loaders.iob_loader import IOBLoader
from models.iob.iob_model import IOBModel
from helpers import constants, utils

base_directory = 'datasets/iob_test'

data_loader = IOBLoader(base_directory, tokenizer_type=constants.TOKENIZER_SPACE)
data_loader.mix_indices()


config = {
    'input_max_length': data_loader.input_max_length,
    'vocab_size': data_loader.vocab.size(),
    'embeddings_size': 25,
    'hidden_size': 30,
    'out_size': 5,
    'num_classes': data_loader.label_vocab.size(),
    'batch_size': 3,
    'learning_rate': 1e-2,
    'save_path': 'iob/logs'}

config_path = 'iob/logs/config.json'
params_path = 'iob/logs/model_params.ckpt'

model = IOBModel(config, embeddings=None)
model.save(config_path, params_path)
model.restore(params_path)

batch = data_loader.get_batch(constants.DATASET_TRAIN, config['batch_size'])

for i in range(0, 100):
    while batch is not None:
      loss, predictions = model.forward(batch)  
      batch = data_loader.get_batch(constants.DATASET_TRAIN, config['batch_size'])
      print(predictions)
      print(loss)

    if i % 3 == 0:
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
  





