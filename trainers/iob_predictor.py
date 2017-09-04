from data_loaders.iob_loader import IOBLoader
from models.iob.iob_model import IOBModel
from helpers import constants, utils
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

embeddings = utils.load_matrix('datasets/squad_iob/word_embeddings.npy')
base_directory = 'datasets/newsqa_iob'
config_path = 'iob/logs/squad/config.json'
params_path = 'iob/logs/squad/model_params_3.ckpt'
predictions_save_path = 'iob/logs/newsqa/train_predictions_1.txt'

data_loader = IOBLoader(base_directory, tokenizer_type=constants.TOKENIZER_SPECIAL_DELIMITER,
    input_max_length=2100)#00)

config = utils.load_json(config_path)
config['batch_size'] = 25
config['input_max_length'] = data_loader.input_max_length
model = IOBModel(config, embeddings=embeddings)
model.restore(params_path)

num_steps = 0

data_loader.reset_indices()
total_predictions = []
num_steps = 0

while True:
    batch = data_loader.get_batch(constants.DATASET_TRAIN, config['batch_size'])
    num_steps += config['batch_size']
    print(num_steps)
    if batch is None:
      break
    predictions = model.predict(batch)
    texts = data_loader.label_vocab.tokens_list(predictions)
    for i in range(0, len(texts)):
      cur_input_length = batch['input_lengths'][i]
      cur_text = texts[i]

      text_str = " ".join(cur_text[0:cur_input_length])
      total_predictions.append(text_str)

utils.save_lines(total_predictions, predictions_save_path)