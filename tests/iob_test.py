from models.iob.iob_model import IOBModel
import numpy as np 

config = {
    'input_max_length': 20,
    'vocab_size': 10,
    'embeddings_size': 25,
    'hidden_size': 30,
    'out_size': 5,
    'num_classes': 3,
    'batch_size': 5,
    'learning_rate': 1e-2
}

model = IOBModel(config)
inputs = np.random.random_integers(0, config['vocab_size'] - 1, 
    size=[config['batch_size'], config['input_max_length']])
input_lengths = np.ones((config['batch_size']), dtype=np.int32) * 1
input_masks = np.ones((config['batch_size'], config['input_max_length']), dtype=np.int32)

for i in range(0, config['batch_size']):
    cur_input_length = input_lengths[i]
    input_masks[i][cur_input_length:] = 0

labels = np.random.random_integers(0, config['num_classes'] - 1, 
    size=[config['batch_size'], config['input_max_length']])

batch = { 'inputs': inputs,
'input_lengths': input_lengths, 
'input_masks': input_masks,
'labels': labels
}

for i in range(0, 100):
    loss, predictions = model.forward(batch)
    print(loss)
    print(predictions)
    print(labels)



