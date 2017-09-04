import tensorflow as tf 
from helpers import utils 

class IOBModel(object):
    def __init__(self, config, embeddings=None):
        self.config = config
        self.embeddings = embeddings
        self.create_placeholders()
        self.create_model()
        self.create_train_op()
        self.create_session()

    def create_session(self):
        self._sess = tf.Session()
        init = tf.initialize_all_variables()
        self._sess.run(init)
        self._saver = tf.train.Saver()

    def create_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
        self.train_op = optimizer.minimize(self.loss)


    def create_placeholders(self):
        # Question inputs
        self._input_placeholder = tf.placeholder(tf.int32, \
            [None, self.config['input_max_length']])
        self._input_lengths_placeholder = tf.placeholder(tf.int32, \
            [None])
        self._input_masks_placeholder = tf.placeholder(tf.float32, \
            [None, self.config['input_max_length']])
        self._labels_placeholder = tf.placeholder(tf.int32, \
            [None, self.config['input_max_length']])

    def create_model(self):
        ### YOUR CODE HERE (~4-6 lines)
        self.embeddings = self.create_embeddings()        
        self.lstm_embeddings, _ = self.create_lstm(self.embeddings, 
            self._input_lengths_placeholder)
        self.class_logits = self.create_fc(self.lstm_embeddings, \
            self.config['out_size'], self.config['num_classes'])
        self.predictions = tf.argmax(self.class_logits, 2)
        self.loss = self.create_loss(self.class_logits, \
            self._labels_placeholder, self._input_masks_placeholder)


    def create_loss(self, logits, labels, masks):
        reshaped_logits = tf.reshape(logits, [-1, self.config['num_classes']])
        reshaped_labels = tf.reshape(labels, [-1])
        reshaped_masks = tf.reshape(masks, [-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=reshaped_labels, logits=reshaped_logits) 
        masked_loss = tf.mul(loss, reshaped_masks)
        avg_loss = tf.reduce_mean(loss)
        return avg_loss


    def create_fc(self, lstm_embeddings, out_size, num_classes):
        input_shape = lstm_embeddings.get_shape()
        feed_in = tf.reshape(lstm_embeddings, [-1, input_shape[-1].value])
        dim = input_shape[-1].value

        with tf.variable_scope("w1"):
            weights = tf.get_variable('weights1', shape=[dim, out_size], trainable=True)
            biases = tf.get_variable('biases1', shape=[out_size],initializer=tf.constant_initializer(0.1), trainable=True)
            fc1 = tf.nn.xw_plus_b(feed_in, weights, biases)
            tanh1 = tf.nn.tanh(fc1)

        with tf.variable_scope("w2"):
            weights2 = tf.get_variable('weights1', shape=[out_size, num_classes], trainable=True)
            biases2 = tf.get_variable('biases1', shape=[num_classes],initializer=tf.constant_initializer(0.1), trainable=True)
            fc2 = tf.nn.xw_plus_b(tanh1, weights2, biases2)

        dim_0 = input_shape[0].value
        dim_1 = input_shape[1].value
        out = tf.reshape(fc2, [-1, input_shape[1].value, num_classes])
        return out

    def create_lstm(self, inputs, input_lengths):
        variable_scope ="LSTM_SCOPE"
        with tf.variable_scope(variable_scope):
            with tf.variable_scope('forward'):
                cell_fw = tf.nn.rnn_cell.LSTMCell(self.config['hidden_size'])

            with tf.variable_scope('backward'):
                cell_bw = tf.nn.rnn_cell.LSTMCell(self.config['hidden_size'])

            outputs_bi, states_bi = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs = inputs,
                sequence_length=input_lengths,
                dtype=tf.float32,
                scope=variable_scope)

            hidden_states = tf.concat(2, [outputs_bi[0], outputs_bi[1]])
            
            states = tf.concat(1, [states_bi[0], states_bi[1]])
        
            return hidden_states, states

    def create_embeddings(self):
        if self.embeddings != None:
            W = tf.get_variable(name="W", shape=[self.config['vocab_size'], self.config['embeddings_size']], \
                initializer=tf.constant_initializer(self.embeddings),
                trainable=True)
        else:
            W = tf.get_variable(name="W", shape=[self.config['vocab_size'], self.config['embeddings_size']], \
                trainable=True) 
        word_embeddings = tf.nn.embedding_lookup(W, self._input_placeholder)
        return word_embeddings 

    def forward(self, batch):
        inputs = batch['input_tokens']
        input_lengths = batch['input_lengths']
        input_masks = batch['input_masks']
        labels = batch['label_tokens']

        feed_dict = {
            self._input_placeholder: inputs,
            self._input_lengths_placeholder: input_lengths,
            self._input_masks_placeholder: input_masks,
            self._labels_placeholder: labels
        }
        _, loss, predictions = self._sess.run([self.train_op, self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def save(self, config_path, params_path):
        print("Saving model to config, param path %s %s" % (config_path, params_path))
        save_path = self._saver.save(self._sess, params_path)
        
        # Open a file for writing
        jsoned_config = self.config
        print(jsoned_config)
        utils.save_json(jsoned_config, config_path)

    def restore(self, params_path):
        print("Restoring model params from path %s")
        self._saver.restore(self._sess, params_path)

    def predict(self, batch):
        inputs = batch['input_tokens']
        input_lengths = batch['input_lengths']
        input_masks = batch['input_masks']

        feed_dict = {
            self._input_placeholder: inputs,
            self._input_lengths_placeholder: input_lengths,
            self._input_masks_placeholder: input_masks,
        }
        predictions = self._sess.run(self.predictions, feed_dict=feed_dict)
        return predictions


