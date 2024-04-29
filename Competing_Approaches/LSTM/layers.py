import tensorflow as tf
# from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#import torch

#create RNN cell (LSTM/GRU)
def create_rnn_cell(cell_type, state_size, hidden_layers, default_initializer, dropout_rate, seed, reuse=None):
    if cell_type == 'GRU':
        return tf.compat.v1.nn.rnn_cell.GRUCell(state_size, activation=tf.nn.tanh, reuse=reuse)
    elif cell_type == 'LSTM':
        #return tf.compat.v1.nn.rnn_cell.LSTMCell(state_size, initializer=default_initializer, activation=tf.nn.tanh, reuse=reuse)

        rnn_layers = []
        for _ in range (hidden_layers):
            #rnn_layers.append(tf.compat.v1.nn.rnn_cell.LSTMCell(state_size))
            rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(state_size, initializer=default_initializer, activation=tf.nn.tanh, reuse=reuse)
            rnn_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1 - dropout_rate, seed=seed)
            rnn_layers.append(rnn_cell)

        # rnn_layers = []
        # for _ in range (hidden_layers):
        #     rnn_cell = tf.keras.layers.LSTMCell(units = state_size, activation = 'tanh')
        #     #rnn_cell = tf.keras.layers.Dropout(rate = dropout_rate, noise_shape=None, seed=seed)
        #     rnn_layers.append(rnn_cell)
        return rnn_layers
    else:
        return tf.compat.v1.nn.rnn_cell.BasicRNNCell(state_size, activation=tf.nn.tanh, reuse=reuse)

# creating RNN encoder using tensorflow
def create_rnn_encoder(x, rnn_units, hidden_layers, dropout_rate, seq_length, rnn_cell_type, param_initializer, seed, reuse=None):
    with tf.compat.v1.variable_scope("RNN_Encoder", reuse=reuse):
        # rnn_cell = create_rnn_cell(rnn_cell_type, rnn_units, hidden_layers, param_initializer, dropout_rate, seed)
        # rnn_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1 - dropout_rate, seed=seed)
        # init_state = rnn_cell.zero_state(tf.shape(x)[0], tf.float32)
        # # RNN Encoder: Iteratively compute output of recurrent network
        # rnn_outputs, _ = tf.compat.v1.nn.dynamic_rnn(rnn_cell, x, initial_state=init_state, sequence_length=seq_length, dtype=tf.float32)
        
        rnn_layers = create_rnn_cell(rnn_cell_type, rnn_units, hidden_layers, param_initializer, dropout_rate, seed)
        multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)
        multi_rnn_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(multi_rnn_cell, input_keep_prob=1 - dropout_rate, seed=seed)
        rnn_outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=x, sequence_length=seq_length,
                                   dtype=tf.float32)

        # rnn_layers = create_rnn_cell(rnn_cell_type, rnn_units, hidden_layers, param_initializer, dropout_rate, seed)
        # rnn_layer = tf.keras.layers.RNN(cell = rnn_layers,
        #                         return_sequences=False,
        #                         return_state=False)
        # #rnn_layer = tf.keras.layers.Dropout(rate = dropout_rate, noise_shape=None, seed=seed)
        # rnn_outputs, _ = rnn_layer(inputs = x)

        # rnn_layer = tf.keras.layers.LSTM(units = rnn_units, activation= 'tanh', return_sequences=False, return_state=False, dropout = dropout_rate)
        # rnn_outputs = rnn_layer(x)
        #rnn_layer = torch.nn.LSTM()

        return rnn_outputs

# tf.keras.layers.Dense
def create_basket_encoder(x, dense_units, param_initializer, activation_func=None, name="Basket_Encoder", reuse=None):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        #return tf.layers.dense(x, dense_units, kernel_initializer=param_initializer,
        return tf.compat.v1.layers.dense(x, dense_units, kernel_initializer=param_initializer,
                            bias_initializer=tf.zeros_initializer, activation=activation_func)
        #return tf.keras.layers.Dense(x, dense_units, kernel_initializer=param_initializer,
        #                    bias_initializer=tf.zeros_initializer, activation=activation_func)

def get_last_right_output(full_output, max_length, actual_length, rnn_units):
    batch_size = tf.compat.v1.shape(full_output)[0]
    # Start indices for each sample
    index = tf.compat.v1.range(0, batch_size) * max_length + (actual_length - 1)
    # Indexing
    return tf.compat.v1.gather(tf.reshape(full_output, [-1, rnn_units]), index)
