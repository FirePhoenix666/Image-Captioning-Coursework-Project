import tensorflow as tf
import time

# Attention modular
class Attend(tf.keras.layers.Layer):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.h2att = tf.keras.layers.Dense(self.hidden_size)
        self.v2att = tf.keras.layers.Dense(self.hidden_size)
        self.alpha_net = tf.keras.layers.Dense(1)

    def call(self, features, hidden, mask = None):
        hidden = tf.expand_dims(hidden, 1)
        att = tf.nn.tanh(self.v2att(features) + self.h2att(hidden))
        alpha = self.alpha_net(att)
        weights = tf.nn.softmax(alpha, axis = 1)
        attended_feature = weights * features
        attended_feature = tf.reduce_sum(attended_feature, axis = 1)
        return attended_feature, weights


class TopDownCore(tf.keras.layers.Layer):
    def __init__(self, units, hidden_size, **kwargs):
        self.units = units
        self.topdown_lstm = tf.keras.layers.LSTMCell(units)
        self.language_lstm = tf.keras.layers.LSTMCell(units)
        self.attend = Attend(hidden_size)
        super(TopDownCore, self).__init__(**kwargs)

    def call(self, xt, avg_features, features, state):
        # xt is the embedded words of step t (=step i)
        # avg_features is the mean pooling of features
        prev_h = state[0][-1]
        # Concatenate state[0][-1] and avg_features
        topdown_input  = tf.concat([prev_h, avg_features, xt], -1)
        # TopDown LSTMCell Layer
        _, (h_topdown, c_topdown) = self.topdown_lstm(topdown_input, (state[0][0], state[1][0]))
        # Attention Layer
        attended, _ = self.attend(features, h_topdown)
        # Concatenate the outputs of above two layers
        language_input = tf.concat([attended, h_topdown], -1)
        # Language LSTMCell Layer
        _, (h_language, c_language) = self.language_lstm(language_input, (state[0][1], state[1][1]))
        # Update the state
        state = [[h_topdown, h_language], [c_topdown, c_language]]
        return h_language, state

# Caption decoder of TopDown model
class TopDownDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, hidden_size, dropout_rate = 0.1):
        super(TopDownDecoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.core = TopDownCore(units, hidden_size)
        self.logit = tf.keras.layers.Dense(vocab_size)

    def call(self, caption, features, state):
        # Embedding Layer to expand the dimension of words
        xt = self.embedding(caption)
        # Calculate the mean of features
        avg_features = tf.reduce_mean(features, axis = 1)
        # Jump to TopDownCore class
        h_language, state = self.core(xt, avg_features, features,state)
        # Dense Layer to set the output dimension to be the vocabulary size
        scores = self.logit(h_language)
        return state, scores
    
    def reset_state(self, batch_size):
        # Reset the hidden state in the LSTMCell
        return [[tf.zeros((batch_size, 512)), tf.zeros((batch_size, 512))], 
                [tf.zeros((batch_size, 512)), tf.zeros((batch_size, 512))]]