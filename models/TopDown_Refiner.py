import tensorflow as tf
import time

# This model consists of AoANet model without decoder and TopDown model

# We have already extracted and saved the image features using InceptionV3 model
# CNN_Encoder passes these features through a Fully connected layer
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        # Dense layer turn the shape of image features into (embedding_dim, 1)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# Attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# Multi-head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth).
        # Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


# Attention on Attention
class AoALayer(tf.keras.Model):
    def __init__(self, d_model, num_heads):
        super(AoALayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.multihead = MultiHeadAttention(self.d_model, self.num_heads)
        self.concat = tf.keras.layers.Concatenate()
        self.linear_i = tf.keras.layers.Dense(self.d_model, activation=None)
        self.linear_g = tf.keras.layers.Dense(self.d_model, activation='sigmoid')
    
    def call(self, q, k, v, mask):
        # Attention Layer to get v
        if self.num_heads > 1:
            v, att_weights = self.multihead(q, k, v, mask)
        else:
            v, att_weights = scaled_dot_product_attention(q, k, v, mask)
        # Concatenate q and v
        if v.shape[1] == 1:
            v = tf.squeeze(v, [1])
        x = self.concat([v, q])
        # Dense Layer with sigmoid function to get G
        i = self.linear_i(x)
        # Dense Layer without activation to get I
        g = self.linear_g(x)
        # Multiply I and G and return the result
        i_hat = tf.multiply(i,g)
        return i_hat


# Refining module in the image encoder
class Refiner(tf.keras.Model):
    def __init__(self, d_model, num_heads):
        super(Refiner, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.wq = tf.keras.layers.Dense(self.d_model, activation=None)
        self.wk = tf.keras.layers.Dense(self.d_model, activation=None)
        self.wv = tf.keras.layers.Dense(self.d_model, activation=None)
        self.aoa = AoALayer(self.d_model, self.num_heads) # include multihead attention
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, A):
        # Dense Layer to get Q
        q = self.wq(A)
        # Dense Layer to get K
        k = self.wk(A)
        # Dense Layer to get V
        v = self.wv(A)
        # AoA Layer with Multi-head Attention
        i_hat = self.aoa(q, k, v, mask=None)
        # Add A and the output of AoA layer
        A_hat = self.add([A, i_hat])
        # Normalization Layer
        A_hat = self.norm(A_hat)
        return A_hat


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