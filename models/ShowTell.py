import numpy as np
import tensorflow as tf
from tqdm import tqdm

# We have already extracted and saved the image features using InceptionV3 model
# CNN_Encoder passes these features through a Fully connected layer
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.compresslayer = tf.keras.layers.GlobalAveragePooling1D() 

    def call(self, x):
        # Dense layer turn the shape of image features into (embedding_dim, 1)
        x = self.fc(x)
        x = self.compresslayer(x)
        return x

# Caption decoder of ShowTell model
# RNN_Decoder use image features and the former word in the caption, to predic the next word.
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_size = vocab_size
        self.embed = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = tf.keras.layers.LSTMCell(self.units)
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, features, state , batch_size, initial_state=None):
        # The first input is the image features
        if initial_state:
            output, state = self.lstm(features, state)
        # Then input words in the captions sequentially
        else:
            # Embedding Layer to expand the dimension of words
            embeddings = self.embed(x)
            embeddings = tf.squeeze(embeddings, [1])
            # LSTMCell Layer to intake words in the captions one by one
            output, state = self.lstm(embeddings, state)
        # Dense Layer to set the output dimension to be the vocabulary size
        outputs = self.dense(output)
        return outputs, state
       
    def reset_state(self, batch_size): 
        # Reset the hidden state in the LSTMCell
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]