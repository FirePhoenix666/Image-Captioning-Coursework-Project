import numpy as np
import tensorflow as tf
from tqdm import tqdm

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embed_size, activation='relu')
        self.compresslayer = tf.keras.layers.GlobalAveragePooling1D() 

    def call(self, x):
        x = self.fc(x)
        x = self.compresslayer(x)
        return x #(32, 256)

#lstm cell: the first cell receives the image features as input and the second cell receives word as input and the hidden state from the first cell
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embed_size, units, vocab_size): #(256, 512, 7319)
        super(RNN_Decoder, self).__init__() 
        self.units = units 
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size) #(7319, 256) 
        self.lstm = tf.keras.layers.LSTMCell(self.units) #possible to make it multi layers here
        self.dense = tf.keras.layers.Dense(vocab_size) #zzh , activation='softmax'
        # self.dropout = tf.keras.layers.Dropout(0.5) #not in original paper

    def call(self, captions, features, state):
        embeddings = self.embed(captions)
        embeddings = tf.squeeze(embeddings, [1])      
        output, state = self.lstm(embeddings, state) #shape (batch_size, hidden_size) (32, 256) -> (32, 256)
        outputs = self.dense(output) #shape (batch_size, vocab_size) 
        return outputs, state #output = (batch_size, vocab_size): (32, 256) -> (32, 7319) 
       
    def reset_state(self, batch_size): 
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))] #tf.zeros((batch_size, self.hidden_size))