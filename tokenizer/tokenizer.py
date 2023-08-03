import json
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Prepare tokenizer
print('Prepare tokenizer...')

# Flat list of all captions
def all_captions(data_dict):
    return ([caption for key, captions in data_dict.items() for caption in captions])

# Calculate the word-length of the caption with the most words
def max_caption_length(captions):
    return max(len(caption.split()) for caption in captions)

# Extend a list of text indices to a given fixed length
def pad_text(text, max_length): 
    text = pad_sequences([text], maxlen=max_length, padding='post')[0]
    return (text)

# Fit a Keras tokenizer given caption descriptions
# The tokenizer uses the captions to learn a mapping from words to numeric word indices
def create_tokenizer(data_dict):
    captions = all_captions(data_dict)
    max_caption_words = max_caption_length(captions)    
    # Initialise a Keras Tokenizer
    tokenizer = Tokenizer()    
    # Fit it on the captions so that it prepares a vocabulary of all words
    tokenizer.fit_on_texts(captions)    
    # Get the size of the vocabulary
    vocab_size = len(tokenizer.word_index) + 1
    return (tokenizer, vocab_size, max_caption_words)

# Path to load image dictionary
image_dict_dir  = 'data/Flickr30k/image_dict.json' # 'data/Flickr8k/image_dict.json'
image_dict = json.load(open(image_dict_dir ))
# Fit a tokenizer
tokenizer, vocab_size, max_caption_words = create_tokenizer(image_dict)
# Print the size of vocabulary and the max length of all captions
print(vocab_size) # Flickr8k: 8425; Flickr30k: 18052
print(max_caption_words) #Flickr8k: 35; Flickr30k: 74
tokenizer_dict = json.dumps(tokenizer.to_json())
# Path to save the tokenizer
save_dir = 'tokenizer/Flickr30k/tokenizer.json' # 'tokenizer/Flickr8k/tokenizer.json'
with open(save_dir,'w') as f:
    f.write(tokenizer_dict)