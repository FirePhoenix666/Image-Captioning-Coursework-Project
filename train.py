import json
import time
import os
from tqdm import tqdm
import numpy as np

import opts
from models import AoANet, BahdanauAttend, ShowTell, TopDown, TopDown_Refiner, TopDown_GRU

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.client import device_lib

# Read parameters
opt = opts.parse_opt()

# Detect available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load tokenizer
print('Load tokenizer...')
max_caption_words = opt.max_caption_words
vocab_size = opt.vocab_size
tokenizer_dir = opt.tokenizer_dir
with open(tokenizer_dir, 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# Prepare training data
print('Prepare training data...')
# Path to training dictionary
training_dict_dir = opt.dict_dir
training_dict = json.load(open(training_dict_dir))
# Path to training images
image_dir = opt.image_dir

# Extend a list of text indices to a given fixed length
def pad_text(text, max_length): 
    text = pad_sequences([text], maxlen=max_length, padding='post')[0]
    return (text)

# Prepare the format of training data, where X stores the image names, and y stores the captions after padding
def data_prep(data_dict, tokenizer, max_length, vocab_size):
    X, y = list(), list()
    # For each image and list of captions
    for image_name, captions in data_dict.items():
        image_name = image_dir + image_name + '.jpg'
        # For each caption in the list of captions
        for caption in captions:
            # Convert the caption words into a list of word indices
            word_idxs = tokenizer.texts_to_sequences([caption])[0]
            # Pad the input text to the same fixed length
            pad_idxs = pad_text(word_idxs, max_length)
            X.append(image_name)
            y.append(pad_idxs)
    return np.array(X), np.array(y)

# Load the numpy files
def map_func(img_name, cap):
   img_tensor = np.load(img_name.decode('utf-8')+'.npy')
   return img_tensor, cap

BATCH_SIZE = opt.batch_size
BUFFER_SIZE = opt.buffer_size
train_X, train_y = data_prep(training_dict, tokenizer, max_caption_words, vocab_size)
dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
# Use map to load the image features from numpy files
# item1 stands for image features, and item2 stands for corresponding captions
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Shuffle and batch dataset
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Load model
print('Load model...')
embedding_dim = opt.embedding_dim
units = opt.units
vocab_size = vocab_size
num_steps = len(train_X) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (attention_features_shape, features_shape)
features_shape = opt.features_shape
attention_features_shape = opt.attention_features_shape
# Use Adam as our optimizer
optimizer = tf.keras.optimizers.Adam(lr=opt.lr, epsilon=opt.epsilon)
# Use Sparse-Categorical-Crossentropy as our loss function of image captioning task
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

if opt.model == 'show_tell':
    # ShowTell model consists of an encoder and an decoder
    encoder = ShowTell.CNN_Encoder(embedding_dim)
    decoder = ShowTell.RNN_Decoder(embedding_dim, units, vocab_size)
    # Use Checkpoint to save the weights of models
    checkpoint_encoder = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
    checkpoint_decoder = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
elif opt.model == 'bahdanau_att':
    # BahdanauAttend model consists of an encoder and an decoder
    encoder = BahdanauAttend.CNN_Encoder(embedding_dim)
    decoder = BahdanauAttend.RNN_Decoder(embedding_dim, units, vocab_size)
    # Use Checkpoint to save the weights of models
    checkpoint_encoder = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
    checkpoint_decoder = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
elif opt.model == 'top_down':
    # TopDown model only include a single decoder
    hidden_size = 512
    decoder = TopDown.TopDownDecoder(embedding_dim, units, vocab_size, hidden_size)
    # Use Checkpoint to save the weights of models
    checkpoint_decoder = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
elif opt.model == 'att_on_att':
    # AoANet model consists of an encoder, an refiner and an decoder
    encoder = AoANet.CNN_Encoder(features_shape)
    refiner = AoANet.Refiner(features_shape, num_heads=2)
    decoder = AoANet.RNN_Decoder(embedding_dim, units, vocab_size, 2, features_shape)
    # Use Checkpoint to save the weights of models
    checkpoint_encoder = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
    checkpoint_refiner = tf.train.Checkpoint(optimizer=optimizer, model=refiner)
    checkpoint_decoder = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
elif opt.model == 'top_down_refiner':
    # TopDown_Refiner model consists of an encoder, an refiner and an decoder
    hidden_size = 512
    encoder = TopDown_Refiner.CNN_Encoder(features_shape)
    refiner = TopDown_Refiner.Refiner(features_shape, num_heads=2)
    decoder = TopDown_Refiner.TopDownDecoder(embedding_dim, units, vocab_size, hidden_size)
    # Use Checkpoint to save the weights of models
    checkpoint_encoder = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
    checkpoint_refiner = tf.train.Checkpoint(optimizer=optimizer, model=refiner)
    checkpoint_decoder = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
elif opt.model == 'top_down_gru':
    # TopDown_GRU model only include a single decoder
    hidden_size = 512
    decoder = TopDown_GRU.TopDownDecoder(embedding_dim, units, vocab_size, hidden_size)
    # Use Checkpoint to save the weights of models
    checkpoint_decoder = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

# Calculate the loss based on real captions and predicted captions
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # loss_object is Sparse-Categorical-Crossentropy function
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# Train step of ShowTell model
@tf.function
def train_step_showtell(img_tensor, target):
    loss = 0
    # Initialize the hidden state for each batch, since captions are not related among different images
    hidden = decoder.reset_state(batch_size=target.shape[0])
    # The first word is always 'startseq'
    dec_input = tf.expand_dims([tokenizer.word_index['startseq']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        # Apply encoder to image features
        features = encoder(img_tensor)
        # Image features first flow into the decoder
        predictions, hidden = decoder(target[:, 0], features, hidden, batch_size=target.shape[0], initial_state=True)
        # For each word in one caption
        # Note: the shape of target[:, i] is (batch_size, 1), where 1 stands for the former word
        for i in range(1, target.shape[1]):
            # Former word flow into decoder, outputing the next word
            predictions, hidden = decoder(dec_input, features, hidden, batch_size=target.shape[0], initial_state=None)
            # Calculate the loss
            loss += loss_function(target[:, i], predictions)
            # Use teacher forcing, so the input word will be the true word in the caption, instead of the predicted one
            dec_input = tf.expand_dims(target[:, i], 1)
    # Calculate the total loss
    total_loss = (loss / int(target.shape[1]))
    # Count the trainale variables
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    # Apply gradients
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    # Return the loss during training
    return loss, total_loss

# Train step of BahdanauAttend model
@tf.function
def train_step_bahdanau(img_tensor, target):
    loss = 0
    # Initialize the hidden state for each batch
    hidden = decoder.reset_state(batch_size=target.shape[0])
    # The first word is always 'startseq'
    dec_input = tf.expand_dims([tokenizer.word_index['startseq']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        # Apply encoder to image features
        features = encoder(img_tensor)
        # Note: the shape of target[:, i] is (batch_size, 1), where 1 stands for the former word
        for i in range(1, target.shape[1]):
            # Pass the features and the former word through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            # Use teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss

# Train step of TopDown model
@tf.function
def train_step_topdown(img_tensor, target):
    loss = 0
    # Initialize the hidden state for each batch
    state = decoder.reset_state(batch_size=target.shape[0])
    # Prepare the first word
    dec_input = [tokenizer.word_index['startseq']] * target.shape[0] # target[:,0]
    target = tf.cast(target, tf.float32)
    dec_input = tf.cast(dec_input, tf.float32)
    with tf.GradientTape() as tape:
        features = img_tensor
        # Note: the shape of target[:, i] is (batch_size, 1), where 1 stands for the former word
        for t in range(1, target.shape[1]):
            # Pass the features and the former word through the decoder
            state, scores = decoder(dec_input, features, state)
            loss += loss_function(target[:,t], scores)
            # Use teacher forcing
            dec_input = target[:,t]
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss

# Train step of AoANet model
@tf.function
def train_step_attonatt(img_tensor, target):
    loss = 0
    # Initialize the hidden state for each batch
    hidden = decoder.reset_state(batch_size=target.shape[0])
    # The first word is always 'startseq'
    dec_input = tf.expand_dims([tokenizer.word_index['startseq']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        # Apply encoder to image features
        features = encoder(img_tensor)
        # Apply refiner to image features
        features = refiner(features)
        # Prepare the inputs of decoder
        a_hat = tf.reduce_mean(features, 1)
        c = tf.random.uniform(tf.shape(a_hat), 0.0001, 0.001)
        # Note: the shape of target[:, i] is (batch_size, 1), where 1 stands for the former word
        for i in range(1, target.shape[1]):
            # Pass the features and the former word through the decoder
            predictions, hidden, c  = decoder(dec_input, features, a_hat, hidden, c)
            loss += loss_function(target[:, i], predictions)
            # Use teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + refiner.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss

# Train step of TopDown_Refiner model
@tf.function
def train_step_topdown_refiner(img_tensor, target):
    loss = 0
    # Initialize the hidden state for each batch
    state = decoder.reset_state(batch_size=target.shape[0])
    # The first word is always 'startseq'
    dec_input = [tokenizer.word_index['startseq']] * target.shape[0] # target[:,0]
    target = tf.cast(target, tf.float32)
    dec_input = tf.cast(dec_input, tf.float32)
    with tf.GradientTape() as tape:
        # Apply encoder to image features
        features = encoder(img_tensor)
        # Apply refiner to image features
        features = refiner(features)
        # Note: the shape of target[:, i] is (batch_size, 1), where 1 stands for the former word
        for t in range(1, target.shape[1]):
            # Pass the features and the former word through the decoder
            state, scores = decoder(dec_input, features, state)
            loss += loss_function(target[:,t], scores)
            # Use teacher forcing
            dec_input = target[:,t]
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + refiner.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss

# Train step of TopDown_GRU model
@tf.function
def train_step_topdown_gru(img_tensor, target):
    loss = 0
    # Initialize the hidden state for each batch
    state = decoder.reset_state(batch_size=target.shape[0])
    # The first word is always 'startseq'
    dec_input = [tokenizer.word_index['startseq']] * target.shape[0] # target[:,0]
    target = tf.cast(target, tf.float32)
    dec_input = tf.cast(dec_input, tf.float32)
    with tf.GradientTape() as tape:
        features = img_tensor
        # Note: the shape of target[:, i] is (batch_size, 1), where 1 stands for the former word
        for t in range(1, target.shape[1]):
            # Pass the features and the former word through the decoder
            state, scores = decoder(dec_input, features, state)
            loss += loss_function(target[:,t], scores)
            # Use teacher forcing
            dec_input = target[:,t]
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss


print('Start epochs...')
EPOCHS = opt.epochs
# For each rpoch
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0
    # For each batch
    for (batch, (img_tensor, target)) in enumerate(dataset):
        # Choose the train_step function based the model
        if opt.model == 'show_tell':
            batch_loss, t_loss = train_step_showtell(img_tensor, target)
        elif opt.model == 'bahdanau_att':
            batch_loss, t_loss = train_step_bahdanau(img_tensor, target)
        elif opt.model == 'top_down':
            batch_loss, t_loss = train_step_topdown(img_tensor, target)
        elif opt.model == 'att_on_att':
            batch_loss, t_loss = train_step_attonatt(img_tensor, target)
        elif opt.model == 'top_down_refiner':
             batch_loss, t_loss = train_step_topdown_refiner(img_tensor, target)
        elif opt.model == 'top_down_gru':
            batch_loss, t_loss = train_step_topdown_gru(img_tensor, target)
        # Calculate the total loss
        total_loss += t_loss
        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')

    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')


# Save the checkpoints of models
print('Saving models...')
def make_dir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
if opt.model == 'show_tell':
    prefix = opt.checkpoint_prefix_dir + 'ShowTell/'
    make_dir(prefix + 'encoder/')
    make_dir(prefix + 'decoder/')
    checkpoint_encoder.save(file_prefix = prefix + 'encoder/model.ckpt')
    checkpoint_decoder.save(file_prefix = prefix + 'decoder/model.ckpt')
elif opt.model == 'bahdanau_att':
    prefix = opt.checkpoint_prefix_dir + 'BahdanauAttend/'
    make_dir(prefix + 'encoder/')
    make_dir(prefix + 'decoder/')
    checkpoint_encoder.save(file_prefix = prefix + 'encoder/model.ckpt')
    checkpoint_decoder.save(file_prefix = prefix + 'decoder/model.ckpt')
elif opt.model == 'top_down':
    prefix = opt.checkpoint_prefix_dir + 'TopDown/'
    make_dir(prefix + 'decoder/')
    checkpoint_decoder.save(file_prefix = prefix + 'decoder/model.ckpt')
elif opt.model == 'att_on_att':
    prefix = opt.checkpoint_prefix_dir + 'AoANet/'
    make_dir(prefix + 'encoder/')
    make_dir(prefix + 'refiner/')
    make_dir(prefix + 'decoder/')
    checkpoint_encoder.save(file_prefix = prefix + 'encoder/model.ckpt')
    checkpoint_refiner.save(file_prefix = prefix + 'refiner/model.ckpt')
    checkpoint_decoder.save(file_prefix = prefix + 'decoder/model.ckpt')
elif opt.model == 'top_down_refiner':
    prefix = opt.checkpoint_prefix_dir + 'TopDown_Refiner/'
    make_dir(prefix + 'encoder/')
    make_dir(prefix + 'refiner/')
    make_dir(prefix + 'decoder/')
    checkpoint_encoder.save(file_prefix = prefix + 'encoder/model.ckpt')
    checkpoint_refiner.save(file_prefix = prefix + 'refiner/model.ckpt')
    checkpoint_decoder.save(file_prefix = prefix + 'decoder/model.ckpt')
elif opt.model == 'top_down_gru':
    prefix = opt.checkpoint_prefix_dir + 'TopDown_GRU/'
    make_dir(prefix + 'decoder/')
    checkpoint_decoder.save(file_prefix = prefix + 'decoder/model.ckpt')

print('Finished')