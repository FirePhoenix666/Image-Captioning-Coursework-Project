import os
import json
import csv
import time
from tqdm import tqdm
from models import AoANet, BahdanauAttend, ShowTell, TopDown, TopDown_Refiner, TopDown_GRU
import opts
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate import bleu_score, meteor_score

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

# Prepare test data
print('Prepare test data...')
# Path to test dictionary
image_dict_dir = opt.dict_dir
test_dict = json.load(open(image_dict_dir))
# Path to training images
image_dir = opt.image_dir

# Extend a list of text indices to a given fixed length
def pad_text(text, max_length): 
    text = pad_sequences([text], maxlen=max_length, padding='post')[0]
    return (text)

# Prepare the format of test data, where X stores the image names, and y stores the captions after padding
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
test_X, test_y = data_prep(test_dict, tokenizer, max_caption_words, vocab_size)
dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
# Use map to load the image features from numpy files
# item1 stands for image features, and item2 stands for corresponding captions
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Shuffle and batch
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# Load model
print('Load model...')
embedding_dim = opt.embedding_dim
units = opt.units
features_shape = opt.features_shape 
attention_features_shape =  opt.attention_features_shape

if opt.model == 'show_tell':
    # ShowTell model consists of an encoder and an decoder
    encoder = ShowTell.CNN_Encoder(embedding_dim)
    decoder = ShowTell.RNN_Decoder(embedding_dim, units, vocab_size)
    # Use Checkpoint to restore the weights of models
    checkpoint_encoder = tf.train.Checkpoint(model=encoder)
    checkpoint_decoder = tf.train.Checkpoint(model=decoder)
    prefix = opt.checkpoint_prefix_dir + 'ShowTell/'
    checkpoint_encoder.restore(tf.train.latest_checkpoint( prefix +'encoder'))
    checkpoint_decoder.restore(tf.train.latest_checkpoint( prefix +'decoder'))
elif opt.model == 'bahdanau_att':
    # BahdanauAttend model consists of an encoder and an decoder
    encoder = BahdanauAttend.CNN_Encoder(embedding_dim)
    decoder = BahdanauAttend.RNN_Decoder(embedding_dim, units, vocab_size)
    # Use Checkpoint to restore the weights of models
    checkpoint_encoder = tf.train.Checkpoint(model=encoder)
    checkpoint_decoder = tf.train.Checkpoint(model=decoder)
    prefix = opt.checkpoint_prefix_dir + 'BahdanauAttend/'
    checkpoint_encoder.restore(tf.train.latest_checkpoint( prefix +'encoder'))
    checkpoint_decoder.restore(tf.train.latest_checkpoint( prefix +'decoder'))
elif opt.model == 'top_down':
    # TopDown model only include a single decoder
    hidden_size = 512
    decoder = TopDown.TopDownDecoder(embedding_dim, units, vocab_size, hidden_size)
    # Use Checkpoint to restore the weights of models
    checkpoint_decoder = tf.train.Checkpoint(model=decoder)
    prefix = opt.checkpoint_prefix_dir + 'TopDown/'
    checkpoint_decoder.restore(tf.train.latest_checkpoint( prefix +'decoder'))
elif opt.model == 'att_on_att':
    # AoANet model consists of an encoder, an refiner and an decoder
    encoder = AoANet.CNN_Encoder(features_shape)
    refiner = AoANet.Refiner(features_shape, num_heads=2)
    decoder = AoANet.RNN_Decoder(embedding_dim, units, vocab_size, 2, features_shape)
    # Use Checkpoint to restore the weights of models
    checkpoint_encoder = tf.train.Checkpoint(model=encoder)
    checkpoint_refiner = tf.train.Checkpoint(model=refiner)
    checkpoint_decoder = tf.train.Checkpoint(model=decoder)
    prefix = opt.checkpoint_prefix_dir + 'AoANet/'
    checkpoint_encoder.restore(tf.train.latest_checkpoint( prefix +'encoder'))
    checkpoint_refiner.restore(tf.train.latest_checkpoint( prefix +'refiner'))
    checkpoint_decoder.restore(tf.train.latest_checkpoint( prefix +'decoder'))
elif opt.model == 'top_down_refiner':
    # TopDownRefiner model consists of an encoder, an refiner and an decoder
    hidden_size = 512
    encoder = TopDown_Refiner.CNN_Encoder(features_shape)
    refiner = TopDown_Refiner.Refiner(features_shape, num_heads=2)
    decoder = TopDown_Refiner.TopDownDecoder(embedding_dim, units, vocab_size, hidden_size)
    # Use Checkpoint to restore the weights of models
    checkpoint_encoder = tf.train.Checkpoint(model=encoder)
    checkpoint_refiner = tf.train.Checkpoint(model=refiner)
    checkpoint_decoder = tf.train.Checkpoint(model=decoder)
    prefix = opt.checkpoint_prefix_dir + 'TopDown_Refiner/'
    checkpoint_encoder.restore(tf.train.latest_checkpoint( prefix +'encoder'))
    checkpoint_refiner.restore(tf.train.latest_checkpoint( prefix +'refiner'))
    checkpoint_decoder.restore(tf.train.latest_checkpoint( prefix +'decoder'))
elif opt.model == 'top_down_gru':
    # TopDown_GRU model only include a single decoder
    hidden_size = 512
    decoder = TopDown_GRU.TopDownDecoder(embedding_dim, units, vocab_size, hidden_size)
    # Use Checkpoint to restore the weights of models
    checkpoint_decoder = tf.train.Checkpoint(model=decoder)
    prefix = opt.checkpoint_prefix_dir + 'TopDown_GRU/'
    checkpoint_decoder.restore(tf.train.latest_checkpoint( prefix +'decoder'))

# Evaluate step of ShowTell model
def evaluate_step_showtell(image, batch_size, max_length):
    # Initialize the hidden state for each batch, since captions are not related among different images
    hidden = decoder.reset_state(batch_size=batch_size)
    # The first word is always 'startseq'
    dec_input = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    # Apply encoder to image features
    features = encoder(image)
    result = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    result = tf.cast(result, dtype=tf.int64)
    # For each word in one caption
    # Note: the shape of dec_input is (batch_size, 1), where 1 stands for the former word
    for i in range(max_length):
        # Former word flow into decoder, outputing the next word
        predictions, hidden = decoder(dec_input, features, hidden, batch_size=opt.batch_size, initial_state=None)
        # Store predicted_id into result list
        predicted_id = tf.random.categorical(predictions, 1)
        result = tf.concat([result, predicted_id], axis=1)
        dec_input = predicted_id
    return result

# Evaluate step of BahdanauAttend model
def evaluate_step_bahdanau(image, batch_size, max_length):
    # Initialize the hidden state for each batch
    hidden = decoder.reset_state(batch_size=batch_size)
    # The first word is always 'startseq'
    dec_input = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    # Apply encoder to image features
    features = encoder(image)
    result = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    result = tf.cast(result, dtype=tf.int64)
    # For each word in one caption
    # Note: the shape of dec_input is (batch_size, 1), where 1 stands for the former word
    for i in range(max_length):
        # Former word flow into decoder, outputing the next word
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        # Store predicted_id into result list
        predicted_id = tf.random.categorical(predictions, 1)
        result = tf.concat([result, predicted_id], axis=1)
        dec_input = predicted_id
    return result

# Train step of TopDown model
def evaluate_step_topdown(image, batch_size, max_length):
    # Initialize the hidden state for each batch
    hidden = decoder.reset_state(batch_size=batch_size)
    # The first word is always 'startseq'
    dec_input = [tokenizer.word_index['startseq']] * batch_size
    dec_input = tf.cast(dec_input, tf.int64)
    features = image
    result = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    result = tf.cast(result, dtype=tf.int64)
    # For each word in one caption
    # Note: the shape of dec_input is (batch_size, 1), where 1 stands for the former word
    for i in range(max_length):
        # Former word flow into decoder, outputing the next word
        hidden, predictions = decoder(dec_input, features, hidden)
        # Store predicted_id into result list
        predicted_id = tf.random.categorical(predictions, 1)
        result = tf.concat([result, predicted_id], axis=1)
        dec_input = tf.squeeze(predicted_id)
    return result

# Train step of AoANet model
def evaluate_step_attonatt(image, batch_size, max_length):
    # Initialize the hidden state for each batch
    hidden = decoder.reset_state(batch_size=batch_size)
    # The first word is always 'startseq'
    dec_input = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    # Apply encoder to image features
    features = encoder(image)
    # Apply refiner to CNN-preprocessed image features
    features = refiner(features)
    a_hat = tf.reduce_mean(features, 1)
    c = tf.random.uniform(tf.shape(a_hat), 0.0001, 0.001)
    result = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    result = tf.cast(result, dtype=tf.int64)
    # For each word in one caption
    # Note: the shape of dec_input is (batch_size, 1), where 1 stands for the former word
    for i in range(max_length):
        # Former word flow into decoder, outputing the next word
        predictions, hidden, c  = decoder(dec_input, features, a_hat, hidden, c)
        # Store predicted_id into result list
        predicted_id = tf.random.categorical(predictions, 1)
        result = tf.concat([result, predicted_id], axis=1)
        dec_input = predicted_id
    return result

# Train step of TopDown_Refiner model
def evaluate_step_topdown_refiner(image, batch_size, max_length):
    # Initialize the hidden state for each batch
    hidden = decoder.reset_state(batch_size=batch_size)
    # The first word is always 'startseq'
    dec_input = [tokenizer.word_index['startseq']] * batch_size 
    dec_input = tf.cast(dec_input, tf.int64)
    # Apply encoder to image features
    features = encoder(img_tensor)
    # Apply refiner to CNN-preprocessed image features
    features = refiner(features)
    result = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    result = tf.cast(result, dtype=tf.int64)
    # For each word in one caption
    # Note: the shape of dec_input is (batch_size, 1), where 1 stands for the former word
    for i in range(max_length):
        # Former word flow into decoder, outputing the next word
        hidden, predictions = decoder(dec_input, features, hidden)
        # Store predicted_id into result list
        predicted_id = tf.random.categorical(predictions, 1)
        result = tf.concat([result, predicted_id], axis=1)
        dec_input = tf.squeeze(predicted_id)
    return result

# Train step of TopDown_GRU model
def evaluate_step_topdown_gru(image, batch_size, max_length):
    # Initialize the hidden state for each batch
    hidden = decoder.reset_state(batch_size=batch_size)
    # The first word is always 'startseq'
    dec_input = [tokenizer.word_index['startseq']] * batch_size
    dec_input = tf.cast(dec_input, tf.int64)
    features = image
    result = tf.expand_dims([tokenizer.word_index['startseq']] * batch_size, 1)
    result = tf.cast(result, dtype=tf.int64)
    # For each word in one caption
    # Note: the shape of dec_input is (batch_size, 1), where 1 stands for the former word
    for i in range(max_length):
        # Former word flow into decoder, outputing the next word
        hidden, predictions = decoder(dec_input, features, hidden)
        # Store predicted_id into result list
        predicted_id = tf.random.categorical(predictions, 1)
        result = tf.concat([result, predicted_id], axis=1)
        dec_input = tf.squeeze(predicted_id)
    return result

print('Start evaluating...')
# Variables to store the evaluation results
length = 0
bleu_ave = []
bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []
meteor = []
# Save results into .csv files
save_dir = opt.log_dir 
with open(save_dir, 'a', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['index','target', 'prediction', 'bleu_ave', 'bleu_1','bleu_2','bleu_3','bleu_4','meteor'])
    for (batch, (img_tensor, target)) in enumerate(dataset):
        # Choose the evaluate_step function based the model
        if opt.model == 'show_tell':
            predictions_id = evaluate_step_showtell(img_tensor, target.shape[0], max_caption_words)
        elif opt.model == 'bahdanau_att':
            predictions_id = evaluate_step_bahdanau(img_tensor, target.shape[0], max_caption_words)
        elif opt.model == 'top_down':
            predictions_id = evaluate_step_topdown(img_tensor, target.shape[0], max_caption_words)
        elif opt.model == 'att_on_att':
            predictions_id = evaluate_step_attonatt(img_tensor, target.shape[0], max_caption_words)
        elif opt.model == 'top_down_refiner':
            predictions_id = evaluate_step_topdown_refiner(img_tensor, target.shape[0], max_caption_words)
        elif opt.model == 'top_down_gru':
            predictions_id = evaluate_step_topdown_gru(img_tensor, target.shape[0], max_caption_words)
        # Prepare the true captions and the predicted captions
        predictions_id = predictions_id.numpy()
        target = target.numpy()
        # For each caption in the current batch
        for i in range(target.shape[0]):
            # Get the predicted captions from word indexes
            prediction_split = []
            for index in predictions_id[i,:]:
                prediction_split.append(tokenizer.index_word[index])
                if prediction_split[-1] == 'endseq':  break
            prediction_split = prediction_split[1:-1]
            #  Get the true captions from word indexes
            target_split = []
            for index in target[i,:]:
                target_split.append(tokenizer.index_word[index])
                if target_split[-1] == 'endseq':  break
            target_split = target_split[1:-1]
            # Evaluation metrics include BLEU and METEOR, using nltk package
            chencherry =  bleu_score.SmoothingFunction()
            length += 1
            bleu_ave.append(bleu_score.sentence_bleu([prediction_split], target_split, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function = chencherry.method7))
            bleu_1.append(bleu_score.sentence_bleu([prediction_split], target_split, weights=[1, 0, 0, 0], smoothing_function = chencherry.method7))
            bleu_2.append(bleu_score.sentence_bleu([prediction_split], target_split, weights=[0, 1, 0, 0], smoothing_function = chencherry.method7))
            bleu_3.append(bleu_score.sentence_bleu([prediction_split], target_split, weights=[0, 0, 1, 0], smoothing_function = chencherry.method7))
            bleu_4.append(bleu_score.sentence_bleu([prediction_split], target_split, weights=[0, 0, 0, 1], smoothing_function = chencherry.method7))
            meteor.append(meteor_score.single_meteor_score(set(prediction_split), set(target_split)))
            csv_writer.writerow([length, ' '.join(target_split), ' '.join(prediction_split), bleu_ave[-1], bleu_1[-1], bleu_2[-1], bleu_3[-1], bleu_4[-1], meteor[-1]])
    # Take the average of all above results
    csv_writer.writerow([length+1, ' ', ' ', sum(bleu_ave)/len(bleu_ave), sum(bleu_1)/len(bleu_1), sum(bleu_2)/len(bleu_2), sum(bleu_3)/len(bleu_3), sum(bleu_4)/len(bleu_4), sum(meteor)/len(meteor)])
print('Finished')