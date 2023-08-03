import os
import re
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the file containing all of the captions into a single long string
def load_captions(filename):
    with open(filename, "r") as fp:
        # Read all text in the file
        text = fp.read()
        return text

# Create a dictionary of photo identifiers (without the .jpg) to captions
# {"image_name_1" : ["caption 1", "caption 2", "caption 3"],
#  "image_name_2" : ["caption 4", "caption 5"]}
def captions_dict(text):
    dict = {}
    # Make a List of each line in the file
    lines = text.split ('\n')
    for line in lines:
        # Split into the <image_data> and <caption>
        line_split = line.split ('\t')
        if (len(line_split) != 2):
            # Added this check because dataset contains some blank lines
            continue
        else:
            image_data, caption = line_split
        # Split into <image_file> and <caption_idx>
        image_file, caption_idx = image_data.split ('#')
        # Split the <image_file> into <image_name>.jpg
        image_name = image_file.split ('.')[0]
        # If this is the first caption for this image, create a new list and add the caption to it.
        # Otherwise append the caption to the existing list
        if int(caption_idx) == 0 :
            dict[image_name] = [caption]
        else:
            dict[image_name].append (caption)
    return dict

# There are two files which contain the names for images used for training and evaluating respectively
# Given a file, return the image names (without .jpg extension) in that file
def subset_image_name(filename):
    data = []
    with open(filename, "r") as fp:
        # Read all text in the file
        text = fp.read()
        # Make a List of each line in the file
        lines = text.split ('\n')
        for line in lines:
            # skip empty lines
            if (len(line) < 1):
                continue
            # Each line is the <image_file>
            # Split the <image_file> into <image_name>.jpg
            image_name = line.split ('.')[0]
            # Add the <image_name> to the list
            data.append (image_name)
    return (set(data))

# Path to images
image_dir = "Flickr/Flickr30k/flickr30k-images/"
# "Flickr/Flickr8k/Flicker8k_Dataset/"

# Path to the file containing the training image names
training_image_name_file = "Flickr/Flickr30k/Flickr_30k.trainImages.txt"
# "Flickr/Flickr8k/Flickr_8k.trainImages.txt"

# Path to save training image features
training_save_dir = "data/Flickr30k/image_features_train/"
# "data/Flickr8k/image_features_train/"

# Path to the file containing the evaluating image names
test_image_name_file = "Flickr/Flickr30k/Flickr_30k.testImages.txt"
# "Flickr/Flickr8k/Flickr_8k.testImages.txt"

# Path to save evaluating image features
test_save_dir = "data/Flickr30k/image_features_test/"
# "data/Flickr8k/image_features_test/"

# Get the paths to all the training images
training_image_names = subset_image_name(training_image_name_file)
training_image_paths = [image_dir + name + '.jpg' for name in training_image_names]
for path in training_image_paths:
    if not os.path.exists(path):
        training_image_paths.remove(path)

# Get the paths to all the evaluating images
test_image_names = subset_image_name(test_image_name_file)
test_image_paths = [image_dir + name + '.jpg' for name in test_image_names]
for path in test_image_paths:
    if not os.path.exists(path):
        test_image_paths.remove(path)

# Get unique training image paths
encode_train = sorted(set(training_image_paths))

# Get unique evaluating image paths
encode_test = sorted(set(test_image_paths))

# Load pretrained InceptionV3 model
image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Preprocess training images using InceptionV3 model and save as .npy files
image_dataset_train = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset_train = image_dataset_train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
# For each image
for img, path in tqdm(image_dataset_train):
    # Apply InceptionV3 model
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        name = path_of_feature.split('/')[-1]
        # Save in the training_save_dir folder
        path_of_feature = training_save_dir + name
        np.save(path_of_feature, bf.numpy())

# Preprocess evaluating images using InceptionV3 model and save as .npy files
image_dataset_test = tf.data.Dataset.from_tensor_slices(encode_test)
image_dataset_test = image_dataset_test.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
# For each image
for img, path in tqdm(image_dataset_test):
    # Apply InceptionV3 model
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        name = path_of_feature.split('/')[-1]
        # Save in the test_save_dir folder
        path_of_feature = test_save_dir + name
        np.save(path_of_feature, bf.numpy())