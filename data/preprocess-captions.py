import numpy as np
from tqdm import tqdm
import json
import os
import re
import tensorflow as tf
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

# Clean the captions data
def captions_clean(image_dict):
    # <key> is the image_name, which can be ignored
    for key, captions in image_dict.items():        
        # Loop through each caption for this image
        for i, caption in enumerate (captions):        
            # Convert the caption to lowercase, and then remove all special characters from it
            caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())            
            # Split the caption into separate words, and collect all words which are more than 
            # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
            clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]            
            # Join those words into a string
            caption_new = ' '.join(clean_words)            
            # Replace the old caption in the captions list with this new cleaned caption
            captions[i] = caption_new

# Path to images
image_dir = "/mntnfs/med_data5/zhangzhihan/ST456/Flickr/Flickr30k/flickr30k-images/"
#"/mntnfs/med_data5/zhangzhihan/ST456/Flickr/Flickr8k/Flicker8k_Dataset/"

# Path to the file contaiing all the image names and their captions
caption_file = "/mntnfs/med_data5/zhangzhihan/ST456/Flickr/Flickr30k/Flickr30k.token.txt"
#"/mntnfs/med_data5/zhangzhihan/ST456/Flickr/Flickr8k/Flickr8k.token.txt"
doc = load_captions(caption_file)
image_dict = captions_dict(doc)

# Clean the image dictionary
captions_clean(image_dict)

# For debug, delete the images that actually do not apear in the image dataset
for key in list(image_dict.keys()):
    path = image_dir + str(key) + '.jpg'
    if not os.path.exists(path):
        del image_dict[key]
    continue

# Add 'startseq' and 'endseq' at the beginning and end respectively of every caption
def add_token(captions):
    for i, caption in enumerate (captions):
        captions[i] = 'startseq ' + caption + ' endseq'
    return (captions)

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

# Given a set of image names, return a dictionary containing their names and corresponding captions
# {"image_name_1" : ["caption 1", "caption 2", "caption 3"],
#  "image_name_2" : ["caption 4", "caption 5"]}
def subset_data_dict(image_dict, image_names):
    dict = { image_name:add_token(captions) for image_name,captions in image_dict.items() if image_name in image_names}
    return (dict)

# Path to the file containing the training image names
training_image_name_file = "Flickr/Flickr30k/Flickr_30k.trainImages.txt"
# "Flickr/Flickr8k/Flickr_8k.trainImages.txt"

# Get the training image names
training_image_names = subset_image_name(training_image_name_file)
# Get the dictionary containing training image names and corresponding captions
test_dict = subset_data_dict(image_dict, test_image_names) 

# Path to the file containing the evaluating image names
test_image_name_file = "Flickr/Flickr30k/Flickr_30k.testImages.txt"
# "Flickr/Flickr8k/Flickr_8k.testImages.txt"

# Get the evaluating image names
test_image_names = subset_image_name(test_image_name_file)
# Get the dictionary containing evaluating image names and corresponding captions
training_dict = subset_data_dict(image_dict, training_image_names) 


# Save image dictionary, training dictionary and evaluating dictionary
image_dict = json.dumps(image_dict)
training_dict = json.dumps(training_dict)
test_dict = json.dumps(test_dict)

with open('data/Flickr30k/image_dict.json','w') as f:
    # 'data/Flickr8k/image_dict.json'
    f.write(image_dict)

with open('data/Flickr30k/training_dict.json','w') as f:
    # 'data/Flickr8k/training_dict.json'
    f.write(training_dict)

with open('data/Flickr30k/test_dict.json','w') as f:
    # 'data/Flickr8k/test_dict.json'
    f.write(test_dict)
