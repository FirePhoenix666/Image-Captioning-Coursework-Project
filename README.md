# Image Captioning Coursework Group Project

## Project description

Our project aims at training models of various depp learning architectures that could automatically gernerate a sentence describing the content of an image. This task, named image captioning, lies at the intersection of Computer Vision and Nature Language Processing.

We apply six models on image captioning task, [ShowTell](https://arxiv.org/abs/1411.4555), [BahdanauAttend](https://arxiv.org/abs/1409.0473), [TopDown](https://arxiv.org/abs/1707.07998), [AoANet](https://arxiv.org/abs/1908.06954), TopDown_Refiner and TopDown_GRU. We build the first four models by reproducing four papers in the links. While reproducing, we find that TopDown model can achieve high performance with relatively low training efforts. Therfore, we design TopDown_Refiner and TopDown_GRU based TopDown model, to discover its potential.

The codes of above six models are built on tensorflow in the folder `models\`. We train and evaluate them using `train.py` and `evaluate.py` respectively. Before that, we need to preprocess Flickr8k and Flickr30k datasets and fit the corresponding tokenizers.

The experiment runs on 4 RTX2080 GPUs.

## Preprocess Captions and Images

To reproduce our project, please download Flickr8k and Flickr30k datasets in the foder `Flickr\`. The download link is also in this folder.

### Captions

Flickr8k provide the images, `Flickr8k.token.txt` file containing the captions of all images, `Flickr_8k.trainImages.txt` file contraining the image names of the train set, and `Flickr_8k.testImages.txt` file contraining the image names of the test set. The situation is the same in Flickr30k.

One image could have up to 5 captions. We hope to get the dictionaries of the full set, the train set and the test set, whose format is shown as following.
```
{"image_name_1" : ["caption 1", "caption 2", "caption 3"],
 "image_name_2" : ["caption 4", "caption 5"]}
```

To do this, please run 
```
cd data/
python preprocess-captions.py
```

The dictionaries of the full set, the train set and the test set will be saved in `\data\Flickr8k\` and `\data\Flickr30k\` as `json` files.

### Images

We apply pretrained InceptionV3 model in tensorflow on the images of Flickr8k and Flickr30k, to get their image features.

To do this, please run
```
cd data/
python preprocess-images.py
```

The image features will be saved in `data\Flickr8k\image_features_train\`, `data\Flickr8k\image_features_test\`, `data\Flickr30k\image_features_train\` and `data\Flickr30k\image_features_test` as `.npy` files.

## Tokenizer

The tokenizer can vectorize a text corpus by creating a dictionary. The tokenizers of Flickr8k and Flickr30k should be different since their vocabularies are not exactly the same.

To fit and save the tokenizers, please run
```
cd tokenizer/
python tokenizer.py
```

## Train

Take training ShowTell model on Flickr8k for example, whose command is listed below. 
* The parameters between `embedding_dim` and `buffer_size` are chosen depending on your preferrence and/or the condition of your CPU and GPU. 
* The `vocab_size` and `max_caption_words` are fixed on which tokenizer we choose to use, this is, which dataset we choose to train on. 
* The rest parameters are paths to load data and save models' checkpoints.
```
python train.py --model show_tell \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --lr 0.001 \
    --epsilon 1e-4 \
    --batch_size 64 \
    --buffer_size 1000 \
    --vocab_size 8425 \
    --max_caption_words 35 \
    --tokenizer_dir tokenizer/Flickr8k/tokenizer.json \
    --image_dir data/Flickr8k/image_features_train/ \
    --dict_dir data/Flickr8k/training_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr8k/
```

Plase see `\scripts\train_models_Flickr8k.sh` and `\scripts\train_models_Flickr30k.sh` for parameter settings of all models.

If you use Linux system, you could run the following commands to reproduce our training results.
```
cd scripts\
train_models_Flickr8k.sh
train_models_Flickr30k.sh
```

## Evaluate

We use BLEU and METEOR metrics as evaluation methods.

Take evaluating ShowTell model on Flickr8k for example, whose command is listed below. 
* The parameters between `embedding_dim` and `buffer_size` are chosen as the same while training.
* The `vocab_size` and `max_caption_words` are fixed on which tokenizer we choose to use, this is, which dataset we choose to train on. 
* The rest parameters are paths to load data and models' checkpoints and save results.
```
python evaluate.py --model show_tell \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --batch_size 64 \
    --vocab_size 8425 \
    --max_caption_words 35 \
    --tokenizer_dir tokenizer/Flickr8k/tokenizer.json \
    --image_dir data/Flickr8k/image_features_test/ \
    --dict_dir data/Flickr8k/test_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr8k/ \
    --log_dir evaluations/Flickr8k/show_tell_flickr8k_test.csv
```

Plase see `\scripts\eval_models_Flickr8k.sh` and `\scripts\eval_models_Flickr30k.sh` for parameter settings of all models.

If you use Linux system, you could run the following commands to reproduce our evaluating results.
```
cd scripts\
eval_models_Flickr8k.sh
eval_models_Flickr30k.sh
```


