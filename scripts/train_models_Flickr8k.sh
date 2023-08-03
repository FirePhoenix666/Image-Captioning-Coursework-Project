#!/bin/bash
#SBATCH -J lama
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2

cd /mntnfs/med_data5/zhangzhihan/ST456/group4-codes

#time=$(date "+%m%d-%H%M")
#python train.py

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

python train.py --model bahdanau_att \
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

python train.py --model top_down \
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

python train.py --model att_on_att \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --lr 0.0001 \
    --epsilon 1e-4 \
    --batch_size 64 \
    --buffer_size 1000 \
    --vocab_size 8425 \
    --max_caption_words 35 \
    --tokenizer_dir tokenizer/Flickr8k/tokenizer.json \
    --image_dir data/Flickr8k/image_features_train/ \
    --dict_dir data/Flickr8k/training_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr8k/

python train.py --model top_down_refiner \
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

python train.py --model top_down_gru \
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
