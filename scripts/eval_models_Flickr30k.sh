#!/bin/bash
#SBATCH -J lama
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2

cd /mntnfs/med_data5/zhangzhihan/ST456/group4-codes

#time=$(date "+%m%d-%H%M")
#python train.py

python evaluate.py --model show_tell \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --batch_size 64 \
    --vocab_size 18052 \
    --max_caption_words 74 \
    --tokenizer_dir tokenizer/Flickr30k/tokenizer.json \
    --image_dir data/Flickr30k/image_features_test/ \
    --dict_dir data/Flickr30k/test_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr30k/ \
    --log_dir evaluations/Flickr30k/show_tell_flickr30k_test.csv

python evaluate.py --model bahdanau_att \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --batch_size 64 \
    --vocab_size 18052 \
    --max_caption_words 74 \
    --tokenizer_dir tokenizer/Flickr30k/tokenizer.json \
    --image_dir data/Flickr30k/image_features_test/ \
    --dict_dir data/Flickr30k/test_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr30k/ \
    --log_dir evaluations/Flickr30k/bahdanau_att_flickr30k_test.csv

python evaluate.py --model top_down \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --batch_size 64 \
    --vocab_size 18052 \
    --max_caption_words 74 \
    --tokenizer_dir tokenizer/Flickr30k/tokenizer.json \
    --image_dir data/Flickr30k/image_features_test/ \
    --dict_dir data/Flickr30k/test_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr30k/ \
    --log_dir evaluations/Flickr30k/top_down_flickr30k_test.csv

python evaluate.py --model att_on_att \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --batch_size 64 \
    --vocab_size 18052 \
    --max_caption_words 74 \
    --tokenizer_dir tokenizer/Flickr30k/tokenizer.json \
    --image_dir data/Flickr30k/image_features_test/ \
    --dict_dir data/Flickr30k/test_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr30k/ \
    --log_dir evaluations/Flickr30k/att_on_att_flickr30k_test.csv

python evaluate.py --model top_down_refiner \
    --embedding_dim 256 \
    --features_shape 1024 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --batch_size 64 \
    --vocab_size 8425 \
    --max_caption_words 74 \
    --tokenizer_dir tokenizer/Flickr8k/tokenizer.json \
    --image_dir data/Flickr30k/image_features_test/ \
    --dict_dir data/Flickr30k/test_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr30k/ \
    --log_dir evaluations/Flickr30k/top_down_refiner_flickr30k_test.csv

python evaluate.py --model top_down_gru \
    --embedding_dim 256 \
    --features_shape 2048 \
    --attention_features_shape 64 \
    --units 512 \
    --epochs 40 \
    --batch_size 64 \
    --vocab_size 18052 \
    --max_caption_words 74 \
    --tokenizer_dir tokenizer/Flickr30k/tokenizer.json \
    --image_dir data/Flickr30k/image_features_test/ \
    --dict_dir data/Flickr30k/test_dict.json \
    --checkpoint_prefix_dir checkpoints/Flickr30k/ \
    --log_dir evaluations/Flickr30k/top_down_gru_flickr30k_test.csv