#!/bin/bash

python motiv.py \
    --head_name mlp \
    --device 0 \
    --epoch 20 \
    --batch_size 128 \
    --seed 0 \
    --width_mult_list 1.0 \
    --depth_mult_list 0.17,0.33,0.5,0.67,0.83,1.0 \
    --model_dir ../model/stanford_car_dynavit


python motiv.py \
    --head_name linear \
    --device 0 \
    --epoch 20 \
    --batch_size 128 \
    --seed 0 \
    --width_mult_list 1.0 \
    --depth_mult_list 0.17,0.33,0.5,0.67,0.83,1.0 \
    --model_dir ../model/stanford_car_dynavit