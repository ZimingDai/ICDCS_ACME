#!/bin/bash

python cloud.py \
    --task_name stanford_car \
    --do_train \
    --width_mult_list 0.25,0.5,0.75,1.0 \
    --width_lambda1 1.0 \
    --width_lambda2 0.1 \
    --training_phase stanford_car_dynavitw \
    --device 0 \
    --output_dir ../model/stanford_car_dynavitw/ \
    --data_aug \
    --output_mode classification \
    --evaluate_during_training \
    --model_dir ../model/vit_stanford_car_model \
    --data_name stanford_car

