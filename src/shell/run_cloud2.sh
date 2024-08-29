#!/bin/bash

python cloud.py \
    --task_name stanford_car \
    --do_train \
    --width_mult_list 0.25,0.5,0.75,1.0 \
    --depth_mult_list 0.25,0.5,0.75,1.0 \
    --width_lambda1 1.0 \
    --width_lambda2 0.1 \
    --depth_lambda1 1.0 \
	  --depth_lambda2 1.0 \
    --training_phase dynavit \
    --device 0 \
    --data_aug \
    --output_mode classification \
    --evaluate_during_training \
    --output_dir ../model/stanford_car_dynavit/ \
    --model_dir ../model/stanford_car_dynavitw \
    --data_name stanford_car

