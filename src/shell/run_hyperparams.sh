#!/bin/bash

# # Arrays for depth_mult and width_mult
# num_cell=(1 2 3)
# num_layer=(1 2 3)

# # Loop through all combinations of depth_mult and width_mult
# for cell in "${num_cell[@]}"
# do
#     for layer in "${num_layer[@]}"
#     do
#         echo "Running with num_cell=$cell and num_layer=$layer"

#         # Run the command with the set parameters
#         python train_search.py \
#             --child_num_cells $cell \
#             --child_num_layers $layer \
#             --width_mult 1 \
#             --depth_mult 1 \
#             --gpu '1' \
#             --save_path ../log/NAS_part/测试超参数
#         wait
#     done
# done

python train_search.py \
            --child_num_cells 3 \
            --child_num_layers 3 \
            --width_mult 1 \
            --depth_mult 0.75 \
            --gpu '1' \
            --dataset_name 'CIFAR100' \
            --save_path ../log/NAS_part \
            --epochs 100