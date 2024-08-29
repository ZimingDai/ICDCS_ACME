#!/bin/bash

# Arrays for depth_mult and width_mult
depth_mults=(0.25 0.5)
width_mults=(0.25 0.5)

# Loop through all combinations of depth_mult and width_mult
for depth in "${depth_mults[@]}"
do
    for width in "${width_mults[@]}"
    do
        echo "Running with depth_mult=$depth and width_mult=$width"

        # Check the condition and set child_num_cells accordingly
        if (( $(echo "$width * $depth > 0.25" | bc -l) )); then
            child_num_cells=1
            child_num_layers=1
        else
            child_num_cells=1
            child_num_layers=3
        fi

        # Run the command with the set parameters
        python train_search.py \
            --child_num_cells $child_num_cells \
            --child_num_layers $child_num_layers \
            --width_mult $width \
            --depth_mult $depth \
            --gpu '0'
        wait
    done
done
