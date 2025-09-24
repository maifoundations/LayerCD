# !/bin/bash

for value in {3..23..2}
do
    export SELECTED_LAYER_LLAVA=$value
    echo "SELECTED_LAYER_LLAVA is $value"

    for seed in {1..5}
    do
        echo "LLaVA MCQ hcd, SELECTED_LAYER_LLAVA=$value, seed=$seed"
        CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python ablation.py --dataset=MCQ --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed
    done
done
