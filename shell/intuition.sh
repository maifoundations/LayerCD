# !/bin/bash

for value in {1..23}
do
    echo "LLaVA MCQ, SELECTED_LAYER_LLAVA=$value"
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python intuition.py --dataset=MCQ --batch_size=64  --model_type=LLaVA --selected_layer=$value
done
