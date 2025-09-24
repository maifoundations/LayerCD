#!/bin/bash

for alpha in 0.2 0.4 0.6 0.8
do
    for seed in {1..5}
    do
        echo "LLaVA MCQ hcd, seed=$seed, alpha=$alpha"
        CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32 --hcd --model_type=LLaVA --seed=$seed --cd_alpha=$alpha
    done
done


for beta  in 0.001 0.01 0.2 0.5 0.9
do
    for seed in {1..5}
    do
        echo "LLaVA MCQ hcd, seed=$seed, alpha=$alpha"
        CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32 --hcd --model_type=LLaVA --seed=$seed  --cd_beta=$beta
    done
done
