#!/bin/bash

echo "MCQ normal"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=MCQ --batch_size=48

echo "POPE normal"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=random         --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=adversarial    --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=popular        --batch_size=48

echo "MCQ chosen layer idx: 1"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=MCQ --clip_chosen_layer_idx=1 --batch_size=48

echo "POPE chosen layer idx: 1"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=random         --clip_chosen_layer_idx=1   --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=adversarial    --clip_chosen_layer_idx=1   --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=popular        --clip_chosen_layer_idx=1   --batch_size=48


echo "MCQ chosen layer idx: 4"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=MCQ --clip_chosen_layer_idx=4 --batch_size=48

echo "POPE chosen layer idx: 4"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=random         --clip_chosen_layer_idx=4   --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=adversarial    --clip_chosen_layer_idx=4   --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=popular        --clip_chosen_layer_idx=4   --batch_size=48

echo "MCQ chosen layer idx: 8"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=MCQ --clip_chosen_layer_idx=8 --batch_size=48

echo "POPE chosen layer idx: 8"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=random         --clip_chosen_layer_idx=8   --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=adversarial    --clip_chosen_layer_idx=8   --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=popular        --clip_chosen_layer_idx=8   --batch_size=48

echo "MCQ chosen layer idx: 12"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=MCQ --clip_chosen_layer_idx=12 --batch_size=48

echo "POPE chosen layer idx: 12"
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=random         --clip_chosen_layer_idx=12  --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=adversarial    --clip_chosen_layer_idx=12  --batch_size=48
/home/bingkui/miniconda3/envs/torch/bin/python HallucinationCD.py --dataset=POPE --POPE_type=popular        --clip_chosen_layer_idx=12  --batch_size=48



echo "所有任务执行完毕！"