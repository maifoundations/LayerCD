# !/bin/bash

for seed in {1..5}
do
    echo "LLaVA MCQ normal"
    CUDA_VISIBLE_DEVICES=0,1 /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MCQ --batch_size=16 --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA MCQ vcd"
    CUDA_VISIBLE_DEVICES=0,1 /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MCQ --batch_size=12 --vcd  --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA MCQ hcd"
    CUDA_VISIBLE_DEVICES=0,1 /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MCQ --batch_size=12 --hcd  --model_type=LLaVA  --seed=$seed 
done