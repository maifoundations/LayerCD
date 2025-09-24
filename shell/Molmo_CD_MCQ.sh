# !/bin/bash

for seed in {1..5}
do
    echo "Molmo MCQ normal $seed"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=MCQ --batch_size=8 --model_type=Molmo  --seed=$seed
    

    echo "Molmo MCQ vcd $seed"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=MCQ --batch_size=8 --vcd  --model_type=Molmo  --seed=$seed
    

    echo "Molmo MCQ hcd $seed"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=MCQ --batch_size=8 --hcd  --model_type=Molmo  --seed=$seed
done