# !/bin/bash

for seed in {1..5}
do
    # echo "Cambrian MCQ normal $seed"
    # /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=2 --model_type=Cambrian  --seed=$seed --num_workers=0
    

    echo "Cambrian MCQ vcd $seed"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=2 --vcd  --model_type=Cambrian  --seed=$seed --num_workers=0
    

    # echo "Cambrian MCQ hcd $seed"
    # /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=2 --hcd  --model_type=Cambrian  --seed=$seed --num_workers=0
done