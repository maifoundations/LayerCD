# !/bin/bash

for seed in {1..5}
do
    # Molmo needs two gpus
    #-----
    echo "Molmo MME normal, seed=$seed"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=MME    --batch_size=8 --model_type=Molmo --noise_step=500 --seed=$seed

    echo "Molmo MME vcd, seed=$seed"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=MME    --batch_size=8 --vcd  --model_type=Molmo --noise_step=500 --seed=$seed

    echo "Molmo MME hcd, seed=$seed"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=MME    --batch_size=8 --hcd  --model_type=Molmo --noise_step=500 --seed=$seed


    # #-----
    # echo "Cambrian MME normal, seed=$seed"
    # /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=4 --model_type=Cambrian --num_workers=0 --noise_step=500 --seed=$seed

    # echo "Cambrian MME vcd, seed=$seed"
    # /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=4 --vcd  --model_type=Cambrian --num_workers=0 --noise_step=500 --seed=$seed

    # echo "Cambrian MME hcd, seed=$seed"
    # /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=4 --hcd  --model_type=Cambrian --num_workers=0 --noise_step=500 --seed=$seed


    # #-----
    # echo "LLaVA MME normal, seed=$seed"
    # CUDA_VISIBLE_DEVICES=3 /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=2 --model_type=LLaVA --noise_step=500 --seed=$seed

    # echo "LLaVA MME vcd, seed=$seed"
    # CUDA_VISIBLE_DEVICES=3 /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=2 --vcd  --model_type=LLaVA --noise_step=500 --seed=$seed


    # echo "LLaVA MME hcd, seed=$seed"
    # CUDA_VISIBLE_DEVICES=3 /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=2 --hcd  --model_type=LLaVA --noise_step=500 --seed=$seed

done