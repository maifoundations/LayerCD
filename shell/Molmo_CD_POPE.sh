# !/bin/bash

for seed in {1..5}
do

    echo "coco"

    echo "Molmo POPE normal $seed"
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --model_type=Molmo  --seed=$seed 
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --model_type=Molmo  --seed=$seed 
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --model_type=Molmo  --seed=$seed 
    

    # echo "Molmo POPE vcd $seed"
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --vcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --vcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --vcd  --model_type=Molmo  --seed=$seed 


    # echo "Molmo POPE hcd $seed"
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --hcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --hcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --hcd  --model_type=Molmo  --seed=$seed 
    

    #-------------------------------------
    echo "aokvqa"

    echo "Molmo POPE normal $seed"
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8  --model_type=Molmo  --seed=$seed 
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8  --model_type=Molmo  --seed=$seed 
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8  --model_type=Molmo  --seed=$seed 
    

    # echo "Molmo POPE vcd $seed"
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8 --vcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8 --vcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8 --vcd  --model_type=Molmo  --seed=$seed 
    

    # echo "Molmo POPE hcd $seed"
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8 --hcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8 --hcd  --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8 --hcd  --model_type=Molmo  --seed=$seed 
    

    #-------------------------------------
    echo "gqa"

    echo "Molmo POPE normal $seed"
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8  --model_type=Molmo  --seed=$seed 
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8  --model_type=Molmo  --seed=$seed 
    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8  --model_type=Molmo  --seed=$seed 
    

    # echo "Molmo POPE vcd $seed"
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8 --vcd --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8 --vcd --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8 --vcd --model_type=Molmo  --seed=$seed 
    

    # echo "Molmo POPE hcd $seed"
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8 --hcd --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8 --hcd --model_type=Molmo  --seed=$seed 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/molmo/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8 --hcd --model_type=Molmo  --seed=$seed 
    

done