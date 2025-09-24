# !/bin/bash

for seed in {4..5}
do

    echo "coco"

    echo "LLaVA POPE normal"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=64 --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=64 --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=64 --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA POPE vcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=32 --vcd  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=32 --vcd  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=32 --vcd  --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA POPE hcd"
     /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed 
     /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed 
     /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed 
    

    #-------------------------------------
    echo "aokvqa"

    echo "LLaVA POPE normal"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=64  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=64  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=64  --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA POPE vcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=32 --vcd  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=32 --vcd  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=32 --vcd  --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA POPE hcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed 
    

    #-------------------------------------
    echo "gqa"

    echo "LLaVA POPE normal"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=64  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=64  --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=64  --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA POPE vcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=32 --vcd --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=32 --vcd --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=32 --vcd --model_type=LLaVA  --seed=$seed 
    

    echo "LLaVA POPE hcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=32 --hcd --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=32 --hcd --model_type=LLaVA  --seed=$seed 
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=32 --hcd --model_type=LLaVA  --seed=$seed 
    
    
    echo "LLaVA MME normal, seed=$seed"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=32 --model_type=LLaVA --noise_step=500 --seed=$seed

    echo "LLaVA MME vcd, seed=$seed"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=32 --vcd  --model_type=LLaVA --noise_step=500 --seed=$seed


    echo "LLaVA MME hcd, seed=$seed"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=MME    --batch_size=32 --hcd  --model_type=LLaVA --noise_step=500 --seed=$seed
done