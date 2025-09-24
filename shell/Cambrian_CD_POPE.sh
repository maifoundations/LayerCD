# !/bin/bash

for seed in {5..5}
do

    echo "coco"

    echo "Cambrian POPE normal"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=4 --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=4 --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=4 --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    echo "Cambrian POPE vcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=4 --vcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=4 --vcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=4 --vcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    echo "Cambrian POPE hcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=4 --hcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=4 --hcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=4 --hcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    #-------------------------------------
    echo "aokvqa"

    echo "Cambrian POPE normal"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=4  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=4  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=4  --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    echo "Cambrian POPE vcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=4 --vcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=4 --vcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=4 --vcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    echo "Cambrian POPE hcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=4 --hcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=4 --hcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=4 --hcd  --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    #-------------------------------------
    echo "gqa"

    echo "Cambrian POPE normal"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=4  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=4  --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=4  --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    echo "Cambrian POPE vcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=4 --vcd --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=4 --vcd --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=4 --vcd --model_type=Cambrian  --num_workers=0 --seed=$seed
    

    echo "Cambrian POPE hcd"
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=4 --hcd --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=4 --hcd --model_type=Cambrian  --num_workers=0 --seed=$seed
    /home/bingkui/miniconda3/envs/LLaVA/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=4 --hcd --model_type=Cambrian  --num_workers=0 --seed=$seed
    

done