# !/bin/bash

echo "coco"

echo "InstructBlip POPE normal"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --model_type=InstructBlip

echo "InstructBlip POPE vcd"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --vcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --vcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --vcd  --model_type=InstructBlip


echo "InstructBlip POPE hcd"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --hcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --hcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --hcd  --model_type=InstructBlip


#-------------------------------------
echo "aokvqa"

echo "InstructBlip POPE normal"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8  --model_type=InstructBlip

echo "InstructBlip POPE vcd"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8 --vcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8 --vcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8 --vcd  --model_type=InstructBlip


echo "InstructBlip POPE hcd"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8 --hcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8 --hcd  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8 --hcd  --model_type=InstructBlip


#-------------------------------------
echo "gqa"

echo "InstructBlip POPE normal"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8  --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8  --model_type=InstructBlip

echo "InstructBlip POPE vcd"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8 --vcd --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8 --vcd --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8 --vcd --model_type=InstructBlip


echo "InstructBlip POPE hcd"
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8 --hcd --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8 --hcd --model_type=InstructBlip
CUDA_VISIBLE_DEVICES=4,5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8 --hcd --model_type=InstructBlip

