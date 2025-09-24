# !/bin/bash

echo "coco"

echo "QWen_VL POPE normal"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --model_type=QWen_VL

echo "QWen_VL POPE vcd"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --vcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --vcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --vcd  --model_type=QWen_VL


echo "QWen_VL POPE hcd"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=random         --batch_size=8 --hcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=adversarial    --batch_size=8 --hcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=coco --POPE_type=popular        --batch_size=8 --hcd  --model_type=QWen_VL


#-------------------------------------
echo "aokvqa"

echo "QWen_VL POPE normal"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8  --model_type=QWen_VL

echo "QWen_VL POPE vcd"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8 --vcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8 --vcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8 --vcd  --model_type=QWen_VL


echo "QWen_VL POPE hcd"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=random         --batch_size=8 --hcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=adversarial    --batch_size=8 --hcd  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=aokvqa --POPE_type=popular        --batch_size=8 --hcd  --model_type=QWen_VL


#-------------------------------------
echo "gqa"

echo "QWen_VL POPE normal"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8  --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8  --model_type=QWen_VL

echo "QWen_VL POPE vcd"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8 --vcd --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8 --vcd --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8 --vcd --model_type=QWen_VL


echo "QWen_VL POPE hcd"
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=random         --batch_size=8 --hcd --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=adversarial    --batch_size=8 --hcd --model_type=QWen_VL
CUDA_VISIBLE_DEVICES=5 /home/bingkui/miniconda3/envs/XCD/bin/python eval.py --dataset=POPE --POPE_sampling_type=gqa --POPE_type=popular        --batch_size=8 --hcd --model_type=QWen_VL

