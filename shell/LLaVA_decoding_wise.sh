# !/bin/bash

# for seed in {1..5}
# do
    
#     # top_p 0.9
#     echo "LLaVA MCQ hcd top_p 0.9"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed --top_p=0.9
    
#     # top_k 50
#     echo "LLaVA MCQ hcd top_k 50"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed --top_k=50

#     # top_k 50 + temperature 0.7
#     echo "LLaVA MCQ hcd top_k 50 + temperature 0.7"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed --top_k=50 --temperature=0.7

#     # top_k 50 + temperature 1.5
#     echo "LLaVA MCQ hcd top_k 50 + temperature 1.5"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed --top_k=50 --temperature=1.5

#     # greedy
#     echo "LLaVA MCQ hcd greedy"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32 --hcd  --model_type=LLaVA  --seed=$seed --greedy
    
#     # beam search 3
#     echo "LLaVA MCQ hcd beam search 3"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=8 --hcd  --model_type=LLaVA  --seed=$seed --greedy --num_beams=3
# done

for seed in {1..5}
do
    
    # # top_p 0.9
    # echo "LLaVA MCQ top_p 0.9"
    # CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32  --model_type=LLaVA  --seed=$seed --top_p=0.9
    
    # # top_k 50
    # echo "LLaVA MCQ top_k 50"
    # CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32  --model_type=LLaVA  --seed=$seed --top_k=50

    # # top_k 50 + temperature 0.7
    # echo "LLaVA MCQ top_k 50 + temperature 0.7"
    # CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32  --model_type=LLaVA  --seed=$seed --top_k=50 --temperature=0.7

    # # top_k 50 + temperature 1.5
    # echo "LLaVA MCQ top_k 50 + temperature 1.5"
    # CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32  --model_type=LLaVA  --seed=$seed --top_k=50 --temperature=1.5

    # # greedy
    # echo "LLaVA MCQ greedy"
    # CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=32  --model_type=LLaVA  --seed=$seed --greedy
    
    echo "LLaVA MCQ beam search 3"
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/bingkui/miniconda3/envs/cambrian/bin/python eval.py --dataset=MCQ --batch_size=16  --model_type=LLaVA  --seed=$seed --greedy --num_beams=3
done