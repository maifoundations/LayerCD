import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys

sys.path.append(os.path.abspath("/home/bingkui/HallucinationCD/model_zoo/"))

from transformers import set_seed
from util.constant import *
from util.dataloader_utils import create_data_loader
from util.cd_utils import evolve_cd_sampling
from util.model_zoo import get_pretrained_model
evolve_cd_sampling()


def eval_model(args):
    assert not (args.layercd and args.vcd)
    
    if args.layercd:
        cd_type = 'layercd'
    elif args.vcd:
        cd_type = 'VCD'
    else:
        cd_type = 'original'
    model_type = args.model_type
    model_attribute = get_pretrained_model(model_type)
    
    answers_path = ANSWERS_PATH
    dataset = args.dataset
    if args.greedy:
        if args.num_beams == 1:
            parameter = 'greedy_alpha_{}_beta_{}_noise_step_{}'.format(args.cd_alpha,
                                                                    args.cd_beta,
                                                                    args.noise_step)
        else:
            parameter = 'beam_{}_alpha_{}_beta_{}_noise_step_{}'.format(args.num_beams,
                                                                    args.cd_alpha,
                                                                    args.cd_beta,
                                                                    args.noise_step)
    else:
        parameter = 'temperature_{}_topk_{}_topp_{}_alpha_{}_beta_{}_noise_step_{}'.format(args.temperature,
                                                                                        args.top_k,
                                                                                        args.top_p,
                                                                                        args.cd_alpha,
                                                                                        args.cd_beta,
                                                                                        args.noise_step)
        
    if dataset == 'POPE':
        pope_type = args.POPE_type
        pope_sampling_type = args.POPE_sampling_type
        assert pope_type in ['adversarial', 'popular', 'random']
        assert pope_sampling_type in ['coco', 'gqa', 'aokvqa']
        file_name = f"{pope_sampling_type}_{pope_type}_seed_{args.seed}"
        answers_path = answers_path.format(cd_type, model_type, dataset, parameter, file_name)
        questions_file = POPE_PATH.format(pope_sampling_type, pope_sampling_type, pope_type)
    elif dataset == 'MME': 
        file_name = f'MME_seed_{args.seed}'
        answers_path = answers_path.format(cd_type, model_type, dataset, parameter, file_name)
        questions_file = MME_PATH
    else:
        raise ValueError
    
    answers_path = os.path.expanduser(answers_path)
    print('Answer saves to: ', answers_path)
    os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    ans_file = open(answers_path, "w")
    data_loader, questions = create_data_loader(
        questions_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args,
        **model_attribute
    )

    for batch in tqdm(data_loader):
        indices = batch.pop('indices', None)
        input_ids = batch.get('input_ids', None)
        attention_masks = batch.get('attention_masks', None)
        image_tensor = batch.get('images_tensors', None)
        image_tensor_vcd = batch.get('images_tensors_vcd', None)
        qformer_input_ids = batch.get('qformer_input_ids', None)
        qformer_attention_mask = batch.get('qformer_attention_mak', None)

        generation_kwargs = {
                'cd_alpha': args.cd_alpha,
                'cd_beta': args.cd_beta,
                'layercd': args.layercd,
                'vcd': args.vcd,
                'model_type': model_type
            }

        if model_type == 'LLaVA':
            generation_kwargs.update({
                'inputs': input_ids.to(device='cuda', non_blocking=True),
                'images': image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                'images_vcd':image_tensor_vcd.to(dtype=torch.float16, device='cuda', non_blocking=True),
                'do_sample': not args.greedy,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': args.top_k,
                'max_new_tokens': args.max_new_tokens,
                'use_cache': True,
                'num_beams': args.num_beams
            })
        elif model_type == "Molmo":
            from transformers import GenerationConfig
            generation_kwargs.update({
                'input_ids': batch['input_ids'].to(device='cuda', non_blocking=True),
                'images': batch['image_tensor'].to(device='cuda', non_blocking=True),
                'images_vcd': batch['image_tensor_vcd'].to(device='cuda', non_blocking=True),
                'image_masks': batch['image_masks'].to(device='cuda', non_blocking=True),
                'image_input_idx': batch['image_input_idx'].to(device='cuda', non_blocking=True),
                'generation_config': GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                'tokenizer': model_attribute.get('tokenizer'),
                'use_cache': True,
                'top_p': args.top_p,
                'top_k': args.top_k,
                'temperature': args.temperature,
                'max_new_tokens': args.max_new_tokens,
                'min_new_tokens': 1,
                'do_sample': not args.greedy,
            })
        elif model_type == "Cambrian":
            image_tensor = [i.to(dtype=torch.float16, device='cuda', non_blocking=True) for i in image_tensor]
            image_tensor_vcd = [i.to(dtype=torch.float16, device='cuda', non_blocking=True) for i in image_tensor_vcd]
            generation_kwargs.update({
                'inputs': input_ids.to(device='cuda', non_blocking=True),
                'images': image_tensor,
                'images_vcd':image_tensor_vcd,
                'image_sizes': batch['image_sizes'],
                'do_sample': not args.greedy,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': args.top_k,
                'max_new_tokens': args.max_new_tokens,
                'use_cache': True
            })
        
        with torch.inference_mode():
            model = model_attribute.get('model')
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output_ids = model.generate(**generation_kwargs)
        

        tokenizer = model_attribute.get('tokenizer')
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            
        for index, output in zip(indices, outputs):
            line = questions[index]
            idx = line["question_id"]
            cur_prompt = line["text"]
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": output.strip(),
                                    "answer_id": ans_id,
                                    "model_id": model_type,
                                    "metadata": line.get('mme_type', None)}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--vcd", action='store_true', default=False)
    parser.add_argument("--layercd", action='store_true', default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--POPE_sampling_type", type=str)
    parser.add_argument("--POPE_type", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action='store_true', default=False)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)