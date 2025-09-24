from dataclasses import dataclass
from typing import List, Tuple
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from model_zoo.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model_zoo.LLaVA.mm_utils import tokenizer_image_token, process_images
from model_zoo.LLaVA.conversation import conv_templates
from .constant import *
from PIL import Image
import os
from util.cd_utils import add_diffusion_noise
from typing import Optional


@dataclass
class DataCollatorForVisualTextGeneration(object):
    tokenizer: transformers.PreTrainedTokenizer
    qformer_tokenizer: Optional[transformers.PreTrainedTokenizer]
    model_type: str

    def pad_sequence(self, input_ids, attention_masks, qformer_input_ids, qformer_attention_mask, batch_first, padding_value):
        # padding input_ids
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])

        # padding attention_masks
        # print(attention_masks)
        # print(type(attention_masks))
        if attention_masks[0] is not None:
            if self.tokenizer.padding_side == "left":
                attention_masks = [torch.flip(_mask, [0]) for _mask in attention_masks]
            attention_masks = torch.nn.utils.rnn.pad_sequence(
                attention_masks,
                batch_first=batch_first,
                padding_value=0)  # NOTE here I assume 0 is the attention mask value
            if self.tokenizer.padding_side == "left":
                attention_masks = torch.flip(attention_masks, [1])

        # padding qformer-wise input
        if self.qformer_tokenizer is not None:
            if self.qformer_tokenizer.padding_side == "left":
                qformer_input_ids = [torch.flip(_qformer_input_ids, [0]) for _qformer_input_ids in qformer_input_ids]
            qformer_input_ids = torch.nn.utils.rnn.pad_sequence(
                    qformer_input_ids,
                    batch_first=batch_first,
                    padding_value=0)  # NOTE here I assume 0 is the attention mask value
            if self.qformer_tokenizer.padding_side == "left":
                qformer_input_ids = torch.flip(qformer_input_ids, [1])
            
            if self.qformer_tokenizer.padding_side == "left":
                qformer_attention_mask = [torch.flip(_qformer_attention_mask, [0]) for _qformer_attention_mask in qformer_attention_mask]
            qformer_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    qformer_attention_mask,
                    batch_first=batch_first,
                    padding_value=0)  # NOTE here I assume 0 is the attention mask value
            if self.qformer_tokenizer.padding_side == "left":
                qformer_attention_mask = torch.flip(qformer_attention_mask, [1])

        return input_ids, attention_masks, qformer_input_ids, qformer_attention_mask
    
    def pad_molmo(self, batch):
        max_patch_cnt = -1 
        for data in batch:
            max_patch_cnt = max(max_patch_cnt, data['image_tensor'].shape[0])
        images = []
        images_vcd = []
        images_input_indices = []
        image_masks = []
        input_ids = []
        indices = []
        for data in batch:
            indices.append(data['index'])
            image = data['image_tensor']
            image_vcd = data['image_tensor_vcd']
            image_input_idx = data['image_input_idx']
            image_mask = data['image_masks']
            input_id = data['input_ids']
            input_ids.append(input_id)
            if image.shape[0] < max_patch_cnt:
                # align image
                mask_shape = (max_patch_cnt - image.shape[0], image.shape[1], image.shape[2])
                masks = torch.full(mask_shape, 0)
                image = torch.cat((image, masks), dim=0)
                images.append(image)

                mask_shape = (max_patch_cnt - image_vcd.shape[0], image_vcd.shape[1], image_vcd.shape[2])
                masks = torch.full(mask_shape, 0)
                image_vcd = torch.cat((image_vcd, masks), dim=0)
                images_vcd.append(image_vcd)

                # align image_input_idx 
                mask_shape = (max_patch_cnt - image_input_idx.shape[0], image_input_idx.shape[1])
                input_idx = torch.full(mask_shape, -100)
                image_input_idx = torch.cat((image_input_idx, input_idx), dim=0)
                images_input_indices.append(image_input_idx)

                # align image_mask
                mask_shape = (max_patch_cnt - image_mask.shape[0], image_mask.shape[1])
                mask = torch.full(mask_shape, -1)
                image_mask = torch.cat((image_mask, mask), dim=0)
                image_masks.append(image_mask)

            else:
                images_vcd.append(image_vcd)
                images.append(image)
                images_input_indices.append(image_input_idx)
                image_masks.append(image_mask)

        images_tensors = torch.stack(images)
        images_tensors_vcd = torch.stack(images_vcd)
        images_input_indices = torch.stack(images_input_indices)
        image_masks = torch.stack(image_masks)
        
        padding_token_id = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=padding_token_id)
        return {
            'indices': indices,
            'input_ids': input_ids,
            'image_tensor': images_tensors,
            'image_tensor_vcd': images_tensors_vcd,
            'image_input_idx': images_input_indices,
            'image_masks': image_masks
        }


    def __call__(self, batch):
        if self.model_type == 'Molmo':
            return self.pad_molmo(batch=batch)
        else:
            fields: list = ['index', 'input_ids', 'image_tensor', 'image_tensor_vcd', 'attention_mask', 'qformer_input_ids', 'qformer_attention_mask', 'image_size']
            indices, input_ids, images, images_vcd, attention_masks, qformer_input_ids, qformer_attention_mask, image_sizes = (
                [data.get(field, None) for data in batch] for field in fields
            )

            input_ids, attention_masks, qformer_input_ids, qformer_attention_mask = self.pad_sequence(
                input_ids,
                attention_masks,
                qformer_input_ids, # padding side: right
                qformer_attention_mask, # padding side: right
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            if isinstance(images[0], list):
                images_final, images_vcd_final = [], []
                for idx in range(len(images[0])):
                    images_at_idx = [img[idx] for img in images]
                    images_vcd_at_idx = [img[idx] for img in images_vcd]
                    images_final.append(torch.cat(images_at_idx, dim=0))
                    images_vcd_final.append(torch.cat(images_vcd_at_idx, dim=0))
                images_tensors = images_final
                images_tensors_vcd = images_vcd_final
            else:
                images_tensors = torch.cat([image.unsqueeze(0) for image in images], dim=0)
                images_tensors_vcd = torch.cat([image_vcd.unsqueeze(0) for image_vcd in images_vcd], dim=0)

            return {
                'indices': indices,
                'input_ids': input_ids,
                'attention_masks': attention_masks,
                'images_tensors': images_tensors,
                'images_tensors_vcd': images_tensors_vcd,
                'qformer_input_ids': qformer_input_ids,
                'qformer_attention_mask': qformer_attention_mask,
                'image_sizes': image_sizes
            }


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, args, **kwargs):
        self.questions = questions
        self.tokenizer = kwargs.get('tokenizer')
        self.image_processor = kwargs.get('image_processor')
        self.qformer_tokenizer = kwargs.get('qformer_tokenizer', None)
        if kwargs.get('model_config') is not None:
            self.model_config = kwargs.get('model_config')
        if args.dataset == 'POPE':
            self.image_folder = IMAGE_BASE
        elif args.dataset == 'MME':
            self.image_folder = MME_PATH.replace('index', 'data')
        else:
            self.image_folder = None
        self.noise_step = args.noise_step
        self.model_type = args.model_type

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        if self.image_folder is not None:
            image_file = os.path.join(self.image_folder, image_file)
        qs = line["text"]
        data = {}
        
        if self.model_type == 'LLaVA':
            from model_zoo.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from model_zoo.LLaVA.mm_utils import tokenizer_image_token, process_images
            from model_zoo.LLaVA.conversation import conv_templates
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates[CONV_MODE].copy()
            conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            image = Image.open(image_file).convert("RGB")
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            image_tensor_vcd = add_diffusion_noise(image_tensor, self.noise_step)

            data.update({
                'index': index, 
                'input_ids': input_ids,
                'image_tensor': image_tensor,
                'image_tensor_vcd': image_tensor_vcd
            })
        elif self.model_type == 'Molmo':
            image = Image.open(image_file).convert("RGB")
            inputs = self.image_processor.process(images=image, text=qs)
            images_vcd = add_diffusion_noise(inputs['images'], self.noise_step)
            data.update({
                'index': index, 
                'input_ids': inputs['input_ids'],
                'image_tensor': inputs['images'],
                'image_tensor_vcd': images_vcd,
                'image_masks': inputs['image_masks'],
                'image_input_idx': inputs['image_input_idx']
            })
        elif self.model_type == 'Cambrian':
            from model_zoo.Cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from model_zoo.Cambrian.conversation import conv_templates
            from model_zoo.Cambrian.mm_utils import process_images, tokenizer_image_token
            image = Image.open(image_file).convert("RGB")

            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates['llama_3'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            image_size = image.size
            image_tensor = process_images([image], self.image_processor, self.model_config)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            image_tensor_vcd = []
            for img in image_tensor:
                image_tensor_vcd.append(add_diffusion_noise(img, self.noise_step))
            
            data.update({
                'index': index, 
                'input_ids': input_ids,
                'image_tensor': image_tensor,
                'image_tensor_vcd': image_tensor_vcd,
                'image_size': image_size
            })
        return data

    def __len__(self):
        return len(self.questions)


def mme_process(folder_path):
    data_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_idx, line in enumerate(file):
                    line_parts = line.strip().split('\t')
                    
                    if len(line_parts) == 3:
                        line_dict = {
                            'image': line_parts[0],
                            'text': line_parts[1],
                            'answer': line_parts[2],
                            'question_id': line_idx,
                            'mme_type': filename.split('.')[0]
                        }
                        data_list.append(line_dict)
    return data_list


# DataLoader
def create_data_loader(questions_file, batch_size=1, num_workers=4, args=None, **kwargs):
    import json
    dataset = args.dataset
    if dataset in ['POPE']:
        questions = [json.loads(q) for q in open(os.path.expanduser(questions_file), "r")]
    elif dataset in ['MME']:
        questions = mme_process(questions_file)
    else:
        raise ValueError('Unknow benchmark name.')

    dataset = CustomDataset(questions, args, **kwargs)
    collator = DataCollatorForVisualTextGeneration(tokenizer=kwargs.get('tokenizer'),
                                                qformer_tokenizer=kwargs.get('qformer_tokenizer', None),
                                                model_type=args.model_type)
    data_loader = DataLoader(dataset, collate_fn=collator, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader, questions