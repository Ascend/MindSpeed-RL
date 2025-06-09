# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and or its affiliates

import os
import torch
from transformers import PreTrainedTokenizer
from datasets import load_dataset
import torchvision.transforms as transforms
from qwen_vl_utils import fetch_video

from mindspeed_rl.datasets.mm_utils import process_image, process_videos
from mindspeed_rl.datasets.mm_utils import get_rope_index, compute_position_id_with_mask, tokenize_and_postprocess_data
from mindspeed_rl.datasets.base_dataset import BaseDataset


class MultiModalDataset(BaseDataset):
    """
    dataset used for multimodal
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 processor=None,
                 prompt_key='prompt',
                 image_key='images',
                 video_key='videos',
                 max_prompt_length=1024,
                 return_raw_chat=False,
                 truncation='error'):
        self.tokenizer = tokenizer.tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.video_key = video_key
        self.max_prompt_length = max_prompt_length

        self.return_raw_chat = return_raw_chat
        self.truncation = truncation


        if os.path.isdir(data_path):
            dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            dataset = load_dataset(data_path, split="train")

        super().__init__(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataset[index]
        chat = row_dict.pop(self.prompt_key)
        has_image = self.image_key in row_dict

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        self.init_multimodal_row_dict(row_dict)
        is_multi_modal = self.image_key in row_dict or self.video_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            raw_prompt = raw_prompt.replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {}
            input_dict = self.multimodal_process(row_dict)

            mm_inputs = self.processor.image_processor(**input_dict, return_tensors='pt')

            row_dict["labels"] = row_dict["extra_info"]["answer"]

            image_grid_thw = mm_inputs['image_grid_thw'] if has_image else mm_inputs['video_grid_thw']
            row_dict['pixel_values'] = mm_inputs['pixel_values'] if has_image else mm_inputs['pixel_values_videos']
            row_dict['image_grid_thw'] = image_grid_thw
            row_dict['multi_modal_inputs'] = {key: val for key, val in mm_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size ** 2
                image_index, video_index = 0, 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[image_index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    image_index += 1
                row_dict['image_num'] = image_index
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)

                while '<video>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<video>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[video_index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    video_index += 1
                row_dict['video_num'] = video_index
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.video_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=mm_inputs.get('image_grid_thw'),
                video_grid_thw=mm_inputs.get('video_grid_thw'),
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict["input_ids_length"] = torch.tensor([attention_mask.sum().item()])
        row_dict['position_ids'] = position_ids
        row_dict['prompts'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def multimodal_process(self, row_dict):
        to_tensor = transforms.ToTensor()
        has_image = self.image_key in row_dict
        has_video = self.video_key in row_dict
        input_dict = {"images": None}
        if has_image:
            row_dict['multi_modal_data']['image'] = [process_image(image, 2048 * 2048, 512 * 512) for image in row_dict.pop(self.image_key)]
            input_dict['images'] = row_dict['multi_modal_data']['image']
            flattened_images = []
            image_shapes = []
            for m in row_dict['multi_modal_data']['image']:
                image = to_tensor(m)
                image_shapes.append(torch.tensor(image.shape))
                image = image.reshape(1, -1)
                flattened_images.append(image)

            row_dict["image"] = torch.cat(flattened_images, dim=1)
            row_dict["image_shape"] = torch.stack(image_shapes, dim=0)

        elif has_video:
            row_dict['multi_modal_data']['video'] = [process_videos(video,
                                                                    image_resolution=getattr(
                                                                        self.processor.image_processor,
                                                                        "video_resolution", 256 * 256),
                                                                    video_fps=getattr(self.processor.image_processor,
                                                                                      "video_fps", 2.0),
                                                                    video_maxlen=getattr(self.processor.image_processor,
                                                                                         "video_maxlen", 128))
                                                     for video in row_dict[self.video_key]]
            input_dict['videos'] = row_dict['multi_modal_data']['video']
            # vllm video inputs
            video_inputs = []
            row_dict["video_shape"] = []
            row_dict["video_fps"] = []
            for video in row_dict.pop(self.video_key):
                video_input, video_sample_fps = fetch_video({'min_pixels': 512 * 512,
                                                             'total_pixels': 20480 * 28 * 28,
                                                             'type': 'video', 'video': video},
                                                            return_video_sample_fps=True)
                video_inputs.append(video_input.reshape(1, -1))
                row_dict["video_shape"].append(torch.tensor(video_input.shape))
                row_dict["video_fps"].append(video_sample_fps)
            row_dict["video"] = torch.cat(video_inputs, dim=1)
            row_dict["video_shape"] = torch.stack(row_dict["video_shape"], dim=0)

        else:
            raise ValueError('current mode must be in image or video!')
        return input_dict

    @staticmethod
    def init_multimodal_row_dict(row_dict):
        row_dict["image"] = torch.zeros(1, 1)
        row_dict["image_shape"] = torch.zeros(1, 1)
        row_dict["image_num"] = 0
        row_dict["video"] = torch.zeros(1, 1)
        row_dict["video_shape"] = torch.zeros(1, 1)
        row_dict["video_fps"] = []
        # 视频个数
        row_dict["video_num"] = 0
        # 所有视频的pixel_value个数
        row_dict["video_image_num"] = 0

