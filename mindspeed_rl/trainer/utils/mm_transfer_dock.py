# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from typing import List, Dict, Union, Tuple
from operator import itemgetter

import ray
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from .transfer_dock import TransferDock


@ray.remote(max_concurrency=100, num_cpus=10)
class MMGRPOTransferDock(TransferDock):
    def __init__(self, prompts_num: int,
                 n_samples_per_prompt: int,
                 reuse_image_embeds: bool = False,
                 timeout: Union[int, None] = None,
                 timeout_interval: Union[int, None] = None) -> None:
        self.experience_columns = [
            'image',
            'pixel_values',  # 'pixel_values_videos'
            'image_grid_thw',  # 'video_grid_thw
            'image_shape',
            'labels',
            'vit_embeds',
            'position_ids',
            'image_num',
            'video',
            'video_shape',
            'video_fps',
            'video_num'
        ]
        super().__init__(
            prompts_num,
            1,
            self.experience_columns,
            timeout,
            timeout_interval
            )

        self.n_samples_per_prompt = n_samples_per_prompt
        self.consumer_columns = {
            "actor_rollout": ["image", "image_shape", "image_num", "video", "video_shape", "video_fps", "video_num"],
            "actor_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
            "ref_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
            "actor_train": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
            "rule_reward": ["labels"],
        }

        if reuse_image_embeds:
            self.consumer_columns["actor_image_embeds"] = ["pixel_values", "image_grid_thw", "image_num", "video_num"]
            self.consumer_columns["actor_rollout"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
            self.consumer_columns["actor_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
            self.consumer_columns["ref_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
            self.consumer_columns["actor_train"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]

    def get_columns(self, consumer: str):
        if consumer not in self.consumer_columns:
            return []

        for column in self.consumer_columns[consumer]:
            if all(data is None for data in self.experience_data[column]):
                return []

        return self.consumer_columns[consumer]

    def get_experience(
            self,
            experience_columns: List[str],
            indexes: List[int] = None,
            get_n_samples: bool = True,
    ):
        """Get multimodal experience data from GRPOTransferDock.

        Args:
            experience_columns: Columns from which to get data.
            indexes: Rows from which to get data.

        Returns: Data dict.

        """
        if indexes is None:
            return {}

        if get_n_samples:
            indexes = indexes[::self.n_samples_per_prompt]

        indexes = [i // self.n_samples_per_prompt for i in indexes]
        experience = []
        for single_column in experience_columns:
            if len(indexes) == 1:
                experience.append([self.experience_data[single_column][indexes[0]]])
            else:
                experience.append(list(itemgetter(*indexes)(self.experience_data[single_column])))

        return trans_mm_experience_to_output(experience, experience_columns)

    def put_experience(
            self,
            batch: Dict[str, Tensor],
            indexes: List[int] = None,
    ):
        """Put data into specified columns and rows.

        Args:
            data_dict: Data dict to put in GRPOTransferDock.
            indexes: Rows to put data in.

        Returns: None

        """
        for experience_column, experience_list in batch.items():
            if experience_column in self.experience_columns:
                for i, experience in enumerate(experience_list):
                    self.experience_data[experience_column][indexes[i]] = experience

    def clear(self):
        """Reset consumer status.Clear data and data status in GRPOTransferDock.

        Returns: None

        """
        self._clear_experience_data_and_status()


def trans_mm_experience_to_output(
        experience: List[Union[Tensor, str]],
        experience_columns: List[str],
):
    """Merge and pad multimodal data into dict.

    Args:
        experience: Data list.
        experience_columns: Columns for the corresponding data.

    Returns: Merged data dict.

    """
    batch = {}
    for i, experience_column in enumerate(experience_columns):
        if experience_column == 'labels':
            data = experience[i]
        elif experience_column in ['image_num', 'video_num', 'video_fps']:
            data = torch.tensor(experience[i]).reshape(-1, 1)
        elif experience_column in ['image', 'video']:
            data = torch.concat(experience[i], dim=1)
        else:
            data = torch.concat(experience[i])

        batch[experience_column] = data

    return batch


def calculate_split_indices(tensor_shapes: Tensor, merge_shape: bool = False) -> Tuple[List[int], List[int]]:
    """
    Calculate tensor sizes and split indices based on tensor shapes.

    Args:
        tensor_shapes: A tensor shape

    Returns:
        A tuple containing:
            - tensor_sizes: A list of total elements in each tensor
            - split_indices: A list of indices to split the flattened tensor
    """
    if merge_shape:
        from megatron.training import get_args
        merge_size = get_args().mm.model.image_encoder.vision_encoder.spatial_merge_size

    if isinstance(tensor_shapes, List):
        tensor_shapes = torch.cat(tensor_shapes)

    tensor_sizes = []
    for shape in tensor_shapes:
        size = shape.prod()
        if merge_shape:
            size //= (merge_size * merge_size)
        tensor_sizes.append(size.item())

    split_indices = [0]
    for size in tensor_sizes:
        split_indices.append(split_indices[-1] + size)

    return tensor_sizes, split_indices


def restore_images_from_tensors(flattened_tensors: Tensor, tensor_shapes: Tensor, image_num: Tensor) -> List:
    """
    Restore PIL images from tensor shapes and flattened tensors.

    Args:
        flattened_tensors: A list of flattened tensors, each representing a flattened image.
        tensor_shapes: A tensor of shape [num_images, 3] where each row contains [channels, height, width]
                      for each image.
        image_num: image nums in prompt

    Returns:
        A list of PIL Image objects reconstructed from the flattened_tensors.
    """
    tensor_sizes, split_indices = calculate_split_indices(tensor_shapes)

    reconstructed_images = []
    to_pil = transforms.ToPILImage()

    flattened_tensors = flattened_tensors.squeeze(0)
    for i in range(len(tensor_sizes)):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        flat_tensor = flattened_tensors[start_idx:end_idx]

        reconstructed_tensor = flat_tensor.reshape(tensor_shapes[i].tolist())
        reconstructed_image = to_pil(reconstructed_tensor)
        reconstructed_images.append(reconstructed_image)

    res_images = []
    start_idx = 0
    image_num = image_num.squeeze(0)
    for i in image_num:
        res_images.append(reconstructed_images[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res_images


def restore_videos_from_tensors(flattened_tensors: Tensor, tensor_shapes: Tensor, video_num: Tensor) -> List:
    tensor_sizes, split_indices = calculate_split_indices(tensor_shapes)

    reconstructed_videos = []
    flattened_tensors = flattened_tensors.squeeze(0)
    for i in range(len(tensor_sizes)):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        flat_tensor = flattened_tensors[start_idx:end_idx]

        reconstructed_videos.append(flat_tensor.reshape(tensor_shapes[i].tolist()))

    res_video = []
    start_idx = 0
    video_num = video_num.squeeze(0)
    for i in video_num:
        res_video.append(reconstructed_videos[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res_video


def restore_pixel_valaues_from_flattend(flattened_tensors: Tensor, tensor_shapes: Tensor, image_num: Tensor = None, merge_shape: bool = False) -> List[Tensor]:
    """
    Restore Ppixel_valaues from tensor shapes and flattened tensors.

    Args:
        flattened_tensors: A list of flattened tensors, each representing a flattened pixel_values.
        tensor_shapes: A tensor_shapes of original pixel_values

    Returns:
        reconstructed_pixel_values: A list of pixel_values reconstructed from the flattened_tensors.
        reconstructed_tensor_shapes: tensor_shapes repeat n at dim 0
    """
    tensor_sizes, split_indices = calculate_split_indices(tensor_shapes, merge_shape=merge_shape)

    reconstructed_pixel_values = []
    for i in range(len(tensor_sizes)):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]

        reconstructed_tensor = flattened_tensors[start_idx:end_idx, :]
        reconstructed_pixel_values.append(reconstructed_tensor)

    if image_num is None:
        return reconstructed_pixel_values

    # multi image pixel value process
    res_pixel_values = []
    start_idx = 0
    image_num = image_num.squeeze(0)
    for i in image_num:
        res_pixel_values.append(torch.cat(reconstructed_pixel_values[start_idx: start_idx + i.item()]))
        start_idx += i.item()

    return res_pixel_values


def restore_split_data(data_tensor: Tensor, split_num: Tensor):
    """
    reconstruct data like image_grid_thw, video_fps:
        [[1,30,40],[1,20,20]]    data1
        [[1,30,40],[1,20,20]]    data2
    will concat like [[1,30,40],[1,20,20],[1,30,40],[1,20,20]] -> [[[1,30,40],[1,20,20]], [[1,30,40],[1,20,20]] ]
    this func used to reconstruct by split_num recorded in data
    """
    res = []
    start_idx = 0
    split_num = split_num.squeeze(0)
    for i in split_num:
        res.append(data_tensor[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res


def restore_image_grid_thw(image_grid_thw_tensor: Tensor, image_num: Tensor):
    res_image_grid_thw = []
    start_idx = 0
    image_num = image_num.squeeze(0)
    for i in image_num:
        res_image_grid_thw.append(image_grid_thw_tensor[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res_image_grid_thw


def unpack_mm_experience(batch_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Handles multimodal data by restoring images and pixel values from flattened tensors.

    Args:
        batch_data (Dict[str, Tensor]): A dictionary containing the batch data.
        n_samples_per_prompt (int): The number of samples per prompt.

    Returns:
        Dict[str, Tensor]: The processed batch data with restored images and pixel values."
    """
    image_keys = {"image", "image_shape", "video", "video_shape"}
    pixel_values_keys = {"pixel_values", "image_grid_thw"}
    vit_embeds_keys = {"vit_embeds", "image_grid_thw"}

    if image_keys.issubset(batch_data.keys()):
        # not support hybrid image&video dataset
        if torch.sum(batch_data["image_num"]).item() > 0:
            batch_data["image"] = restore_images_from_tensors(batch_data["image"], batch_data["image_shape"], batch_data["image_num"])
        else:
            batch_data["video"] = restore_videos_from_tensors(batch_data["video"], batch_data["video_shape"], batch_data["video_num"])
            batch_data["video_fps"] = restore_split_data(batch_data["video_fps"], batch_data["video_num"])

    if pixel_values_keys.issubset(batch_data.keys()):
        mm_data_num = batch_data["image_num"] if torch.sum(batch_data["image_num"]).item() else batch_data["video_num"]
        batch_data["pixel_values"] = restore_pixel_valaues_from_flattend(batch_data["pixel_values"],
                                                                         batch_data["image_grid_thw"], mm_data_num)
        batch_data["image_grid_thw"] = restore_split_data(batch_data["image_grid_thw"], mm_data_num)

    if vit_embeds_keys.issubset(batch_data.keys()):
        mm_data_num = batch_data["image_num"] if torch.sum(batch_data["image_num"]).item() else batch_data["video_num"]
        batch_data["vit_embeds"] = restore_pixel_valaues_from_flattend(batch_data["vit_embeds"],
                                                                       batch_data["image_grid_thw"], mm_data_num,
                                                                       merge_shape=True)
        batch_data["image_grid_thw"] = restore_split_data(batch_data["image_grid_thw"], mm_data_num)

    return batch_data
