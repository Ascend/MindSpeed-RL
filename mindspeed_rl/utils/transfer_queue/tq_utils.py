"""
Provides utility functions for tensor and experience data manipulation in distributed reinforcement learning.  

Supports padding, packing, serialization, and load balancing for variable-length sequences (e.g., prompts/responses) and multimodal inputs (images, videos).  
Key features include length-balanced batching, tensor packing/unpacking, binary serialization, and format conversion.  
All operations are PyTorch-compatible and optimized for large-scale actor-critic or policy gradient systems.
"""
import copy
import struct
import heapq
from typing import Dict, List, Optional, Tuple, Union, Any

import logging as logger
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from mindspeed_rl.utils.pad_process import padding_dict_to_tensor_dict


def pad_experience(
        experience_batch: Dict[str, List[torch.Tensor]],
        pad_id: int,
        multiple: int = 1,
):
    """ Pad dict data.

    Args:
        experience_batch: Dict
            {
                'prompts': [ tensor([1, 1, 1, 1]),
                             tensor([2, 2, 2, 2]),
                             tensor([3, 3, 3, 3]),
                             tensor([4, 4, 4, 4])],
                'attention_mask': [ tensor([1]),
                                    tensor([2, 2]),
                                    tensor([3, 3, 3]),
                                    tensor([4, 4, 4, 4])],
            }
        pad_id: Pad token.
            0.0
        multiple: The multiple of TP to pad.
            1

    Returns: Merged and padded data dict.
        {
            "prompts": tensor(
                [[1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4]]),
            "attention_mask": tensor(
                [[1, 0, 0, 0],
                [2, 2, 0, 0],
                [3, 3, 3, 0],
                [4, 4, 4, 4]]),
        }

    """
    def pad_multiples(data_list: List[torch.Tensor], pad_id: Union[float, int], multiple: int = 1) -> torch.Tensor:
        """Pad method for data list.

        Args:
            data_list: Data list.
            pad_id: Pad token.
            multiple: The multiple of TP to pad.

        Returns: Padded tensor.

        """
        padded = pad_sequence(data_list, batch_first=True, padding_value=pad_id)
        max_len = padded.size(1)
        target_len = ((max_len + multiple - 1) // multiple) * multiple
        padded = F.pad(padded, (0, target_len - max_len), value=pad_id)
        return padded

    batch = {}
    if not experience_batch:
        raise ValueError("ERROR: when pad, get an empty experience_batch")
    else:
        for experience_column, experience in experience_batch.items():
            if experience_column in ["prompt_length", "response_length", "age"]:
                padded = torch.cat(experience).reshape(-1, 1)
            elif experience_column in ["position_ids"]:
                padded = pad_sequence(experience, batch_first=True, padding_value=pad_id)
            elif experience[0].is_floating_point():
                padded = pad_multiples(experience, pad_id=0.0, multiple=multiple)
            else:
                padded = pad_multiples(experience, pad_id=pad_id, multiple=multiple)

            batch[experience_column] = padded

    return batch


def pack_experience_columns(experience_consumer_stage, experience_dict, experience_count, enable_partial_rollout=False):
    """
    Compress experiences by packing tensors into ONE.
    from experience_dict
        {
            'prompts': [ tensor([1, 1, 1]),
                            tensor([2, 2, 2, 2]),
                            tensor([3, 3, 3]),
                            tensor([4, 4, 4, 4])],
            'attention_mask': [ tensor([1]),
                                tensor([2, 2]),
                                tensor([3, 3, 3]),
                                tensor([4, 4, 4, 4])],
        }
    To batch_data
        {
            'prompts': tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'attention_mask': tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        }
        batch_data_length
        {
            'prompts': tensor([3, 4, 3, 4]),
            'attention_mask': tensor([1, 2, 3, 4])
        }
    """

    if not experience_dict:
        raise ValueError(f"ERROR: when pack, get an empty experience_dict")

    batch_data = {}
    batch_data_length = {}

    if enable_partial_rollout and experience_consumer_stage == 'actor_rollout':
        value = experience_dict['prompts']
        experience_count = len(value)
    else:
        for key, value in experience_dict.items():
            if len(value) != experience_count:
                raise ValueError(f"ERROR: when pack, experience '{key}' number={len(value)} does not match {experience_count=}")

    for key, value in experience_dict.items():
        is_2d = len(value[0].shape) > 1
        if is_2d:
            first_dim = value[0].shape[0]
            for i in range(experience_count):
                if value[i].shape[0] != first_dim:
                    raise ValueError(f"ERROR: when pack 2D tensor, first dimension must be the same for all experiences")

            packed_data = []
            for dim_idx in range(first_dim):
                dim_data = []
                for i in range(experience_count):
                    dim_data.extend(value[i][dim_idx].tolist())
                packed_data.append(dim_data)

            batch_data[key] = torch.tensor(packed_data, dtype=value[0].dtype)
            data_length = [value[i].shape[1] for i in range(experience_count)]
            batch_data_length[key] = torch.tensor(data_length, dtype=torch.int32)
        else:
            packed_experience = []
            data_length = []
            for i in range(experience_count):
                packed_experience.extend(value[i].tolist())
                data_length.append(len(value[i]))

            batch_data[key] = torch.tensor(packed_experience, dtype=value[0].dtype)
            batch_data_length[key] = torch.tensor(data_length, dtype=torch.int32)

    return batch_data, batch_data_length


def unpack_pad_experience(batch_data, batch_data_length, pad_id, multiple):
    """
    1. restore the received experience dict
    2. pad the tensor (consider the requirement of multiple)
    from batch_data
        {
            'prompts': tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'attention_mask': tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        }
        batch_data_length
        {
            'prompts': tensor([3, 4, 3, 4]),
            'attention_mask': tensor([1, 2, 3, 4])
        }
    To padded_batch_data (multiple=2)
        {
            "prompts": tensor(
                [[1, 1, 1, -1, -1, -1, -1, -1],
                [2, 2, 2, 2, -1, -1, -1, -1],
                [3, 3, 3, -1, -1, -1, -1, -1],
                [4, 4, 4, 4, -1, -1, -1, -1]]),
            "attention_mask": tensor(
                [[1, -1, -1, -1, -1, -1, -1, -1],
                [2, 2, -1, -1, -1, -1, -1, -1],
                [3, 3, 3, -1, -1, -1, -1, -1],
                [4, 4, 4, 4, -1, -1, -1, -1]]),
        }
    """
    if not batch_data:
        raise ValueError(f"ERROR: empty batch_data")

    if set(batch_data.keys()) != set(batch_data_length.keys()):
        raise ValueError(f"ERROR: when unpack, keys from batch_data and batch_data_length dictionaries do not match")

    data_device = batch_data[list(batch_data.keys())[0]].device

    padded_batch_data = {}
    for key, length_list in batch_data_length.items():
        if key in ['prompt_length', 'response_length', 'age']:
            padded_batch_data[key] = batch_data[key].view(-1, 1)
            continue
        data = batch_data[key]
        data_dtype = batch_data[key].dtype
        lengths = length_list.to(data_device)

        is_2d = len(data.shape) > 1
        if is_2d:
            first_dim = data.shape[0]
            max_row_len = torch.max(lengths).item()
            if multiple > 1:
                max_row_len = ((max_row_len + multiple - 1) // multiple) * multiple

            sample_count = len(lengths)
            result = []
            if data[0].is_floating_point():
                padded_tensor = torch.full((sample_count, first_dim, max_row_len), 0.0,
                                          dtype=data_dtype, device=data_device)
            else:
                padded_tensor = torch.full((sample_count, first_dim, max_row_len), pad_id,
                                          dtype=data_dtype, device=data_device)

            cum_length = torch.cat([torch.tensor([0], device=data_device),
                                   torch.cumsum(lengths, 0)])
            for i in range(sample_count):
                seq_len = lengths[i]
                for dim_idx in range(first_dim):
                    start_idx = cum_length[i]
                    end_idx = cum_length[i] + seq_len
                    padded_tensor[i, dim_idx, :seq_len] = data[dim_idx, start_idx:end_idx]

            padded_batch_data[key] = padded_tensor
        else:
            max_row_len = torch.max(lengths).item()
            if multiple > 1:
                max_row_len = ((max_row_len + multiple - 1) // multiple) * multiple

            if data.is_floating_point():
                padded_tensor = torch.full((len(lengths), max_row_len), 0.0,
                                       dtype=data_dtype, device=data_device)
            else:
                padded_tensor = torch.full((len(lengths), max_row_len), pad_id,
                                       dtype=data_dtype, device=data_device)

            cum_length = torch.cat([torch.tensor([0], device=data_device
                                             ), torch.cumsum(lengths, 0)])
            for i, _ in enumerate(lengths):
                seq_len = lengths[i]
                padded_tensor[i, :seq_len] = data[cum_length[i]:cum_length[i + 1]]
            padded_batch_data[key] = padded_tensor

    return padded_batch_data


def trans_data_dict_to_experience(
        experience_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
) -> Tuple[List[str], List[List[torch.Tensor]]]:
    """
    Split data dict into columns and data list. int64->int32

    Args:
        experience_dict: Data dict.
            {
                "prompts": tensor(
                    [[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4]]),
                "attention_mask": [
                    tensor([1]),
                    tensor([2, 2]),
                    tensor([3, 3, 3]),
                    tensor([4, 4, 4, 4])]
            }

    Returns: Columns and data list.
        ['prompts', 'attention_mask']
        [
            [
                tensor([1, 1, 1, 1]),
                tensor([2, 2, 2, 2]),
                tensor([3, 3, 3, 3]),
                tensor([4, 4, 4, 4])
            ],
            [
                tensor([1]),
                tensor([2, 2]),
                tensor([3, 3, 3]),
                tensor([4, 4, 4, 4])
            ]
        ]

    """
    experience_columns = []
    experience_list = []
    for key, value in experience_dict.items():
        if value is not None:
            experience_columns.append(key)
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.int64:
                    value = value.to(torch.int32)
                value = list(torch.unbind(value, dim=0))
            elif isinstance(value, List):
                if isinstance(value[0], torch.Tensor):
                    value = [val.to(torch.int32) if val.dtype == torch.int64 else val for val in value]
                else:
                    raise ValueError(f"Unsupported list element type: {type(value[0])} (expected tensor)")
            else:
                raise ValueError(f"value type {type(value)} not supported")
            experience_list.append(value)

    return experience_columns, experience_list


DTYPE_TO_ID: Dict[str, int] = {
    'torch.float32': 0, 'torch.float64': 1, 'torch.int32': 2,
    'torch.int64': 3, 'torch.uint8': 4, 'torch.int8': 5,
    'torch.int16': 6, 'torch.bool': 10, 'torch.float16': 11,
}

ID_TO_DTYPE: Dict[int, torch.dtype] = {
    0: torch.float32, 1: torch.float64, 2: torch.int32,
    3: torch.int64, 4: torch.uint8, 5: torch.int8,
    6: torch.int16, 10: torch.bool, 11: torch.float16,
}


def serialize_tensor_lists(data: List[List[torch.Tensor]]) -> bytes:
    """
    Serializes a List[List[torch.Tensor]] into bytes.

    Assumptions:
    - Tensors within a sub-list must have the same dtype.
    - Tensors are 1D-tensor
    """
    header_chunks = []
    data_chunks = []  # Temporarily stores the bytes of each tensor

    # 1. Prepare metadata and data chunks
    header_chunks.append(struct.pack('>I', len(data)))
    for sub_list in data:
        if not sub_list:
            # Handle empty sub-lists
            header_chunks.append(struct.pack('>BI', DTYPE_TO_ID['torch.float32'], 0))
            continue

        first_tensor = sub_list[0]
        dtype_str = str(first_tensor.dtype)

        if dtype_str not in DTYPE_TO_ID:
            raise TypeError(f"Unsupported dtype: {dtype_str}. Please add it to the DTYPE_TO_ID map.")

        dtype_id = DTYPE_TO_ID[dtype_str]
        num_tensors = len(sub_list)

        header_chunks.append(struct.pack('>BI', dtype_id, num_tensors))

        lengths = []
        for tensor in sub_list:
            # First copy: from Tensor memory to an independent bytes object
            data_chunks.append(tensor.cpu().numpy().tobytes())
            lengths.append(len(tensor))

        header_chunks.append(struct.pack(f'>{num_tensors}I', *lengths))

    # 2. Aggregate metadata and data
    header_bytes = b''.join(header_chunks)

    # Second copy: merge all independent data chunks into one large payload.
    # This step is handled by the highly optimized C implementation of b''.join() and is very fast.
    payload_bytes = b''.join(data_chunks)

    # 3. Assemble the final message
    header_len_bytes = struct.pack('>Q', len(header_bytes))

    _result = header_len_bytes + header_bytes + payload_bytes
    return _result


def deserialize_tensor_lists(serialized_data: bytes) -> List[List[torch.Tensor]]:
    """
    The corresponding pure Python deserialization function.
    """
    # 1. Unpack the metadata header
    header_len = struct.unpack('>Q', serialized_data[:8])[0]
    header_end = 8 + header_len
    header_bytes = serialized_data[8:header_end]
    payload_bytes = serialized_data[header_end:]

    header_offset = 0
    payload_offset = 0
    result_list = []

    num_sub_lists = struct.unpack('>I', header_bytes[header_offset: header_offset + 4])[0]
    header_offset += 4

    # 2. Reconstruct Tensors from the payload according to the metadata
    for _ in range(num_sub_lists):
        dtype_id, num_tensors = struct.unpack('>BI', header_bytes[header_offset: header_offset + 5])
        header_offset += 5

        new_sub_list = []
        if num_tensors > 0:
            torch_dtype = ID_TO_DTYPE[dtype_id]
            numpy_dtype = np.dtype(str(torch_dtype).replace('torch.', ''))
            item_size = numpy_dtype.itemsize

            lengths_unpack_size = num_tensors * 4
            lengths = struct.unpack(f'>{num_tensors}I',
                                    header_bytes[header_offset: header_offset + lengths_unpack_size])
            header_offset += lengths_unpack_size

            for length in lengths:
                num_bytes = length * item_size
                tensor_bytes = payload_bytes[payload_offset: payload_offset + num_bytes]

                tensor = torch.tensor(np.frombuffer(tensor_bytes, dtype=numpy_dtype))
                new_sub_list.append(tensor)
                payload_offset += num_bytes

        result_list.append(new_sub_list)

    return result_list


def print_data_dict_full(data_dict):
    """Print every tensor/list-of-tensors in a dict without truncation."""
    torch.set_printoptions(threshold=10_000_000, edgeitems=256, linewidth=500)
    for k, v in data_dict.items():
        if(k == "prompts"):
            print(f"\n== {k} ==")
            if isinstance(v, torch.Tensor):
                print(v)
            elif isinstance(v, list):
                for i, t in enumerate(v):
                    print(f"[{i}]")
                    print(t if isinstance(t, torch.Tensor) else t)
            else:
                print(v)
    torch.set_printoptions(profile="default")


def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seq length across dp ranks and micro batches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    if k_partitions > len(seqlen_list):
        raise ValueError(f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]")

    def _check_and_sort_partitions(partitions):
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        return sorted_partitions

    partitions = heapq_partition(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def heapq_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    equal_part_num = len(seqlen_list) // k_partitions

    sorted_seqlen = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)

    # Initialize the heap: each group maintains [current sum, number of elements, group index, elements in the group]
    groups = [[0, 0, i, []] for i in range(k_partitions)]
    heapq.heapify(groups)

    partitions = []
    for seqlen, i in sorted_seqlen:
        current_group = heapq.heappop(groups)
        current_group[3].append(i)
        current_group[0] += seqlen
        current_group[1] += 1
        if equal_size:
            if current_group[1] < equal_part_num:
                heapq.heappush(groups, current_group)
            else:
                partitions.append(current_group[3])
        else:
            heapq.heappush(groups, current_group)

    partitions.extend([group[3] for group in groups])

    if equal_size:
        for i, partition in enumerate(partitions):
            if len(partition) * k_partitions != len(seqlen_list):
                raise ValueError(f"Partition {i} has {len(partition)} items, expected {len(seqlen_list) // k_partitions}")
    return partitions


def prepare_prompts_experience(
        batch: Dict[str, torch.Tensor], n_samples_per_prompt, dataset_additional_keys: List[str] = None, indexes=None, add_another_batch=False,
):
    """Prepare prompts experiences to put into TQ.

    Args:
        batch: Batch datas from original dataloader.
        n_samples_per_prompt: n_samples_per_prompt
        dataset_additional_keys: The additional experience types from the dataset.
        indexes: Batch datas indexes.
    Returns: TensorDict

    """

    prompts = batch["prompts"]
    prompt_length = []
    for prompt in prompts:
        for _ in range(n_samples_per_prompt):
            prompt_length.append(torch.tensor([len(prompt)]))

    prompts_data = prompts
    prompts = []
    for prompt in prompts_data:
        for _ in range(n_samples_per_prompt):
            prompts.append(copy.deepcopy(prompt))

    add_vals = {}
    for add_keys in dataset_additional_keys:
        if add_keys in batch.keys():
            values = []
            for value in batch[add_keys]:
                for _ in range(n_samples_per_prompt):
                    values.append(value)
            add_vals[add_keys] = values
    prompt_nums = len(prompt_length)
    if add_another_batch:
        indexes = [prompt_nums + i for i in range(prompt_nums)]
    elif indexes is None:
        indexes = [i for i in range(len(prompt_length))]

    data_dict = dict(
        {"prompt_length": prompt_length, "prompts": prompts}, **add_vals
    )
    return padding_dict_to_tensor_dict(data_dict), indexes


def prepare_batch_mm(batch: Dict[str, torch.Tensor], experience_columns: List[str], num_prompts: int):
    batch_mm = {}
    for experience_column, experience_list in batch.items():
        if experience_column in experience_columns:
            if experience_column in ['video_fps', 'image', 'video']:
                temp = [torch.empty(0, dtype=torch.float32) 
                            if (isinstance(exp, torch.Tensor) and exp.numel() == 0) or (isinstance(exp, (list, tuple, dict)) and len(exp) == 0)
                            else torch.as_tensor(exp, dtype=torch.float32).squeeze() for exp in experience_list]
                batch_mm[experience_column] = [t.unsqueeze(0) if t.ndim == 0 else t for t in temp]
            elif experience_column in ['image_shape', 'image_num', 'video_shape', 'video_num', 'image_grid_thw']:
                temp = [torch.empty(0, dtype=torch.int32) 
                            if (isinstance(exp, torch.Tensor) and exp.numel() == 0) or (isinstance(exp, (list, tuple, dict)) and len(exp) == 0)
                            else torch.as_tensor(exp, dtype=torch.int32).squeeze() for exp in experience_list]
                batch_mm[experience_column] = [t.unsqueeze(0) if t.ndim == 0 else t for t in temp]
            else:
                batch_mm[experience_column] = experience_list
                if experience_column not in ['labels', 'pixel_values', 'position_ids']:
                    logger.warning(f"Detected new experience column {experience_column}, the processing logic for multimodal data needs to be adjusted accordingly.")

    indexes_multimodal = [i for i in range(num_prompts)]
    return batch_mm, indexes_multimodal


def prepare_dummy_response(indexes: List[int] = None):
    if not indexes:
        raise ValueError("No indexes provided for put_dummy_response.")

    dummy_responses = [torch.tensor([-1], dtype=torch.int32) for _ in range(len(indexes))]
    dummy_response_length = [torch.tensor([0], dtype=torch.int32) for _ in range(len(indexes))]
    dummy_dict = dict(
        {"responses": dummy_responses, "response_length": dummy_response_length}
    )

    return padding_dict_to_tensor_dict(dummy_dict)


def padding_experience_for_dp(experience_batch, indexes, experience_columns, experience_count, dp_size):
    sample_num = len(indexes)
    if sample_num < experience_count and sample_num > 0:
        min_dp_size_multiple = ((sample_num + dp_size - 1) // dp_size) * dp_size
        indexes_extend = indexes + [-2] * (min_dp_size_multiple - sample_num)
        for col in experience_columns:
            for _, _ in enumerate(indexes_extend[sample_num:]):
                experience_batch[col].append(experience_batch[col][sample_num - 1])
        indexes = indexes_extend

    return experience_batch, indexes


def process_mm_experience(data_dict: Dict[str, Union[List[str], List[torch.Tensor]]]):
    for column, experience in data_dict.items():
        if column == 'labels':
            continue
        elif column in ['image_num', 'video_num', 'video_fps']:
            data_dict[column] = torch.cat(experience, dim=0).unsqueeze(dim=1)
        elif column in ['image', 'video']:
            data_dict[column] = torch.cat(experience, dim=0).unsqueeze(dim=0)
        elif column in ['pixel_values', 'position_ids']:
            data_dict[column] = torch.cat(experience, dim=0)
        elif column in ['image_shape', 'video_shape', 'image_grid_thw']:
            data_dict[column] = torch.stack(experience, dim=0)
        else:
            raise ValueError(f"Detected new experience column {column}, the processing logic for multimodal data needs to be adjusted accordingly.")

    return data_dict


def unpack_mm_experience(batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Handles multimodal data by restoring images and pixel values from flattened tensors.

    Args:
        batch_data (Dict[str, torch.Tensor]): A dictionary containing the batch data.
        n_samples_per_prompt (int): The number of samples per prompt.

    Returns:
        Dict[str, torch.Tensor]: The processed batch data with restored images and pixel values."
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


def restore_images_from_tensors(flattened_tensors: torch.Tensor, tensor_shapes: torch.Tensor, image_num: torch.Tensor) -> List:
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


def restore_videos_from_tensors(flattened_tensors: torch.Tensor, tensor_shapes: torch.Tensor, video_num: torch.Tensor) -> List:
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


def restore_split_data(data_tensor: torch.Tensor, split_num: torch.Tensor):
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


def restore_pixel_valaues_from_flattend(flattened_tensors: torch.Tensor, tensor_shapes: torch.Tensor, image_num: torch.Tensor = None, merge_shape: bool = False) -> List[torch.Tensor]:
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


def calculate_split_indices(tensor_shapes: torch.Tensor, merge_shape: bool = False) -> Tuple[List[int], List[int]]:
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


def restore_image_grid_thw(image_grid_thw_tensor: torch.Tensor, image_num: torch.Tensor):
    res_image_grid_thw = []
    start_idx = 0
    image_num = image_num.squeeze(0)
    for i in image_num:
        res_image_grid_thw.append(image_grid_thw_tensor[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res_image_grid_thw