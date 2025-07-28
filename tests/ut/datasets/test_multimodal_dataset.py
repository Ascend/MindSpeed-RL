from mindspeed_rl.datasets.multimodal_dataset import MultiModalDataset
from mindspeed_rl.utils.tokenizer import get_tokenizer
from mindspeed_rl.datasets.mm_utils import get_processor

from tests.test_tools.dist_test import DistributedTest


class TestMultimodalDataset(DistributedTest):
    world_size = 1
    is_dist_test = False

    def test_prompt_dataset(self):
        hf_directory = '/data/for_dt/weights/Qwen2.5-VL-3B-Instruct'
        data_path = '/data/for_dt/datasets/multimodal/multi_pic_dataset_100.parquet'
        hf_tokenizer = get_tokenizer(hf_directory)
        processor = get_processor(model_path=hf_directory, usr_fast=True)
        dataset = MultiModalDataset(
            data_path=data_path,
            tokenizer=hf_tokenizer,
            processor=processor,
            prompt_key='prompt',
            image_key='images',
            video_key='videos',
            max_prompt_length=1024,
            return_raw_chat=False,
            truncation='left'
        )
        data = dataset[0]
        assert data['prompts'][0] == 151644
