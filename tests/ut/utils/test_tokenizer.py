# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
"""Just an initialize test"""

from mindspeed_rl import get_tokenizer

from tests.test_tools.dist_test import DistributedTest


class TestTokenizer(DistributedTest):
    world_size = 1

    def test_tokenizer(self):
        tokenizer_directory = '/data/models/llama2-7b'
        tokenizer_path = '/data/models/llama2-7b/tokenizer.model'

        hf_tokenizer = get_tokenizer('HuggingFaceTokenizer', tokenizer_directory)
        llama2_tokenizer = get_tokenizer('Llama2Tokenizer', tokenizer_path)

        sentence = 'hello, this is a test case for tokenizer'

        assert hf_tokenizer.tokenize(sentence) ==\
               [1, 22172, 29892, 445, 338, 263, 1243, 1206, 363, 5993, 3950], "hf_tokenizer.tokenize Failed!"
        assert hf_tokenizer.vocab_size == 32000, "hf_tokenizer.vocab_size Failed"
        assert hf_tokenizer.detokenize([1, 22172, 29892, 445, 338, 263, 1243, 1206, 363, 5993, 3950]) ==\
               "<s> hello, this is a test case for tokenizer"


        assert llama2_tokenizer.tokenize(sentence) ==\
               [1, 22172, 29892, 445, 338, 263, 1243, 1206, 363, 5993, 3950], "hf_tokenizer.tokenize Failed!"
        assert llama2_tokenizer.vocab_size == 32000, "hf_tokenizer.vocab_size Failed"
        assert llama2_tokenizer.detokenize([1, 22172, 29892, 445, 338, 263, 1243, 1206, 363, 5993, 3950]) ==\
               "hello, this is a test case for tokenizer"
