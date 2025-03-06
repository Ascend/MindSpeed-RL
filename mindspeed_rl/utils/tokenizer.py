# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os
import json
from typing import Any
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy

from .loggers import Loggers

os.environ['TOKENIZERS_PARALLELISM'] = "true"


def get_tokenizer(
        tokenizer_model: str,
        tokenizer_type: str = 'HuggingFaceTokenizer',
        eos_token_id: int = None,
        eos_token: str = None,
        pad_token_id: int = None,
        pad_token: str = None,
        eos_for_pad: bool = True
        ):
    """Get tokenizer.

    Args:
        tokenizer_model: A directory of HuggingFace Tokenizer
        tokenizer_type: 'HuggingFaceTokenizer' is supported only.
        eos_token_id: eos_token_id
        eos_token: eos_token
        pad_token_id: pad_token_id
        pad_token: pad_token
        eos_for_pad: if tokenizer has no pad, use eos for pad.
    """
    logger = Loggers(name='get_tokenizer')

    if tokenizer_type == 'HuggingFaceTokenizer':
        if not os.path.isdir(tokenizer_model):
            raise ValueError('tokenizer_model {} should be a directory'
                             ' for HuggingFaceTokenizer'.format(tokenizer_model))
        tokenizer = _HuggingFaceTokenizer(tokenizer_model)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(tokenizer_type))

    if pad_token_id is not None and pad_token is None:
        raise ValueError("pad_token should be set, while pad_token_id is given.")
    if pad_token_id is None and pad_token is not None:
        raise ValueError("pad_token_id should be set, while pad_token is given.")
    if eos_token_id is not None and eos_token is None:
        raise ValueError("eos_token should be set, while eos_token_id is given.")
    if eos_token_id is None and eos_token is not None:
        raise ValueError("eos_token_id should be set, while eos_token is given.")

    if tokenizer.eod_token is not None and eos_token is not None:
        raise ValueError("tokenizer has already had an eod_token.")
    if tokenizer.pad_token is not None and pad_token is not None:
        raise ValueError("tokenizer has already had a pad_token.")

    if eos_token:
        tokenizer.eod_token = eos_token
        tokenizer.eod = eos_token_id

    if tokenizer.eod_token is None or tokenizer.eod is None:
        raise ValueError("eos_token and eos_token_id are required for tokenizer.")

    if pad_token is not None:
        tokenizer.pad_token = pad_token
        tokenizer.pad = pad_token_id
    elif eos_for_pad:
        tokenizer.pad_token = tokenizer.eod_token
        tokenizer.pad = tokenizer.eod
        logger.info("eos token {} and id {} are used for"
                    " pad token and id".format(tokenizer.eod_token, tokenizer.eod))
    else:
        logger.warning("pad token and id are none.")

    return tokenizer


class BaseTokenizer(ABC):
    """Abstract class for tokenizer

    Absent a config or class-specific tracking of which objects are uniquely identifying, we must
    include all key word arguments as unique identifiers

    Args:
        tokenizer_paths (Tuple[str]): All tokenizer source paths or prefixes

        tokenizer_options (Dict[str, Any]): All tokenizer options
    """

    def __init__(self, *tokenizer_paths: str, **tokenizer_options: Any):

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["tokenizer_path"] = list(tokenizer_paths)
        for option in tokenizer_options:
            self.unique_identifiers[option] = str(tokenizer_options[option])

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)

        super().__init__()

    @abstractmethod
    def tokenize(self, text: str) -> numpy.ndarray:
        """Convert text to embedding ids

        Args:
            text (str): The text to convert

        Returns:
            numpy.ndarray: The converted embedding ids
        """
        pass

    def detokenize(self, ids: numpy.ndarray) -> str:
        """Convert embedding ids to text

        Args:
            ids (numpy.ndarray): The ids to convert

        Returns:
            str: The converted text

        Raises:
            NotImplementedError: Non-abstract, optional method
        """
        raise NotImplementedError("{} has no method 'detokenize'".format(type(self).__name__))

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token
        """
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """The vocabulary size
        """
        pass

    @property
    def cls(self):
        """The CLS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'cls'".format(type(self).__name__))

    @property
    def sep(self):
        """The SEP token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'sep'".format(type(self).__name__))

    @property
    def pad(self):
        """The PAD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'pad'".format(type(self).__name__))

    @property
    def eod(self):
        """The EOD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'eod'".format(type(self).__name__))

    @property
    def bos(self):
        """The BOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'bos'".format(type(self).__name__))

    @property
    def eos(self):
        """The EOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'eos'".format(type(self).__name__))

    @property
    def mask(self):
        """The MASK token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'mask'".format(type(self).__name__))


class _HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__(pretrained_model_name_or_path)
        try:
            import transformers
        except ImportError as e:
            raise ImportError(f"The transformers library must be"
                              f" installed to use huggingface_tokenizer_provider") from e

        # TODO(bnorick): download tokenizer once to lustre and use force offline to make sure all tasks read it from there
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self._vocab = self.tokenizer.get_vocab()
        self._inv_vocab = {token_id: token for token, token_id in self._vocab.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self._vocab

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    def tokenize(self, text):
        return self.tokenizer(text).input_ids

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @eod.setter
    def eod(self, value):
        self.tokenizer.eos_token_id = value

    @property
    def eod_token(self):
        return self.tokenizer.eos_token

    @eod_token.setter
    def eod_token(self, value):
        self.tokenizer.eos_token = value

    @property
    def pad(self):
        return self.tokenizer.pad_token_id

    @pad.setter
    def pad(self, value):
        self.tokenizer.pad_token_id = value


    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, value):
        self.tokenizer.pad_token = value
