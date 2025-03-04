# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import numpy


def get_tokenizer(tokenizer_type: str, tokenizer_model: str):
    """Build tokenizer.

    Args:
        tokenizer_type: 'HuggingFaceTokenizer' and 'Llama2Tokenizer' are supported.

        tokenizer_model: A directory of HuggingFace Tokenizer, or a path to Llama2 tokenizer.model
    """

    if tokenizer_type == 'HuggingFaceTokenizer':
        if not os.path.isdir(tokenizer_model):
            raise ValueError('tokenizer_model {} should be a directory'
                             ' for HuggingFaceTokenizer'.format(tokenizer_model))
        tokenizer = _HuggingFaceTokenizer(tokenizer_model)

    elif tokenizer_type == 'Llama2Tokenizer':
        if os.path.isdir(tokenizer_model):
            raise ValueError('tokenizer_model {} should be a path'
                             ' to Llama2Tokenizer'.format(tokenizer_model))
        tokenizer = _Llama2Tokenizer(tokenizer_model)

    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(tokenizer_type))

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
            raise ImportError(
                f"The transformers library must be installed to use huggingface_tokenizer_provider") from e

        # TODO(bnorick): download tokenizer once to lustre and use force offline to make sure all tasks read it from there
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self._vocab = self._tokenizer.get_vocab()
        self._inv_vocab = {token_id: token for token, token_id in self._vocab.items()}

    @property
    def vocab_size(self):
        return len(self._tokenizer)

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
        return self._tokenizer(text).input_ids

    def detokenize(self, token_ids):
        return self._tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self._tokenizer.eos_token_id


class _SentencePieceTokenizer(BaseTokenizer):
    """SentencePieceTokenizer wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        super().__init__(model_file, vocab_extra_ids=vocab_extra_ids)

        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _populate_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    @property
    def encoder(self):
        return self._vocab

    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]


class _Llama2Tokenizer(_SentencePieceTokenizer):
    """Llama2Tokenizer wrapper"""

    def __init__(self, model_file,):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        # BOS / EOS token IDs
        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        if not self.tokenizer.vocab_size() == self.tokenizer.get_piece_size():
            raise ValueError("tokenizer.vocab_size should be equal to tokenizer.get_piece_size")

    def tokenize(self, s: str, bos=True, eos=False):
        '''Default args for text completion, not chat/dialog.'''

        if not isinstance(s, str):
            raise TypeError("input should be a str")

        t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self.eos_id

    @property
    def additional_special_tokens_ids(self):
        return None
