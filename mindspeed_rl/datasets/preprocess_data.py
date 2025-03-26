# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import os
import sys

try:
    import nltk
except ImportError:
    nltk = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from mindspeed_rl.datasets.indexed_dataset import (
    IndexedDatasetBuilder,
    IndexedDataset,
    get_bin_path,
    get_idx_path,
)
from mindspeed_rl.datasets.data_handler import get_dataset_handler

from mindspeed_rl.utils.loggers import Loggers

logger = Loggers(name="preprocess_data")


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars if nltk else object):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


def build_splitter(args):
    if nltk and args.split_sentences:
        nltk.download("punkt", quiet=True)
    if args.split_sentences:
        if not nltk:
            logger.error("NLTK is not available to split sentences.")
            raise Exception("nltk is not available")
        splitter = nltk.load("tokenizers/punkt/english.pickle")
        if args.keep_newlines:
            # this prevents punkt from eating newlines after sentences
            final_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                train_text=splitter._params,
                lang_vars=CustomLanguageVars())
        else:
            final_splitter = splitter

    else:
        final_splitter = IdentitySplitter()
    return final_splitter


def cut_range_to_subs(n, gap):
    n_ = n // gap
    mod = n % gap
    if mod != 0:
        return [(k * gap, (k + 1) * gap) for k in range(0, n_)] + [(gap * n_, n)]
    else:
        return [(k * gap, (k + 1) * gap) for k in range(0, n_)]


def handle_subset(params):
    """params: [args, dataset, tokenizer, splitter]"""
    handler = get_dataset_handler(params[0], params[1], params[2], params[3])
    handler.serialize_to_disk()
    return handler.output_idx_files


def merge_datasets(args):
    prefixes = {key: set() for key in args.merge_group_keys}
    for key in prefixes:
        for basename in os.listdir(args.input):
            prefix, ext = os.path.splitext(basename)

            if prefix in prefixes[key] or key not in prefix:
                continue

            if not os.path.isfile(os.path.join(args.input, basename)):
                continue

            ext_pair = ".bin" if ext == ".idx" else ".idx"
            if not os.path.isfile(os.path.join(args.input, prefix) + ext_pair):
                raise FileNotFoundError(f"{ext_pair} file not provided for {os.path.join(args.input, prefix)}")

            prefixes[key].add(prefix)

    for key in prefixes:
        builder = None
        for prefix in sorted(prefixes[key]):
            if builder is None:
                dataset = IndexedDataset(os.path.join(args.input, prefix), multimodal=False)
                builder = IndexedDatasetBuilder(
                    get_bin_path(f'{args.output_prefix}_{key}'), dtype=dataset.index.dtype, multimodal=False
                )
                del dataset

            builder.add_index(os.path.join(args.input, prefix))

        builder.finalize(get_idx_path(f'{args.output_prefix}_{key}'))
