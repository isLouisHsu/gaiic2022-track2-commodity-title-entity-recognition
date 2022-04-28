import os
import re
import sys
import json
import yaml
import math
import jieba
import random
import argparse
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from copy import deepcopy
from itertools import chain, product
from collections import defaultdict, Counter

use_wandb = False
if os.environ.get("WANDB_ENABLE", False):
    try:
        import wandb
        use_wandb = True
        """
        - https://github.com/wandb/examples
        - https://docs.wandb.ai/guides/sweeps
        """
    except ImportError:
        use_wandb = False

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import (
    BertConfig, BertPreTrainedModel, BertModel, BertTokenizerFast, BertTokenizer,
    RobertaConfig, RobertaPreTrainedModel, RobertaModel, RobertaTokenizerFast,
    PreTrainedModel, AdamW,
)
from transformers.file_utils import ModelOutput
from nezha.modeling_nezha import NeZhaPreTrainedModel, NeZhaModel

sys.path.append("TorchBlocks/")
# from torchblocks.data.dataset import DatasetBase
from torchblocks.callback import ProgressBar, ModelCheckpoint
from torchblocks.data.process_base import ProcessBase
from torchblocks.metrics.sequence_labeling.scheme import get_scheme
from torchblocks.metrics.sequence_labeling.precision_recall_fscore import _precision_recall_fscore_support
from torchblocks.metrics.sequence_labeling.seqTag_score import SequenceLabelingScore
from torchblocks.layers.layer_norm import ConditionalLayerNorm
from torchblocks.losses.focal_loss import FocalLoss
from torchblocks.losses.label_smoothing import LabelSmoothingCE
from torchblocks.core import TrainerBase
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.utils.device import prepare_device
from torchblocks.utils.paths import check_dir, load_pickle, check_file, is_file
from torchblocks.utils.paths import find_all_checkpoints
from torchblocks.utils.seed import seed_everything
from tokenization_bert_zh import BertTokenizerZh
from utils import get_spans_bio, check_example, get_synonym
from run_chinese_ref import is_chinese

IGNORE_INDEX = -100
Span = NewType("Span", Tuple[int, int, str])
Entity = NewType("Entity", List[Span])

import logging
logger = logging.getLogger(__name__)

def default_data_collator(features: List[Dict[str, torch.Tensor]],
                          dynamic_batch=False,
                          dynamic_keys=[]) -> Dict[
    str, Any]:
    batch = {}
    first = features[0]
    max_input_length = first['input_ids'].size(0)
    if dynamic_batch:
        max_input_length = max([torch.sum(f["attention_mask"]) for f in features])
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            bv = torch.stack([f[k] for f in features]) if isinstance(v, torch.Tensor) else torch.tensor(
                [f[k] for f in features])
            batch[k] = bv
    if dynamic_batch:
        for k in dynamic_keys:
            if k not in batch: continue
            if batch[k].dim() >= 2: batch[k] = batch[k][:, : max_input_length]
    return batch


class DatasetBase(torch.utils.data.Dataset):
    keys_to_truncate_on_dynamic_batch = ['input_ids', 'attention_mask', 'token_type_ids']

    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines: List[Callable],
                 max_examples: int = None,
                 use_cache: bool = False,
                 collate_dynamic: bool = True,
                 cached_features_file: str = None,
                 overwrite_cache: bool = False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        if not is_file(data_name):
            file_path = os.path.join(data_dir, data_name)
            check_file(file_path)
        self.examples = self.create_examples(self.read_data(file_path), data_type)
        if max_examples is not None: self.examples = self.examples[: max_examples]
        self.process_piplines = process_piplines if isinstance(process_piplines, list) else [process_piplines]
        self.num_examples = len(self.examples)
        self.num_labels = len(self.get_labels())
        self.use_cache = use_cache
        self.collate_dynamic = collate_dynamic
        self.cached_features_file = cached_features_file
        if self.cached_features_file is None: self.cached_features_file = ""
        self.overwrite_cache = overwrite_cache
        if self.use_cache:
            self.create_cache()

    def create_cache(self):
        self.cached_features_file = os.path.join(self.data_dir, self.cached_features_file)
        if is_file(self.cached_features_file) and not self.overwrite_cache:
            logger.info("Loading features from cached file %s", self.cached_features_file)
            self.features = torch.load(self.cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {self.data_dir}")
            self.features = [
                self.process_example(example) for example in
                tqdm(self.examples, total=self.num_examples, desc="Converting examples to features...")]
            # FIXED: is_file必须在文件存在情况下才返回True
            # if is_file(self.cached_features_file):
            if not os.path.isdir(self.cached_features_file):
                logger.info("Saving features to cached file %s", self.cached_features_file)
                torch.save(self.features, self.cached_features_file)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.use_cache:
            feature = self.features[index]
        else:
            feature = self.process_example(self.examples[index])
        return feature

    def __len__(self):
        return self.num_examples

    @classmethod
    def get_labels(self) -> List[str]:
        raise NotImplementedError('Method [get_labels] should be implemented.')

    @classmethod
    def label2id(cls):
        return {label: i for i, label in enumerate(cls.get_labels())}

    @classmethod
    def id2label(cls):
        return {i: label for i, label in enumerate(cls.get_labels())}

    def read_data(self, input_file: str) -> Any:
        raise NotImplementedError('Method [read_data] should be implemented.')

    def create_examples(self, data: Any, set_type: str, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError('Method [create_examples] should be implemented.')

    def process_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        for proc in self.process_piplines:
            if proc is None: continue
            example = proc(example)
        return example

    def collate_fn(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return default_data_collator(features,
                                     dynamic_batch=self.collate_dynamic,
                                     dynamic_keys=self.keys_to_truncate_on_dynamic_batch)

class SpanClassificationDataset(DatasetBase):
    """
    Attributes
    ----------
    
        examples: 
        !!! WARN: word-level

    """
    keys_to_truncate_on_dynamic_batch = [
        "input_ids", "attention_mask", "token_type_ids", "conditional_ids", "syntactic_upos_ids",
    ]

    def __init__(
        self,
        data_name,
        data_dir,
        data_type,
        process_piplines: List[Callable],
        context_size: int = 0,
        max_examples: int = None,
        labels: List[str] = None,
        use_cache: bool = False,
        collate_dynamic: bool = True,
        cached_features_file: str = None,
        overwrite_cache: bool = False
    ) -> None:
        
        self.labels = labels
        self.context_size = max(0, context_size)
        super().__init__(data_name, data_dir, data_type, process_piplines, max_examples,
            use_cache,collate_dynamic, cached_features_file, overwrite_cache)

        # add context
        if self.context_size > 0:
            examples = []
            for i, example in tqdm(enumerate(self.examples), total=len(self.examples), 
                                    desc=f"Adding Context({self.context_size})..."):
                examples.append(self.set_example_context(example, i))
            self.examples = examples

    def set_example_type(self, example):
        is_overlap = False
        is_discontinuous = False

        if example["entities"] is not None: 

            entities = []
            for entity in example["entities"]:
                entities.append(set())
                if len(entity) > 1:
                    is_discontinuous = True
                for start, end, _, _ in entity:
                    for pos in range(start, end):
                        entities[-1].add(pos)

            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    entity_a, entity_b = entities[i], entities[j]
                    if len(entity_a.intersection(entity_b)) > 0:
                        is_overlap = True

        example["is_overlap"] = is_overlap
        example["is_discontinuous"] = is_discontinuous
        return example
    
    def set_example_context(self, example, example_no):
        example = deepcopy(example)
        # left context
        left_context_size = 0
        context_no = example_no
        while True:
            context_no -= 1
            if context_no < 0:
                break

            # context_text = self.examples[context_no]["text"]
            # if left_context_size + len(context_text) > self.context_size:   # 不足整句
            #     break
            # left_context_size += len(context_text)
            # example["text"] = context_text + example["text"]
            
            context_size = max(0, self.context_size - left_context_size)
            context_text = self.examples[context_no]["text"][- context_size: ]
            left_context_size += len(context_text)
            example["text"] = context_text + example["text"]
            if left_context_size >= self.context_size:
                break

        # right context
        right_context_size = 0
        context_no = example_no
        while True:
            context_no += 1
            if context_no >= len(self.examples):
                break

            # context_text = self.examples[context_no]["text"]
            # if right_context_size + len(context_text) > self.context_size:  # 不足整句
            #     break
            # right_context_size += len(context_text)
            # example["text"] = example["text"] + context_text

            context_size = max(0, self.context_size - right_context_size)
            context_text = self.examples[context_no]["text"][: context_size]
            right_context_size += len(context_text)
            example["text"] = example["text"] + context_text
            if right_context_size >= self.context_size:
                break

        example["sent_start"] = left_context_size
        example["sent_end"] = len(example["text"]) - right_context_size
        if example["entities"] is not None:
            for entity in example["entities"]:
                for i, (start, end, label, string) in enumerate(entity):
                    entity[i] = (start + left_context_size, end + left_context_size, label, string)
                
        return example

    def collate_fn(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {}
        first = features[0]
        max_input_length = None
        if self.collate_dynamic:
            max_input_length = max([torch.sum(f["attention_mask"]) for f in features])
        for k in first.keys():
            bv = None
            if k in ["input_ids", "attention_mask", "token_type_ids", "conditional_ids", "syntactic_upos_ids"]:
                bv = torch.stack([f[k] for f in features], dim=0)                   # (batch_size, sequence_length)
            elif k in ["spans", "spans_mask", "labels"]:
                if first[k] is not None:
                    bv = pad_sequence([f[k] for f in features], batch_first=True)   # (batch_size, num_spans, *)
            elif k in []:
                bv = torch.stack([f[k] for f in features])                          # (batch_size,)
            elif k in []:
                bv = [f[k] for f in features]                                       # (batch_size,)
            else:
                continue
            batch[k] = bv
        if self.collate_dynamic:
            for k in self.keys_to_truncate_on_dynamic_batch:
                if k in batch:
                    if batch[k].dim() >= 2: batch[k] = batch[k][:, : max_input_length]
        if batch["labels"] is None: batch.pop("labels")
        return batch


# class GaiicTrack2SpanClassificationDataset(SpanClassificationDataset):

#     @classmethod
#     def get_labels(cls) -> List[str]:
#         return ["O",] + [
#             # str(i) for i in range(55) if i not in [0, 27, 45]
#             str(i) for i in range(55)   # TODO:
#         ]

#     def _generate_examples(self, data_path):
#         sentence_counter = 0
#         with open(data_path, encoding="utf-8") as f:
#             lines = f.readlines()
        
#         current_words = []
#         current_labels = []
#         for row in lines:
#             row = row.rstrip("\n")
#             if row != "":
#                 token, label = row[0], row[2:]
#                 current_words.append(token)
#                 current_labels.append(label)
#             else:
#                 if not current_words:
#                     continue
#                 assert len(current_words) == len(current_labels), "word len doesn't match label length"
#                 sentence = (
#                     sentence_counter,
#                     {
#                         "id": str(sentence_counter),
#                         "tokens": current_words,
#                         "ner_tags": current_labels,
#                     },
#                 )
#                 sentence_counter += 1
#                 current_words = []
#                 current_labels = []
#                 yield sentence

#         # if something remains:
#         if current_words:
#             sentence = (
#                 sentence_counter,
#                 {
#                     "id": str(sentence_counter),
#                     "tokens": current_words,
#                     "ner_tags": current_labels,
#                 },
#             )
#             yield sentence

#     def read_data(self, input_file: str) -> Any:
#         return list(self._generate_examples(input_file))

#     def create_examples(self, data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
#         examples = []
#         # TODO:
#         if data_type == "train":
#             data = data[:400]
#         else:
#             data = data[400:]
#         get_entities = get_scheme("BIO")
#         for (i, line) in enumerate(data):
#             guid = f"{data_type}-{i}"
#             tokens = line[1]["tokens"]
#             entities = None
#             if data_type != "test":
#                 entities = []
#                 for label, start, end in get_entities(line[1]["ner_tags"]):
#                     entities.append((start, end + 1, label, tokens[start: end + 1]))
#             examples.append(dict(guid=guid, text=tokens, entities=entities, sent_start=0, sent_end=len(tokens)))
#         return examples


class GaiicTrack2SpanClassificationDataset(SpanClassificationDataset):

    @classmethod
    def get_labels(cls) -> List[str]:
        return ["O",] + [
            str(i +1) for i in range(54)
        ]
        # return ["O",] + [
        #     str(i) for i in range(55) if i not in [0, 27, 45]
        # ]
    
    @classmethod
    def get_xlabels(cls) -> List[str]:
        return ["O",] + [
            str(i + 1) for i in range(3)
        ]

    @classmethod
    def xlabel2id(cls):
        return {label: i for i, label in enumerate(cls.get_xlabels())}

    @classmethod
    def xid2label(cls):
        return {i: label for i, label in enumerate(cls.get_xlabels())}
    
    @classmethod
    def get_ylabels(cls) -> List[str]:
        return ["O",] + [
            str(i + 1) for i in range(18)
        ]

    @classmethod
    def ylabel2id(cls):
        return {label: i for i, label in enumerate(cls.get_ylabels())}

    @classmethod
    def yid2label(cls):
        return {i: label for i, label in enumerate(cls.get_ylabels())}

    @classmethod
    def fx(cls, label):
        if label == "O": return "O"
        return str((int(label) - 1) // 18 + 1)

    @classmethod
    def fy(cls, label):
        if label == "O": return "O"
        return str((int(label) - 1) %  18 + 1)

    @classmethod
    def flabel(cls, x, y):
        if x == "O" or y == "O": return "O"
        return str(int(x) * int(y))

    def read_data(self, input_file: str) -> Any:
        with open(input_file, encoding="utf-8") as f:
            examples = [json.loads(line) for line in f.readlines()]
        return examples

    def create_examples(self, data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{i}"
            tokens = line["text"]
            entities = None
            if data_type != "test":
                entities = line["entities"]
            examples.append(dict(guid=guid, text=tokens, entities=entities, sent_start=0, sent_end=len(tokens)))
        return examples
    
    def process_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        for proc in self.process_piplines:
            if proc is None: continue
            if isinstance(proc, AugmentBaseDual):
                example_b = self.select_another_example_randomly(self.examples, example)
                example = proc(example, example_b)
            else:
                example = proc(example)
        return example

    def select_another_example_randomly(self, examples, example):
        another = np.random.choice(examples)
        while another["guid"] == example["guid"]:
            another = np.random.choice(examples)
        return another

class LevelConvertorBase:

    do_convert_check = False

    def _convert(self, raw: str) -> Tuple[List[str], List[List[int]]]:
        raise NotImplementedError
    
    def _forward(self, tokens: List[str]) -> str:
        raise NotImplementedError

    def _backward(self, raw: str) -> List[str]:
        return self._convert(raw)[0]

    def forward(self, raw, raw_level_entities):
        tokens, offset_mapping = self._convert(raw)
        raw2token_map = dict()
        for i, (raw_start, raw_end) in enumerate(offset_mapping):
            for j in range(raw_start, raw_end):
                raw2token_map[j] = i
        
        if raw_level_entities is not None:
            token_level_entities = []
            for raw_level_entity in raw_level_entities:
                token_level_entity= []
                for raw_level_start, raw_level_end, label, span_raws in raw_level_entity:

                    try:
                        token_level_start = raw2token_map[raw_level_start]
                        token_level_end = raw2token_map[raw_level_end - 1] + 1
                    except KeyError as e:
                        import pdb; pdb.set_trace()
                    
                    span_tokens = tokens[token_level_start: token_level_end]
                    token_level_string = self._forward(span_tokens)
                    raw_level_string = raw[raw_level_start: raw_level_end]
                    try:
                        assert token_level_string == raw_level_string
                    except AssertionError as e:
                        if self.do_convert_check:
                            print(e)
                            import pdb; pdb.set_trace()

                    token_level_entity.append((token_level_start, token_level_end, label, span_tokens))
                token_level_entities.append(token_level_entity)
        else:
            token_level_entities = None

        return tokens, token_level_entities

    def backward(self, tokens, token_level_entities):
        chars = self._forward(tokens)
        tokens, offset_mapping = self._convert(chars)
        
        if token_level_entities is not None:
            char_level_entities = []
            token2char_map = dict(enumerate(offset_mapping))
            for token_level_entity in token_level_entities:
                char_level_entity = []
                for token_level_start, token_level_end, label, span_tokens in token_level_entity:
                    char_level_start, char_level_end = float('inf'), float('-inf')
                    for i in range(token_level_start, token_level_end):
                        start, end = token2char_map[i]
                        char_level_start = min(char_level_start, start)
                        char_level_end   = max(char_level_end,   end  )
                    
                    span_tokens = tokens[token_level_start: token_level_end]
                    token_level_string = self._forward(span_tokens)
                    char_level_string = chars[char_level_start: char_level_end]

                    try:
                        assert token_level_string == char_level_string
                    except AssertionError as e:
                        if self.do_convert_check:
                            print(e)
                            import pdb; pdb.set_trace()

                    char_level_entity.append((char_level_start, char_level_end, label, char_level_string))
                char_level_entities.append(char_level_entity)
        else:
            char_level_entities = None

        return chars, char_level_entities


class LevelConvertorWhitespace(LevelConvertorBase):
    """
    Example:
        >>> text = "Unified Named Entity  Recognition as Word-Word Relation Classification"
        >>> entities = [
        >>>     [(8, 8 + len("Named Entity  Recognition"), "x", "Named Entity  Recognition")],
        >>> ]
        >>> convertor = LevelConvertorWhitespace()
        >>> tokens2, entities2 = convertor.forward(text, entities)
        >>> chars3, entities3 = convertor.backward(tokens2, entities2)
    """
    def _convert(self, text):
        segments = re.split(r"( )", text)
        words = []; offset_mapping = []
        word_idx = 0; start = 0
        for seg in segments:
            if seg == " ":
                start += 1
                continue
            words.append(seg)
            word_idx += 1
            offset_mapping.append((start, start + len(seg)))
            start += len(seg)
        return words, offset_mapping
    
    def _forward(self, words):
        return " ".join(words)


class LevelConvertorHuggingFace(LevelConvertorBase):
    """
    Example:
        >>> text = "Unified Named Entity Recognition as Word-Word Relation Classification"
        >>> entities = [
        >>>     [(8, 8 + len("Named Entity Recognition"), "x", "Named Entity Recognition")],
        >>> ]
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        >>> convertor = LevelConvertorHuggingFace(tokenizer)
        >>> tokens2, entities2 = convertor.forward(text, entities)
        >>> chars3, entities3 = convertor.backward(tokens2, entities2)
    """

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def _convert(self, text):
        inputs = self.tokenizer(text, return_offsets_mapping=True, return_tensors="np")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        offset_mapping = inputs["offset_mapping"][0].tolist()[1:-1]     # [CLS], [SEP]
        return tokens, offset_mapping
    
    def _forward(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens).strip()


class LevelConvertorHuggingFaceZh(LevelConvertorHuggingFace):

    def _convert(self, text):
        num_chars = len(text)
        if getattr(self.tokenizer, "do_ref_tokenize"):
            text = "".join(text)
            words = []
            for word in jieba.cut(text):
                if is_chinese(word):
                    words.append(word)
                else:
                    words.extend(list(word))
            text = words
        inputs = self.tokenizer(
            text,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_tensors="np"
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        assert len(tokens) == num_chars
        offset_mapping = inputs["offset_mapping"][0].tolist()[1:-1]     # [CLS], [SEP]
        return tokens, offset_mapping
   
    def _forward(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)


class ProcessConvertLevel(ProcessBase):

    def __init__(self, tokenizer, conversion, lang="zh"):
        self.tokenizer = tokenizer
        self.conversion = conversion

        self.whitespace_converter = LevelConvertorWhitespace()
        if lang == "en":
            self.huggingface_converter = LevelConvertorHuggingFace(tokenizer)
        elif lang == "zh":
            self.huggingface_converter = LevelConvertorHuggingFaceZh(tokenizer)

        self.conversion2func_map = {
            "word2char": self.word_to_char,
            "char2word": self.char_to_word,
            "char2token": self.char_to_token,
            "token2char": self.token_to_char,
            "word2token": self.word_to_token,
            "token2word": self.token_to_word,
        }
        self.convert_func = self.conversion2func_map[self.conversion]
    
    def __call__(self, example):
        example = deepcopy(example)
        is_flat = False
        if example["entities"] and not isinstance(example["entities"][0][0], Iterable):
            is_flat = True
        if is_flat:
            example["entities"] = [[entity] for entity in example["entities"]]

        converted, entities = self.convert_func(
            example["text"], example["entities"])
        if "sent_start" in example and "sent_end" in example:
            sent_start, sent_end = example["sent_start"], example["sent_end"]
            sentence = [[(sent_start, sent_end, "_", example["text"][sent_start: sent_end])]]
            _, sentence = self.convert_func(example["text"], sentence)
            example["sent_start"], example["sent_end"] = sentence[0][0][:2]

        if is_flat:
            entities = [entity[0] for entity in entities]
            
        example["text"] = converted
        example["entities"] = entities
        return example
    
    def char_to_word(self, chars, char_level_entities):
        return self.whitespace_converter.forward(chars, char_level_entities)

    def word_to_char(self, words, word_level_entities):
        return self.whitespace_converter.backward(words, word_level_entities)
        
    def char_to_token(self, chars, char_level_entities):
        return self.huggingface_converter.forward(chars, char_level_entities)
    
    def token_to_char(self, tokens, token_level_entities):
        return self.huggingface_converter.backward(tokens, token_level_entities)
    
    def word_to_token(self, words, word_level_entities):
        return self.char_to_token(*self.word_to_char(words, word_level_entities))
    
    def token_to_word(self, tokens, token_level_entities):
        return self.char_to_word(*self.token_to_char(tokens, token_level_entities))

FH_SPACE = FHS = ((u"　", u" "),)
FH_NUM = FHN = (
    (u"０", u"0"), (u"１", u"1"), (u"２", u"2"), (u"３", u"3"), (u"４", u"4"),
    (u"５", u"5"), (u"６", u"6"), (u"７", u"7"), (u"８", u"8"), (u"９", u"9"),
)
FH_ALPHA = FHA = (
    (u"ａ", u"a"), (u"ｂ", u"b"), (u"ｃ", u"c"), (u"ｄ", u"d"), (u"ｅ", u"e"),
    (u"ｆ", u"f"), (u"ｇ", u"g"), (u"ｈ", u"h"), (u"ｉ", u"i"), (u"ｊ", u"j"),
    (u"ｋ", u"k"), (u"ｌ", u"l"), (u"ｍ", u"m"), (u"ｎ", u"n"), (u"ｏ", u"o"),
    (u"ｐ", u"p"), (u"ｑ", u"q"), (u"ｒ", u"r"), (u"ｓ", u"s"), (u"ｔ", u"t"),
    (u"ｕ", u"u"), (u"ｖ", u"v"), (u"ｗ", u"w"), (u"ｘ", u"x"), (u"ｙ", u"y"), (u"ｚ", u"z"),
    (u"Ａ", u"A"), (u"Ｂ", u"B"), (u"Ｃ", u"C"), (u"Ｄ", u"D"), (u"Ｅ", u"E"),
    (u"Ｆ", u"F"), (u"Ｇ", u"G"), (u"Ｈ", u"H"), (u"Ｉ", u"I"), (u"Ｊ", u"J"),
    (u"Ｋ", u"K"), (u"Ｌ", u"L"), (u"Ｍ", u"M"), (u"Ｎ", u"N"), (u"Ｏ", u"O"),
    (u"Ｐ", u"P"), (u"Ｑ", u"Q"), (u"Ｒ", u"R"), (u"Ｓ", u"S"), (u"Ｔ", u"T"),
    (u"Ｕ", u"U"), (u"Ｖ", u"V"), (u"Ｗ", u"W"), (u"Ｘ", u"X"), (u"Ｙ", u"Y"), (u"Ｚ", u"Z"),
)
FH_PUNCTUATION = FHP = (
    (u"．", u"."), (u"，", u","), (u"！", u"!"), (u"？", u"?"), (u"”", u'"'),
    (u"’", u"'"), (u"‘", u"`"), (u"＠", u"@"), (u"＿", u"_"), (u"：", u":"),
    (u"；", u";"), (u"＃", u"#"), (u"＄", u"$"), (u"％", u"%"), (u"＆", u"&"),
    (u"（", u"("), (u"）", u")"), (u"‐", u"-"), (u"＝", u"="), (u"＊", u"*"),
    (u"＋", u"+"), (u"－", u"-"), (u"／", u"/"), (u"＜", u"<"), (u"＞", u">"),
    (u"［", u"["), (u"￥", u"\\"), (u"］", u"]"), (u"＾", u"^"), (u"｛", u"{"),
    (u"｜", u"|"), (u"｝", u"}"), (u"～", u"~"),
)
FH_ASCII = HAC = lambda: ((fr, to) for m in (FH_ALPHA, FH_NUM, FH_PUNCTUATION) for fr, to in m)
HF_SPACE = HFS = ((u" ", u"　"),)
HF_NUM = HFN = lambda: ((h, z) for z, h in FH_NUM)
HF_ALPHA = HFA = lambda: ((h, z) for z, h in FH_ALPHA)
HF_PUNCTUATION = HFP = lambda: ((h, z) for z, h in FH_PUNCTUATION)
HF_ASCII = ZAC = lambda: ((h, z) for z, h in FH_ASCII())

class ProcessPreprocess(ProcessBase):

    def convert(self, text, *maps, **ops):
        """ 全角/半角转换
        args:
            text: unicode string need to convert
            maps: conversion maps
            skip: skip out of character. In a tuple or string
            return: converted unicode string
        """
        if "skip" in ops:
            skip = ops["skip"]
            if isinstance(skip, str):
                skip = tuple(skip)
            def replace(text, fr, to):
                return text if fr in skip else text.replace(fr, to)
        else:
            def replace(text, fr, to):
                return text.replace(fr, to)
        for m in maps:
            if callable(m):
                m = m()
            elif isinstance(m, dict):
                m = m.items()
            for fr, to in m:
                text = replace(text, fr, to)
        return text

    def __call__(self, example):
        example = deepcopy(example)
        full_half_convert = lambda x: self.convert(
            x, 
            FH_SPACE, 
            FH_NUM, 
            FH_ALPHA, 
            FH_PUNCTUATION
        )
        for i, ch in enumerate(example["text"]):
            example["text"][i] = full_half_convert(ch)
        return example


class ProcessMergeDiscontinuousSpans(ProcessBase):

    def __call__(self, example):
        if example["entities"] is None:
            return example

        example = deepcopy(example)
        entities = []
        for entity in example["entities"]:
            if len(entity) == 1:
                entities.append(entity)
            else:
                entities.append([])
                entity = sorted(entity, key=lambda x: x[:2])
                for span in entity:
                    start, end, label, string = span
                    if len(entities[-1]) == 0: 
                        entities[-1].append(span)
                        continue
                    last_start, last_end, last_label, last_string = entities[-1][-1]
                    if label != last_label or example["text"][last_end: start].strip() != "":
                        entities[-1].append(span)
                        continue
                    last_entity = entities[-1].pop(-1)
                    new_string = example["text"][last_start: end]
                    try:
                        assert new_string == last_string + " " + string
                    except AssertionError:
                        import pdb; pdb.set_trace()
                    merged_span = (last_start, end, label, new_string)
                    entities[-1].append(merged_span)
                    # print(f"\nMerge {last_entity}, {span} to {merged_span}")
        example["entities"] = entities
        return example

class AugmentBase(ProcessBase):
    
    def __call__(self, example, inverse=False):
        if not inverse:
            example = self.process(example)
        else:
            example = self.process_inv(example)
        return example
    
    def process(self, example):
        raise NotImplementedError('Method [process] should be implemented.')
    
    def process_inv(self, example):
        raise NotImplementedError('Method [process_inv] should be implemented.')

class AugmentBaseDual(AugmentBase):
    
    def __call__(self, example_a, example_b=None, inverse=False):
        if not inverse:
            example = self.process(example_a, example_b)
        else:
            example = self.process_inv(example_a, example_b)
        return example
    
    def process(self, example_a, example_b=None):
        raise NotImplementedError('Method [process] should be implemented.')
    
    def process_inv(self, example_a, example_b=None):
        raise NotImplementedError('Method [process_inv] should be implemented.')

class ProcessConcateExamplesRandomly(AugmentBaseDual):
    """ 随机选择一个其他样本拼接，中间用[SEP]分隔 """
    
    def __init__(self, sep_token="[SEP]", p=0.5):
        self.sep_token = sep_token
        self.p = p
    
    def process(self, example_a, example_b):
        example_a = deepcopy(example_a)
        example_b = deepcopy(example_b)
        is_replaced = False
        if np.random.random() < self.p:
            is_replaced = True
            example_a, example_b = example_b, example_a
        
        guid = f"{example_a['guid']}|{example_b['guid']}"
        text_a, text_b = example_a["text"], example_b["text"]
        text = text_a + [self.sep_token] + text_b
        
        sep_id = len(text_a)
        entities_a, entities_b = example_a["entities"], example_b["entities"]
        entities = entities_a + [
            [start + sep_id + 1, end + sep_id + 1, label, string]
            for start, end, label, string in entities_b
        ]

        example = dict(guid=guid, text=text, entities=entities, sent_start=0, sent_end=len(text))
        if "status" not in example:
            example["status"] = dict()
        example["status"][__class__] = dict(
            is_replaced=is_replaced, 
            sep_id=sep_id
        )

        self.process_inv(example)

        return example
    
    def process_inv(self, example, dummpy=None):
        if __class__ not in example["status"]:
            return example
        example = deepcopy(example)
        status = example["status"].pop(__class__)
        is_replaced = status["is_replaced"]
        sep_id = status["sep_id"]

        guid_a, guid_b = example["guid"].split("|")
        text_a, text_b = example["text"][: sep_id], example["text"][sep_id + 1:]
        entities_a, entities_b = [], []
        for entity in example["entities"]:
            start, end, label, string = entity
            if start < sep_id:
                entities_a.append(entity)
            else:
                start -= (sep_id + 1)
                end   -= (sep_id + 1)
                entities_b.append([start, end, label, string])
        
        example_a = dict(guid=guid_a, text=text_a, entities=entities_a, sent_start=0, sent_end=len(text_a))
        example_b = dict(guid=guid_b, text=text_b, entities=entities_b, sent_start=0, sent_end=len(text_b))

        if is_replaced:
            example_a, example_b = example_b, example_a
        
        return example_a, example_b

class AugmentDropRandomEntity(AugmentBase):
    """ 随机选择实体丢弃 """

    def __init__(self, p=0.1):
        self.p = p

    def process(self, example):
        example = deepcopy(example)
        text = example["text"]
        entities = example["entities"]
        num_entities = len(entities)

        num_drop = int(np.round(num_entities * self.p))
        if num_drop == 0: return example
        dropped_indices = np.random.choice(
            num_entities, size=num_drop, replace=False)

        kept = []; dropped = []
        flag = np.array([0] * len(text))
        offset = np.array([0] * len(text))
        for index in range(num_entities):
            entity = entities[index]
            if index in dropped_indices:
                start, end, label, string = entity
                offset[start: ] -= (end - start)
                flag[start: end] = 1
                dropped.append(entity)
            else:
                kept.append(entity)
        
        text = [t for t, f in zip(text, flag) if f == 0]
        for i, (start, end, label, string) in enumerate(kept):
            start += offset[start]
            end   += offset[end-1]
            kept[i] = [start, end, label, string]
        
        example["text"] = text
        example["sent_start"] = 0
        example["sent_end"] = len(text)
        example["entities"] = kept
        check_example(example)

        if "status" not in example:
            example["status"] = dict()
        example["status"][__class__] = dict(
            flag=flag, 
            offset=offset,
            dropped=dropped,
        )

        return example

    def process_inv(self, example):
        if __class__ not in example["status"]:
            return example
        example = deepcopy(example)
        status = example["status"].pop(__class__)
        flag = status["flag"]
        offset = status["offset"]
        dropped = status["dropped"]

        ptr = 0
        text = [None] * len(flag)
        for i, f in enumerate(flag):
            if f == 0:
                text[i] = example["text"][ptr]
                ptr += 1
        for start, end, label, string in dropped:
            text[start: end] = string
        assert all([ch is not None for ch in text])

        entities = []
        for i, (start, end, label, string) in enumerate(example["entities"]):
            start -= offset[start]
            end   -= offset[end-1]
            entities.append([start, end, label, string])
        entities.extend(dropped)
        entities = sorted(entities, key=lambda x: x[:2])

        example["text"] = text
        example["sent_start"] = 0
        example["sent_end"] = len(text)
        example["entities"] = entities
        check_example(example)

        return example

class AugmentMaskRandomEntity(AugmentBase):
    """ 随机选择实体遮盖 """

    def __init__(self, mask_token="[MASK]", p=0.1):
        self.mask_token = mask_token
        self.p = p

    def process(self, example):
        example = deepcopy(example)
        text = example["text"]
        entities = example["entities"]
        num_entities = len(entities)

        num_mask = int(np.round(num_entities * self.p))
        if num_mask == 0: return example
        masked_indices = np.random.choice(
            num_entities, size=num_mask, replace=False)
        
        kept = []; masked = []
        for index in range(num_entities):
            entity = entities[index]
            if index in masked_indices:
                start, end, label, string = entity
                text[start: end] = [self.mask_token] * (end - start)
                masked.append(entity)
            else:
                kept.append(entity)

        example["text"] = text
        example["entities"] = kept
        check_example(example)

        if "status" not in example:
            example["status"] = dict()
        example["status"][__class__] = dict(
            masked=masked, 
        )

        return example

    def process_inv(self, example):
        if __class__ not in example["status"]:
            return example
        example = deepcopy(example)
        status = example["status"].pop(__class__)
        masked = status["masked"]

        for entity in masked:
            start, end, label, string = entity
            example["text"][start: end] = string
            example["entities"].append(entity)
        assert all([ch != self.mask_token for ch in example["text"]])
        check_example(example)

        return example

class AugmentRandomMask(AugmentBase):
    """ 后处理 """

    def __init__(self, p=0.1):
        self.p = p

    def process(self, example):
        example = deepcopy(example)

        # TODO:
        check_example(example)
        self.process_inv(example)

        return example

    def process_inv(self, example):
        if __class__ not in example["status"]:
            return example
        example = deepcopy(example)
        check_example(example)
        # TODO:
        return example

class AugmentSynonymReplace(AugmentBase):
    """ 同义词替换 
    
    TODO: 替换实体存在嵌套
    """
    def __init__(self, p=0.1, augment_labels=None):
        self.p = p
        self.augment_labels = augment_labels

    def process(self, example):
        example = deepcopy(example)
        text = example["text"]
        entities = example["entities"]

        idx2entity_map = self._find_synonym(entities)
        num_entities = len(idx2entity_map)
        num_replace = int(np.round(num_entities * self.p))
        if num_replace == 0: return example
        replaced_indices = np.random.choice(
            list(idx2entity_map.keys()), size=num_replace, replace=False)

        idx2entity_map = {k: v for k, v in idx2entity_map.items() if k in replaced_indices}
        text, entities, idx2entity_map = self._replace_entities(
            text, entities, idx2entity_map)

        example["text"] = text
        example["entities"] = entities
        example["sent_start"] = 0
        example["sent_end"] = len(text)
        check_example(example)

        if "status" not in example:
            example["status"] = dict()
        example["status"][__class__] = dict(
            idx2entity_map=idx2entity_map,
        )

        return example

    def process_inv(self, example):
        if __class__ not in example["status"]:
            return example
        example = deepcopy(example)
        status = example["status"].pop(__class__)

        text, entities = example["text"], example["entities"]
        idx2entity_map = status["idx2entity_map"]
        text, entities, idx2entity_map = self._replace_entities(
            text, entities, idx2entity_map)

        example["text"] = text
        example["entities"] = entities
        example["sent_start"] = 0
        example["sent_end"] = len(text)

        check_example(example)
        return example

    def _find_synonym(self, entities):
        idx2entity_map = dict()
        for idx, (start, end, label, string) in enumerate(entities):
            if label not in self.augment_labels:
                continue
            word = "".join(string)
            synonyms = get_synonym(word, None)
            if synonyms is None:
                continue
            synonym = synonyms[np.random.choice(len(synonyms))][0]
            entity_new = [start, start + len(synonym), label, list(synonym)]
            idx2entity_map[idx] = entity_new
        return idx2entity_map

    def _replace_entities(self, text, entities, idx2entity_map):
        for idx, entity_new in sorted(idx2entity_map.items(), key=lambda x: x[0], reverse=True):
            start, end, label, string = entities[idx]
            start_new, end_new, label_new, string_new = entity_new
            text = text[: start] + string_new + text[end: ]
            entities[idx] = entity_new

            offset = len(string_new) - len(string)
            for entity in entities[idx + 1: ]:
                entity[0] += offset
                entity[1] += offset

            idx2entity_map[idx] = (start, end, label, string)

        return text, entities, idx2entity_map

class ProcessExample2Feature(ProcessBase):

    def __init__(self, label2id, tokenizer, max_sequence_length, 
            max_span_length, negative_sampling, stanza_nlp=None):
        super().__init__()
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_span_length = max_span_length
        self.negative_sampling = negative_sampling
        self.converter = ProcessConvertLevel(tokenizer, "char2token")
        self.stanza_nlp = stanza_nlp
        if stanza_nlp is not None:
            self.stanza_upos_unit2id = self.stanza_nlp.processors['pos'].vocab._vocabs['upos']._unit2id
            
    def _encode_text(self, text: str):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_sequence_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

    def _encode_spans(self, input_length, sent_start, sent_end, skip_indices=None):
        spans = []; spans_mask = []
        sent_start = min(sent_start, self.max_sequence_length - 2)
        sent_end = min(sent_end, self.max_sequence_length - 2)
        
        span_starts = np.arange(sent_start, sent_end) + 1       # (sequence_length,)
        span_lengths = np.arange(self.max_span_length)          # (input_length,)
        span_ends = span_starts.reshape(-1, 1) + span_lengths   # (sequence_length, input_length)
        span_starts = np.expand_dims(span_starts, 1) \
            .repeat(self.max_span_length, axis=1)               # (sequence_length, input_length)
        span_starts, span_ends = span_starts.reshape(-1), span_ends.reshape(-1)
        spans = np.stack([span_starts, span_ends], axis=-1)     # (sequence_length * input_length, 2)
        spans = spans[span_ends <= sent_end]                    # (num_spans, 2)

        if skip_indices is not None:
            for index in skip_indices:
                mask = (index >= spans[:, 0]) & (index <= spans[:, 1])
                spans = spans[~mask]
        
        spans = [tuple(span) for span in spans.tolist()]
        spans_mask = np.ones(len(spans), dtype=np.int).tolist()

        return spans, spans_mask
    
    def _encode_syntactic(self, text, bert_offset_mapping):
        doc = self.stanza_nlp([text.split()]).to_dict()
        offset = 0
        upos = []; deprel = []
        stanza_offset_mapping = []
        for sent in doc:
            for token in sent:
                upos.append(token["upos"])
                deprel.append((
                    token["id"] + offset - 1, 
                    token["head"] - 1,
                    token["deprel"],
                ))
                stanza_offset_mapping.append((
                    token["start_char"],
                    token["end_char"],
                ))
            offset += len(sent)

        j = 0
        syntactic_upos_ids = []
        for i, (start, end) in enumerate(bert_offset_mapping):
            if start == end:
                syntactic_upos_ids.append(self.stanza_upos_unit2id["<PAD>"])
                continue
            syntactic_upos_ids.append(self.stanza_upos_unit2id[upos[j]])
            if end >= stanza_offset_mapping[j][1]:
                j += 1
        return syntactic_upos_ids

    def _encode_labels(self, entities, spans, input_length, offset_mapping):
        span2label_map = dict()
        for start, end, label, string in entities:
            # keep entities which are not truncated
            if start >= input_length or end >= start >= input_length:
                continue
            # span-to-label map(token-level)
            start, end = start + 1, end + 1                             # CLS, SEP
            span = (start, end - 1)
            span_label = span2label_map.get(span, None)
            if span_label is not None and span_label != label:
                print(f"\nLabel conflict of span {span}(current: {span_label}, new: {label})")
            else:
                span2label_map[span] = label                                # 左闭右闭

        labels = []
        for span in spans:
            label = span2label_map.get(span, "O")
            if label == "O":
                if random.random() < self.negative_sampling:
                    label = IGNORE_INDEX
                else:
                    label = self.label2id[label]
            else:
                label = self.label2id[label]
            labels.append(label)
        return labels

    def __call__(self, example):
        text: str = example["text"]

        inputs = self._encode_text(text)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        input_length = inputs["attention_mask"].sum().item()
        offset_mapping = inputs["offset_mapping"].numpy().tolist()

        example = self.converter(example)   # char -> token
        tokens, entities = example["text"], example["entities"]
        sent_start = example.get("sent_start", 0)
        sent_end = example.get("sent_end", len(tokens))

        # codec_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"])[1:-1]
        # assert len(tokens) and all([t1 == t2 for t1, t2 in zip(tokens, codec_tokens)])

        # encode spans
        skip_indices = [idx for idx, token in enumerate(tokens) 
            if token in [
                # self.tokenizer.unk_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
                self.tokenizer.cls_token,
                # self.tokenizer.mask_token,
            ]
        ]
        spans, spans_mask = self._encode_spans(input_length, sent_start, sent_end, skip_indices)
        inputs["spans"], inputs["spans_mask"] = torch.tensor(spans), torch.tensor(spans_mask)

        # encode pos & depparse
        if self.stanza_nlp is not None:
            syntactic_upos_ids = self._encode_syntactic(text, offset_mapping)
            inputs["syntactic_upos_ids"] = torch.tensor(syntactic_upos_ids)

        if entities is None:
            inputs["labels"] = None
            return inputs

        labels = self._encode_labels(entities, spans, input_length - 2, offset_mapping)
        inputs["labels"] = torch.tensor(labels)  # (num_spans,)

        return inputs


class ProcessExample2FeatureZh(ProcessExample2Feature):

    def __init__(self, label2id, tokenizer, max_sequence_length, 
            max_span_length, negative_sampling, stanza_nlp=None):
        super().__init__(label2id, tokenizer, max_sequence_length, 
            max_span_length, negative_sampling, stanza_nlp)
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_span_length = max_span_length
        self.negative_sampling = negative_sampling
        self.converter = lambda x: x
        self.stanza_nlp = stanza_nlp
        if stanza_nlp is not None:
            self.stanza_upos_unit2id = self.stanza_nlp.processors['pos'].vocab._vocabs['upos']._unit2id

    def _encode_text(self, text: List[str]):
        num_chars = len(text)
        if getattr(self.tokenizer, "do_ref_tokenize"):
            text = "".join(text)
            words = []
            for word in jieba.cut(text):
                if is_chinese(word):
                    words.append(word)
                else:
                    words.extend(list(word))
            text = words
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_sequence_length,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_length = inputs["attention_mask"].sum().item() - 2
        pad_length = self.max_sequence_length - input_length - 2 - 1
        if getattr(self.tokenizer, "do_ref_tokenize"):
            tokens = self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0])[1: 1 + input_length]
            assert len(tokens) == num_chars
        return inputs


class GaiicTrack2ProcessExample2Feature(ProcessExample2FeatureZh):
    pass


class ConditionalProcessExample2Feature(ProcessExample2Feature):

    def __call__(self, example):
        inputs = super().__call__(example)
        
        entities = example["conditional_entities"]
        offset_mapping = inputs["offset_mapping"].numpy().tolist()

        conditional_ids = [0] * self.max_sequence_length
        for entity in entities:
            for span in entity:
                start, end, label, string = span
                start += 1; end += 1    # CLS
                for pos in range(start, end):
                    conditional_ids[pos] = 1
        inputs["conditional_ids"] = torch.tensor(conditional_ids)

        return inputs


def load_dataset(data_class, process_class, data_name, data_dir, data_type, tokenizer, max_sequence_length, 
                 context_size, max_span_length, negative_sampling, stanza_nlp=None, **kwargs):
    do_preprocess = "do_preprocess" in kwargs and kwargs.pop("do_preprocess")
    process_piplines = [
        # AugmentRandomMask(p=0.1) if data_type == "train" else None,
        # AugmentDropRandomEntity(p=0.1) if data_type == "train" else None,
        # AugmentMaskRandomEntity(mask_token="[MASK]", p=0.1) if data_type == "train" else None,
        # AugmentSynonymReplace(p=0.1, augment_labels=[
        #     label for label in data_class.get_labels() if label not in {
        #         "1", "2", "3", "4",
        #         "19", "20", "21", "23",
        #         "37", "38", "39", "40",
        #     }
        # ]) if data_type == "train" else None,
        ProcessPreprocess() if do_preprocess else None,
        ProcessConvertLevel(tokenizer, "word2char") if data_class in [  # english

        ] else None,
        process_class(
            data_class.label2id(), tokenizer, max_sequence_length,
            max_span_length, negative_sampling, stanza_nlp,
        ),
    ]
    return data_class(data_name, data_dir, data_type, process_piplines, 
        context_size=context_size, use_cache=True, **kwargs)


class CoAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input1, input1_mask, input2, input2_mask):
        pass

class XBilinear(nn.Module):

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, div: int = 1) -> None:
        super().__init__()
        self.features1_projection = nn.Linear(in1_features, in1_features // div)
        self.features2_projection = nn.Linear(in2_features, in2_features // div)
        self.bilinear = nn.Bilinear(in1_features // div, in2_features // div, out_features, bias=bias)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = self.features1_projection(input1)
        input2 = self.features2_projection(input2)
        return self.bilinear(input1, input2)


class XBiaffineRel(nn.Module):

    def __init__(self, hidden_size: int, out_features: int, bias: bool = True, div: int = 1) -> None:
        super().__init__()

        self.W1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size // div, out_features))
        self.W2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size // div))
        self.linear = nn.Linear(hidden_size + hidden_size + hidden_size // div, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
        relative_positions_embeddings = self._generate_relative_positions_embeddings(512, hidden_size // div)
        self.register_buffer("relative_positions_embeddings", relative_positions_embeddings)

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.W2.size(1))
        init.uniform_(self.W1, -bound, bound)
        init.uniform_(self.W2, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)
        # # init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        # # init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        # self.W1.data.normal_(mean=0.0, std=0.02)
        # self.W2.data.normal_(mean=0.0, std=0.02)
        # if self.bias is not None:
        #     self.bias.data.zero_()
    
    @classmethod
    def _generate_relative_positions_embeddings(cls, length, depth, max_relative_position=127):
        vocab_size = max_relative_position * 2 + 1
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        embeddings_table = np.zeros([vocab_size, depth])
        for pos in range(vocab_size):
            for i in range(depth // 2):
                embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
                embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

        embeddings_table_tensor = torch.tensor(embeddings_table).float()
        flat_relative_positions_matrix = final_mat.view(-1)
        one_hot_relative_positions_matrix = torch.nn.functional.one_hot(
            flat_relative_positions_matrix,num_classes=vocab_size).float()
        embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
        my_shape = list(final_mat.size())
        my_shape.append(depth)
        embeddings = embeddings.view(my_shape)
        return embeddings

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            input1: torch.Tensor[batch_size, num_spans, hidden_size]
            input2: torch.Tensor[batch_size, num_spans, hidden_size]
            spans: torch.Tensor[batch_size, num_spans, 2]
        
        """
        pe = self.relative_positions_embeddings[spans[..., 0], spans[..., 1]]
        # [x; y; p] W3
        output = self.linear(torch.cat([input1, input2, pe], dim=-1))
        # (x W1)(y W2 + p) / d
        input1 = torch.einsum("bnh,hdc->bndc", input1, self.W1)
        input2 = torch.einsum("bnh,hd->bnd", input2, self.W2)
        scale = math.sqrt(self.relative_positions_embeddings.size(-1))
        output = output + torch.einsum("bndc,bnd->bnc", input1, input2 + pe) / scale
        if self.bias is not None: 
            output = output + self.bias
        return output

class SpanClassificationHead(nn.Module):

    def __init__(self, hidden_size, num_labels, max_span_length, width_embedding_size,
                 do_projection=False, do_cln=False, do_biaffine=False, 
                 do_co_attention=False, extract_method="endpoint"):
        super().__init__()

        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_size)

        self.do_projection = do_projection
        if self.do_projection:
            self.start_projection = nn.Linear(hidden_size, hidden_size)
            self.end_projection   = nn.Linear(hidden_size, hidden_size)

        self.do_cln = do_cln
        if self.do_cln:
            # assert self.do_projection, "在投影后进行CLN才有意义"
            self.start_cln = ConditionalLayerNorm(hidden_size, hidden_size)
            self.end_cln   = ConditionalLayerNorm(hidden_size, hidden_size)

        self.do_co_attention = do_co_attention
        if self.do_co_attention:
            ... # TODO:

        extract_method_func_map = {
            "endpoint": (
                self.forward_endpoint, 
                hidden_size * 2,
            ),
            "maxpool": (
                lambda sequence_output, spans: self.forward_pool(
                    sequence_output, spans, pool_method="max"), 
                hidden_size,
            ),
            "meanpool": (
                lambda sequence_output, spans: self.forward_pool(
                    sequence_output, spans, pool_method="mean"),
                hidden_size,
            ),
            "endpoint-pool": (
                self.forward_endpoint_pool,
                hidden_size * 4,
            ),
            "cln": (
                self.forward_cln,
                hidden_size * 2,
            )
        }
        self.forward, num_features = extract_method_func_map[extract_method]
        if extract_method == "cln":
            self.head2tail_cln = ConditionalLayerNorm(hidden_size, hidden_size)
            self.tail2head_cln = ConditionalLayerNorm(hidden_size, hidden_size)

        num_features += width_embedding_size
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

        self.do_biaffine = do_biaffine
        if self.do_biaffine:
            self.bilinear = XBiaffineRel(hidden_size, num_labels, bias=True, div=4)

    def _extract_spans_embedding_endpoint(self, sequence_output, spans):

        if self.do_projection:
            sequence_start_output = self.start_projection(sequence_output)
            sequence_end_output   = self.end_projection  (sequence_output)
        else:
            sequence_start_output = sequence_output
            sequence_end_output   = sequence_output

        if self.do_cln:
            start_cls, end_cls = sequence_start_output[:, 0], sequence_end_output[:, 0]
            sequence_start_output = self.start_cln(sequence_start_output, end_cls)
            sequence_end_output   = self.end_cln  (sequence_end_output, start_cls)
        
        if self.do_co_attention:
            ... # TODO:

        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end   = spans[:, :, 1].view(spans.size(0), -1)

        spans_start_embedding = self.batched_index_select(sequence_start_output, spans_start)
        spans_end_embedding   = self.batched_index_select(sequence_end_output,   spans_end  )

        return spans_start_embedding, spans_end_embedding
    
    def _extract_spans_embedding_pool(self, sequence_output, spans, pool_method):

        batch_size, sequence_length, hidden_size = sequence_output.size()
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end   = spans[:, :, 1].view(spans.size(0), -1)
        spans_width = spans_end - spans_start + 1

        # span indices & mask
        max_span_width = spans_width.max().item()
        max_span_range_indices = torch.arange(0, max_span_width, 
            dtype=torch.long, device=sequence_output.device).view(1, 1, -1)     # (1, 1, max_span_width)
        raw_span_indices = spans_start.unsqueeze(-1) + max_span_range_indices   # (batch_size, num_spans, max_span_width)

        span_mask = max_span_range_indices <= spans_width.unsqueeze(-1)         # (batch_size, num_spans, max_span_width)
        span_mask = span_mask & (raw_span_indices < sequence_length) & (0 <= raw_span_indices)
        span_indices = raw_span_indices * span_mask                             # (batch_size, num_spans, max_span_width)

        # batched index select
        span_indices_flatten = span_indices.view(batch_size, -1)                # (batch_size, num_spans * max_span_width)
        span_indices_embedding = self.batched_index_select(
            sequence_output, span_indices_flatten)                              # (batch_size, num_spans * max_span_width, hidden_size)
        span_indices_embedding = span_indices_embedding.view(
            batch_size, -1, max_span_width, hidden_size)                        # (batch_size, num_spans, max_span_width, hidden_size)
        
        # masked max/mean
        extend_span_mask = span_mask.unsqueeze(-1)                              # (batch_size, num_spans, max_span_width, 1)
        if pool_method == "max":
            span_indices_embedding.masked_fill_(
                ~extend_span_mask, torch.finfo(sequence_output.dtype).min)                         
            span_indices_embedding, _ = span_indices_embedding.max(dim=2)       # (batch_size, num_spans, hidden_size)
        elif pool_method == "mean":
            span_indices_embedding = span_indices_embedding * extend_span_mask
            span_indices_embedding = span_indices_embedding.sum(dim=2) / extend_span_mask.sum(dim=2)
        
        return span_indices_embedding
    
    def _extract_width_embedding(self, spans):
        
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end   = spans[:, :, 1].view(spans.size(0), -1)
        spans_width = spans_end - spans_start + 1

        return self.width_embedding(spans_width)

    def forward_endpoint(self, sequence_output, spans, return_spans_embedding=False):

        spans_start_embedding, spans_end_embedding = \
            self._extract_spans_embedding_endpoint(sequence_output, spans)
        spans_width_embedding = self._extract_width_embedding(spans)
        
        spans_embedding = torch.cat([
            spans_start_embedding, 
            spans_end_embedding,
            spans_width_embedding,
        ], dim=-1)  # (batch_size, num_spans, num_features)

        if return_spans_embedding:
            return spans_embedding

        logits = self.classifier(spans_embedding)
        if self.do_biaffine:
            logits = logits + self.bilinear(spans_start_embedding, spans_end_embedding, spans)

        return logits

    def forward_pool(self, sequence_output, spans, pool_method):
        
        spans_width_embedding = self._extract_width_embedding(spans)

        spans_pool_embedding = self._extract_spans_embedding_pool(
            sequence_output, spans, pool_method),
        spans_embedding = torch.cat([
            spans_pool_embedding,
            spans_width_embedding,
        ], dim=-1)             # (batch_size, num_spans, num_features)

        logits = self.classifier(spans_embedding)

        return logits

    def forward_endpoint_pool(self, sequence_output, spans):

        spans_start_embedding, spans_end_embedding = \
            self._extract_spans_embedding_endpoint(sequence_output, spans)
        spans_meanpool_embedding = self._extract_spans_embedding_pool(
            sequence_output, spans, "mean")
        spans_maxpool_embedding = self._extract_spans_embedding_pool(
            sequence_output, spans, "max")
        spans_width_embedding = self._extract_width_embedding(spans)
        
        spans_embedding = torch.cat([
            spans_start_embedding,
            spans_end_embedding,
            spans_meanpool_embedding,
            spans_maxpool_embedding,
            spans_width_embedding
        ], dim=-1)  # (batch_size, num_spans, num_features)

        logits = self.classifier(spans_embedding)
        if self.do_biaffine:
            logits = logits + self.bilinear(spans_start_embedding, spans_end_embedding)

        return logits

    def forward_cln(self, sequence_output, spans):

        spans_start_embedding, spans_end_embedding = \
            self._extract_spans_embedding_endpoint(sequence_output, spans)
        spans_width_embedding = self._extract_width_embedding(spans)
        
        spans_embedding = torch.cat([
            self.tail2head_cln(spans_start_embedding.unsqueeze(1), spans_end_embedding).squeeze(1),
            self.head2tail_cln(spans_end_embedding.unsqueeze(1), spans_start_embedding).squeeze(1),
            spans_width_embedding
        ], dim=-1)  # (batch_size, num_spans, num_features)

        logits = self.classifier(spans_embedding)
        if self.do_biaffine:
            logits = logits + self.bilinear(spans_start_embedding, spans_end_embedding)

        return logits   # TODO:

    @classmethod
    def batched_index_select(cls, input, index):
        batch_size, sequence_length, hidden_size = input.size()
        index_onehot = F.one_hot(index, num_classes=sequence_length).float()
        output = torch.bmm(index_onehot, input)
        return output

    @classmethod
    def decode(cls, logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits=True):
        other_id = label2id["O"]
        if is_logits:
            probas = logits_or_labels.softmax(dim=-1)
            # TODO: 这题不利于提高召回率，应确保精确率为主。
            #       因为当分词出现错误时，同时降低准确率、召回率，而直接预测为非实体，仅降低召回率。
            # probas, labels = probas.max(dim=-1)     # 实体类别概率
            _, labels = probas.max(dim=-1)
            probas = 1 - probas[..., other_id]      # 是实体的概率
            labels = torch.where((probas < thresh) | (labels == IGNORE_INDEX),      # 提精度
                torch.full_like(labels, other_id), labels)
            # ---
            # probas[..., other_id] = torch.where(probas[..., other_id] < thresh,
            #     torch.zeros_like(probas[..., other_id]), probas[..., other_id])     # 提召回
            # # probas, labels = probas.max(dim=-1) # TODO: 实体类别概率
            # _, labels = probas.max(dim=-1)
            # probas = 1 - probas[..., other_id]  # 是实体的概率

            # 无27、45两类
            labels[labels == 27] = other_id
            labels[labels == 45] = other_id
        else:
            probas, labels = torch.ones_like(logits_or_labels), logits_or_labels    # (batch_size, sequence_length)

        batch_size = logits_or_labels.size(0)
        labels = [[id2label.get(id, "O") for id in ids] for ids in labels.cpu().numpy().tolist()]
        probas = probas.cpu().numpy().tolist()
        spans = spans.cpu().numpy().tolist()
        spans_mask = spans_mask.cpu().numpy().tolist()

        decode_entities = []
        for i in range(batch_size):
            entities = []
            sent_start = float("inf")
            for span, mask, label, proba in zip(spans[i], spans_mask[i], labels[i], probas[i]):
                if mask == 0: break
                start, end = span       # 左闭右闭
                sent_start = min(sent_start, start)
                if label == "O": continue
                start -= sent_start; end -= sent_start
                entities.append((start, end + 1, label, proba))   # 左闭右开
            # entities = sorted(entities, key=lambda x: (x[0], x[1] - x[0]))
            # entities = cls.drop_overlap_baseline(entities)
            entities = cls.drop_overlap_nms(entities)
            entities = [entity[:-1] for entity in entities]
            decode_entities.append(entities)

        return decode_entities
    
    @classmethod
    def drop_overlap_baseline(cls, entities):
        if len(entities) == 0: return []
        entities = sorted(entities, key=lambda x: (x[0], x[0] - x[1]))
        sequence_length = max([entity[1] for entity in entities])
        ner_tags = entities_to_ner_tags(sequence_length, entities)
        spans = get_spans_bio(ner_tags)
        entities = [(start, end + 1, label, 0.0) for label, start, end in spans]
        return entities
    
    @classmethod
    def drop_overlap_nms(cls, entities):
        """
        Parameters
        ----------
            entities: List[Tuple[int, int, str, float]]

        Return
        ------
            entities: List[Tuple[int, int, str, float]]

        Notes
        -----
            - 简化iou计算，仅计算重叠长度；
            - 未考虑标签，若需对同类别实体去重，则在传入该函数前处理；
        """
        if len(entities) == 0: return []
        
        X = np.array(entities)
        starts = X[:, 0].astype(np.int)
        ends   = X[:, 1].astype(np.int)
        scores = X[:, 3].astype(np.float)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            # 保留得分最高的一个
            keep.append(i)
            # 计算相交区间
            starts_ = np.maximum(starts[i], starts[order[1:]])
            ends_   = np.minimum(ends  [i], ends  [order[1:]])
            # 计算相交长度
            inter = np.maximum(0, ends_ - starts_)
            # 保留不相交实体
            indices = np.where(inter <= 0)[0]
            order = order[indices + 1]
        
        entities = [entities[idx] for idx in keep]
        return entities

class SpanClassificationLoss(nn.Module):

    def __init__(
        self, 
        num_labels, 
        loss_type="lsr", 
        label_smoothing_eps=0.0, 
        focal_gamma=2.0, 
        focal_alpha=0.25,
        reduction="mean", 
        ignore_index=-100
    ):
        super().__init__()
        self.num_labels = num_labels
        self.reduction = reduction
        if loss_type == "ce":
            self.loss_fct = nn.CrossEntropyLoss( 
                reduction="none", ignore_index=ignore_index)
        elif loss_type == "lsr":
            self.loss_fct = LabelSmoothingCE(eps=label_smoothing_eps, 
                reduction="none", ignore_index=ignore_index)
        elif loss_type == "focal":
            self.loss_fct = FocalLoss(num_labels=num_labels, 
                gamma=focal_gamma, alpha=focal_alpha, reduction="none")
        elif loss_type == "lsrol":
            self.loss_fct = None    # TODO:

    def forward(
        self,
        logits,  # (batch_size, num_spans, num_labels)
        labels,  # (batch_size, num_spans,)
        mask,    # (batch_size, num_spans,)
    ):
        num_labels = logits.size(-1)
        loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1))
        activate_mask = ((mask == 1) & (labels != IGNORE_INDEX)).view(-1)
        loss = loss[activate_mask]
        if loss.size(0) == 0: return 0
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class SpanClassificationRDropLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, p, q, mask=None):
        
        batch_size, num_spans, num_labels = p.size()
        if mask is None:
            mask = torch.ones(batch_size, num_spans, dtype=torch.bool, device=p.device)
        mask = (mask > 0).unsqueeze(-1).expand(batch_size, num_spans, num_labels)
        
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # mask is for seq-level tasks
        p_loss.masked_fill_(~mask, 0.)
        q_loss.masked_fill_(~mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss


class SpanClassificationOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    groundtruths: List[List[Entity]] = None
    predictions: List[List[Entity]] = None


class ModelForSpanClassification(PreTrainedModel):

    head_class = SpanClassificationHead
    loss_class = SpanClassificationLoss
    rdrop_loss_clsss = SpanClassificationRDropLoss

    def __init__(self, config):
        super().__init__(config)

        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(
            config, "classifier_dropout") else config.hidden_dropout_prob)
        
        if config.conditional:
            self.conditional_embeddings = nn.Embedding(2, self.config.hidden_size)
            self.conditional_layer_norm = ConditionalLayerNorm(
                self.config.hidden_size, self.config.hidden_size)
        
        if config.do_lstm:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, 
                num_layers=config.num_lstm_layers, batch_first=True, bidirectional=True)

        if config.use_syntactic:
            self.syntactic_upos_embeddings = nn.Embedding(config.syntactic_upos_size, config.hidden_size)
        
        self.head = self.head_class(
            config.hidden_size, config.num_labels,
            config.max_span_length, config.width_embedding_size,
            config.do_projection, config.do_cln, config.do_biaffine,
            config.do_co_attention, config.extract_method,
        )

        self.loss_fct = self.loss_class(
            num_labels=config.num_labels, 
            loss_type=config.loss_type,
            label_smoothing_eps=config.label_smoothing,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            reduction="mean",
            ignore_index=IGNORE_INDEX,
        )

        if config.do_rdrop:
            self.rdrop_loss_func = self.rdrop_loss_clsss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        syntactic_upos_ids=None,
        conditional_ids=None,
        spans=None,
        spans_mask=None,
        labels=None,
        rdrop_forward=False,
        return_dict=None,
    ):
        if self.config.do_rdrop and (not rdrop_forward) and self.training:
            outputs1 = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                syntactic_upos_ids=syntactic_upos_ids,
                conditional_ids=conditional_ids,
                spans=spans,
                spans_mask=spans_mask,
                labels=labels,
                rdrop_forward=True,
                return_dict=return_dict,
            )
            outputs2 = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                syntactic_upos_ids=syntactic_upos_ids,
                conditional_ids=conditional_ids,
                spans=spans,
                spans_mask=spans_mask,
                labels=labels,
                rdrop_forward=True,
                return_dict=return_dict,
            )
            loss = (outputs1["loss"] + outputs2["loss"]) / 2. + \
                self.config.rdrop_weight * self.rdrop_loss_func(
                    outputs1["logits"], outputs2["logits"], spans_mask)
            return SpanClassificationOutput(
                loss=loss,
                logits=outputs1["logits"],
                predictions=outputs1["predictions"],
                groundtruths=outputs1["groundtruths"],
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = self.config.use_last_n_layers is not None

        if not (self.config.conditional or self.config.use_syntactic):

            # TODO: Albert-style
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # self.base_model.encoder(outputs["last_hidden_state"], attention_mask) # AlBert Style

            # TODO: LayerPooling
            # https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
            if self.config.use_last_n_layers is not None:
                last_hidden_state = outputs["hidden_states"]
                last_hidden_state = torch.stack(last_hidden_state[- self.config.use_last_n_layers: ], dim=-1)
                if self.config.agg_last_n_layers == "mean":
                    last_hidden_state = torch.mean(last_hidden_state, dim=-1)
                elif self.config.agg_last_n_layers == "sum":
                    last_hidden_state = torch.sum(last_hidden_state, dim=-1)
                elif self.config.agg_last_n_layers == "max":
                    last_hidden_state = torch.max(last_hidden_state, dim=-1)[0]
                outputs["last_hidden_state"] = last_hidden_state
                
        elif self.config.conditional:
            embedding_output = self.base_model.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )
            conditional_output = self.conditional_embeddings(conditional_ids)
            embedding_output = embedding_output + conditional_output
            
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_ids.size(), input_ids.device)
            outputs = self.base_model.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if self.config.use_last_n_layers is not None:
                last_hidden_state = outputs["hidden_states"]
                last_hidden_state = torch.stack(last_hidden_state[- self.config.use_last_n_layers: ], dim=-1)
                if self.config.agg_last_n_layers == "mean":
                    last_hidden_state = torch.mean(last_hidden_state, dim=-1)
                elif self.config.agg_last_n_layers == "sum":
                    last_hidden_state = torch.sum(last_hidden_state, dim=-1)
                elif self.config.agg_last_n_layers == "max":
                    last_hidden_state = torch.max(last_hidden_state, dim=-1)[0]
                outputs["last_hidden_state"] = last_hidden_state

            last_hidden_state = outputs["last_hidden_state"]
            condition = last_hidden_state * conditional_ids.unsqueeze(-1)
            condition = condition.sum(dim=1) / conditional_ids.sum(dim=1).unsqueeze(-1)
            last_hidden_state = self.conditional_layer_norm(last_hidden_state, condition)
            outputs["last_hidden_state"] = last_hidden_state
            if hasattr(self.base_model, "pooler") and self.base_model.pooler is not None:
                pooler_output = self.base_model.pooler(last_hidden_state)
                outputs["pooler_output"] = pooler_output
        
        elif self.config.use_syntactic:
            embedding_output = self.base_model.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )
            syntactic_upos_output = self.syntactic_upos_embeddings(syntactic_upos_ids)
            embedding_output = embedding_output + syntactic_upos_output
            
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_ids.size(), input_ids.device)
            outputs = self.base_model.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if self.config.use_last_n_layers is not None:
                last_hidden_state = outputs["hidden_states"]
                last_hidden_state = torch.stack(last_hidden_state[- self.config.use_last_n_layers: ], dim=-1)
                if self.config.agg_last_n_layers == "mean":
                    last_hidden_state = torch.mean(last_hidden_state, dim=-1)
                elif self.config.agg_last_n_layers == "sum":
                    last_hidden_state = torch.sum(last_hidden_state, dim=-1)
                elif self.config.agg_last_n_layers == "max":
                    last_hidden_state = torch.max(last_hidden_state, dim=-1)[0]
                outputs["last_hidden_state"] = last_hidden_state

        sequence_output = outputs["last_hidden_state"]

        if self.config.do_lstm:
            sequence_lengths = attention_mask.sum(dim=1)
            packed_sequence_output = pack_padded_sequence(
                sequence_output, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_sequence_output, _ = self.lstm(packed_sequence_output)
            unpacked_sequence_output, _ = pad_packed_sequence(
                packed_sequence_output, batch_first=True, total_length=sequence_lengths.max())
            sequence_output = sequence_output + unpacked_sequence_output

        # self.dropout.p = np.random.uniform(0, 0.5)
        sequence_output = self.dropout(sequence_output)
        logits = self.head(sequence_output, spans)          # (batch_size, num_spans, num_labels)

        # logits = None
        # n = 5
        # p = self.dropout.p
        # for i in range(n):
        #     self.dropout.p = (i + 1) * 0.1
        #     if i == 0:
        #         logits = self.head(self.dropout(sequence_output), spans)
        #     else:
        #         logits = logits + self.head(self.dropout(sequence_output), spans)
        # self.dropout.p = p
        # logits = logits / n

        predictions = None
        if not self.training:
            predictions = self.decode(logits, spans, spans_mask, self.config.decode_thresh,
                                      self.config.label2id, self.config.id2label, is_logits=True)

        loss = None
        groundtruths = None
        if labels is not None:
            loss = self.loss_fct(logits, labels, spans_mask)
            if not self.training:
                groundtruths = self.decode(labels, spans, spans_mask, self.config.decode_thresh,
                                           self.config.label2id, self.config.id2label, is_logits=False)

        """
        Q1：在哪一层对词向量混合？
        A1：如果在BERT输入层对词向量混合，容易导致BERT预训练权重失效，因此选择在BERT模型输出进行混合；
            注意由于span无法线性混合，需先计算span表征后混合；
            同样地，biaffine需要感知span，无法计算biaffine打分，因此仅作用在MLP分类层
        Q2：如何采样样本进行混合？
        A2：对同批次内数据打乱后，一一对应进行混合
        Q3：如何实现标签混合？
        A3：经推导，不必显式混合标签，可通过损失混合，如下：
            $$
            L(\overline{X}, \overline{Y}) =
                \lambda L(\overline{X}, Y_1) +
                (1 - \lambda) L(\overline{X}, Y_2)
            $$
        Q4：关于span mask处理？是取并集还是跟随标签进行样本集打乱？
        A4：为减少噪声引入，这里采用后者
        """
        if self.config.do_mixup and labels is not None:
            indices = torch.randperm(logits.size(0))    # 用于打乱样本，同批次内数据进行混合
            beta = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
            spans_embedding = self.head(sequence_output, spans, reutrn_spans_embedding=True)
            # spans_embedding.masked_fill_(spans_mask.unsqueeze(-1).eq(0), value=0.0)
            spans_embedding_mixup = beta * spans_embedding + (1 - beta) * spans_embedding[indices]
            logits_mixup = self.head.classifier(spans_embedding_mixup)  # XXX: biaffine需要感知span，故不用于计算打分
            loss_mixup = beta * self.loss_fct(logits_mixup, labels, spans_mask) + \
                (1 - beta) * self.loss_fct(logits_mixup, labels[indices], spans_mask[indices])
            loss = loss + loss_mixup * self.config.mixup_weight

        if not return_dict:
            outputs = (logits, predictions, groundtruths) + outputs[2:]
            return ((loss,) + outputs) if loss is not None else outputs

        return SpanClassificationOutput(
            loss=loss,
            logits=logits,
            predictions=predictions,
            groundtruths=groundtruths,
        )

    @classmethod
    def decode(cls, logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits=True):
        return cls.head_class.decode(logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits)


class BertForSpanClassification(BertPreTrainedModel, ModelForSpanClassification):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.init_weights()


class RobertaForSpanClassification(RobertaPreTrainedModel, ModelForSpanClassification):

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.init_weights()


class NeZhaForSpanClassification(NeZhaPreTrainedModel, ModelForSpanClassification):

    def __init__(self, config):
        super().__init__(config)

        self.bert = NeZhaModel(config)
        self.init_weights()

class XYClassifier(nn.Module):
    """ 目的是减少两组分类任务的共享参数 """

    def __init__(self, in_features, hidden_size, num_xlabels, num_ylabels):
        super().__init__()
        # TODO: 一般bert后网络层次不会很深，太多参数影响微调性能；但可借助该层交互任务信息？
        # self.share_fc = nn.Sequential(
        #     nn.Linear(in_features, hidden_size),
        #     nn.ReLU()
        # )
        # in_features = hidden_size
        self.x_fc = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_xlabels),
        )
        self.y_fc = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_ylabels),
        )

    def forward(self, x):
        # x = self.share_fc(x)
        logits = torch.cat([self.x_fc(x), self.y_fc(x)], dim=-1)
        return logits
class SpanClassificationXYHead(SpanClassificationHead):

    def __init__(self, hidden_size, num_labels, max_span_length, width_embedding_size,
                 do_projection=False, do_cln=False, do_biaffine=False,
                 do_co_attention=False, extract_method="endpoint",
                 num_xlabels=None, num_ylabels=None):
        super().__init__(hidden_size, num_labels, max_span_length, width_embedding_size,
                 do_projection=do_projection, do_cln=do_cln, do_biaffine=do_biaffine,
                 do_co_attention=do_co_attention, extract_method=extract_method)

        if num_xlabels is not None and num_ylabels is not None: # for the first initialization
            self.num_xlabels = num_xlabels
            self.num_ylabels = num_ylabels

            # TODO: 以下方式分类时，除最后一层fc，其他参数均共享，不利于特征多样化？
            num_features = self.classifier[0].in_features
            # self.classifier = nn.Sequential(
            #     nn.Linear(num_features, hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size, num_xlabels + num_ylabels),
            # )
            self.classifier = XYClassifier(num_features, hidden_size, num_xlabels, num_ylabels)
            self.do_biaffine = do_biaffine
            if self.do_biaffine:
                self.bilinear = XBiaffineRel(hidden_size, num_xlabels + num_ylabels, bias=True, div=4)

    @classmethod
    def decode(cls, logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits=True):
        other_id = label2id["O"]
        if is_logits:
            xprobas = logits_or_labels[..., : 4].softmax(dim=-1)    # XXX: num_xlabels不是类变量
            yprobas = logits_or_labels[..., 4: ].softmax(dim=-1)

            # TODO: 这题不利于提高召回率，应确保精确率为主。
            #       因为当分词出现错误时，同时降低准确率、召回率，而直接预测为非实体，仅降低召回率。
            _, xlabels = xprobas.max(dim=-1)
            _, ylabels = yprobas.max(dim=-1)
            xlabels = torch.where((1 - xprobas[..., other_id] < thresh) | (xlabels == IGNORE_INDEX),   # 提精度
                torch.full_like(xlabels, other_id), xlabels)
            ylabels = torch.where((1 - yprobas[..., other_id] < thresh) | (ylabels == IGNORE_INDEX),   # 提精度
                torch.full_like(ylabels, other_id), ylabels)
            # ---
            # xprobas[..., other_id] = torch.where(xprobas[..., other_id] < thresh,
            #     torch.zeros_like(xprobas[..., other_id]), xprobas[..., other_id])   # 提召回
            # yprobas[..., other_id] = torch.where(yprobas[..., other_id] < thresh,
            #     torch.zeros_like(yprobas[..., other_id]), yprobas[..., other_id])   # 提召回
            # _, xlabels = xprobas.max(dim=-1)
            # _, ylabels = yprobas.max(dim=-1)

            # 两组标签取交集，即均预测出实体时，最终才输出实体，提高精确率
            # [0, 1, 2, 3] x [0, 1, 2, 3, ..., 18] -> [0, 1, 2, 3, ..., 54]
            labels = torch.where((xlabels * ylabels) > 0, 18 * (xlabels - 1) + ylabels, 0)
            probas = (1 - xprobas[..., other_id]) * (1 - yprobas[..., other_id])    # 是实体的概率
            # 两组标签取并集，即一组预测为实体时，召回另一组标签实体，提高召回率
            # p = xprobas.clone(); p[..., 0] = 0.0
            # q = yprobas.clone(); q[..., 0] = 0.0
            # xlabels, ylabels = torch.where(ylabels != 0, p.max(dim=-1)[1], xlabels), \
            #                    torch.where(xlabels != 0, q.max(dim=-1)[1], ylabels)
            # # [0, 1, 2, 3] x [0, 1, 2, 3, ..., 18] -> [0, 1, 2, 3, ..., 54]
            # labels = torch.where((xlabels * ylabels) > 0, 18 * (xlabels - 1) + ylabels, 0)
            # probas = 1 - xprobas[..., other_id] * yprobas[..., other_id]    # 是实体的概率

            # 无27、45两类
            labels[labels == 27] = other_id
            labels[labels == 45] = other_id
        else:
            probas, labels = torch.ones_like(logits_or_labels), logits_or_labels    # (batch_size, sequence_length)

        batch_size = logits_or_labels.size(0)
        labels = [[id2label.get(id, "O") for id in ids] for ids in labels.cpu().numpy().tolist()]
        probas = probas.cpu().numpy().tolist()
        spans = spans.cpu().numpy().tolist()
        spans_mask = spans_mask.cpu().numpy().tolist()

        decode_entities = []
        for i in range(batch_size):
            entities = []
            sent_start = float("inf")
            for span, mask, label, proba in zip(spans[i], spans_mask[i], labels[i], probas[i]):
                if mask == 0: break
                start, end = span       # 左闭右闭
                sent_start = min(sent_start, start)
                if label == "O": continue
                start -= sent_start; end -= sent_start
                entities.append((start, end + 1, label, proba))   # 左闭右开
            # entities = sorted(entities, key=lambda x: (x[0], x[1] - x[0]))
            # entities = cls.drop_overlap_baseline(entities)
            entities = cls.drop_overlap_nms(entities)
            entities = [entity[:-1] for entity in entities]
            decode_entities.append(entities)

        return decode_entities

class SpanClassificationXYLoss(nn.Module):

    def __init__(
        self,
        num_labels,
        num_xlabels=None,
        num_ylabels=None,
        loss_type="lsr",
        label_smoothing_eps=0.0,
        focal_gamma=2.0,
        focal_alpha=0.25,
        reduction="mean",
        ignore_index=-100
    ):
        super().__init__()
        if num_xlabels is not None:     # for the first initialization
            self.loss_fct_x = SpanClassificationLoss(
                num_xlabels,
                loss_type=loss_type,
                label_smoothing_eps=label_smoothing_eps,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                reduction=reduction,
                ignore_index=ignore_index
            )
        if num_xlabels is not None:     # for the first initialization
            self.loss_fct_y = SpanClassificationLoss(
                num_ylabels,
                loss_type=loss_type,
                label_smoothing_eps=label_smoothing_eps,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                reduction=reduction,
                ignore_index=ignore_index
            )

    def forward(
        self,
        logits,  # (batch_size, num_spans, num_labels)
        labels,  # (batch_size, num_spans,)
        mask,    # (batch_size, num_spans,)
    ):
        num_xlabels = self.loss_fct_x.num_labels
        num_ylabels = self.loss_fct_y.num_labels
        # [0, 1, 2, ..., 54] -> [0, 1, 2, 3]
        xlabels = torch.where(labels > 0, (labels - 1) // 18 + 1, labels)
        # [0, 1, 2, ..., 54] -> [0, 1, 2, ..., 18]
        ylabels = torch.where(labels > 0, (labels - 1) %  18 + 1, labels)
        loss_x = self.loss_fct_x(logits[..., : num_xlabels], xlabels, mask)
        loss_y = self.loss_fct_y(logits[..., num_xlabels: ], ylabels, mask)
        # TODO: 多任务权重问题？
        return loss_x + loss_y
        # return loss_x.log() + loss_y.log()  # https://kexue.fm/archives/8870/comment-page-1

class SpanClassificationXYRDropLoss(SpanClassificationRDropLoss):

    def __init__(self, num_xlabels=None, num_ylabels=None):
        super().__init__()
        self.num_xlabels = num_xlabels
        self.num_ylabels = num_ylabels
   
    def forward(self, p, q, mask=None):
        px, py = p[..., : self.num_xlabels], p[..., self.num_xlabels: ]
        qx, qy = q[..., : self.num_xlabels], q[..., self.num_xlabels: ]
        loss = super().forward(px, qx, mask) + super().forward(py, qy, mask)
        return loss

class ModelForSpanClassificationXY(ModelForSpanClassification):

    head_class = SpanClassificationXYHead
    loss_class = SpanClassificationXYLoss
    rdrop_loss_clsss = SpanClassificationXYRDropLoss

    def __init__(self, config):
        super().__init__(config)

        self.head = self.head_class(
            config.hidden_size, config.num_labels,
            config.max_span_length, config.width_embedding_size,
            config.do_projection, config.do_cln, config.do_biaffine,
            config.do_co_attention, config.extract_method,
            config.num_xlabels, config.num_ylabels,
        )

        self.loss_fct = self.loss_class(
            num_labels=config.num_labels,
            num_xlabels=config.num_xlabels,
            num_ylabels=config.num_ylabels,
            loss_type=config.loss_type,
            label_smoothing_eps=config.label_smoothing,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            reduction="mean",
            ignore_index=IGNORE_INDEX,
        )

        if config.do_rdrop:
            self.rdrop_loss_func = self.rdrop_loss_clsss(config.num_xlabels, config.num_ylabels)

class NeZhaForSpanClassificationXY(NeZhaPreTrainedModel, ModelForSpanClassificationXY):

    def __init__(self, config):
        super().__init__(config)

        self.bert = NeZhaModel(config)
        self.init_weights()


def precision_recall_fscore_support(y_true: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    y_pred: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    *,
                                    average: Optional[str] = None,
                                    labels: Optional[List[str]] = None,
                                    entity_type: str = "all",
                                    label_convert: Callable = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False):
    """Compute precision, recall, F-measure and support for each class.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.

        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        suffix : bool, False by default.

    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]

        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]

        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]

        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.

    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """

    def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):

            if entity_type == "all":
                pass
            if entity_type == "without_label":
                true = [(start, end, "_") for start, end, _ in true]
                pred = [(start, end, "_") for start, end, _ in pred]
            
            if label_convert is not None:
                true = [(start, end, label_convert(label)) for start, end, label in true]
                pred = [(start, end, label_convert(label)) for start, end, label in pred]

            for start, end, label in true:
                entities_true[label].add((i, (start, end)))

            for start, end, label in pred:
                entities_pred[label].add((i, (start, end)))

        if labels is not None:
            entities_true = {k: v for k, v in entities_true.items() if k in labels}
            entities_pred = {k: v for k, v in entities_pred.items() if k in labels}

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


class SequenceLabelingScoreEntity(SequenceLabelingScore):

    def __init__(self, labels, average="micro", entity_type="all", label_convert=None):
        super().__init__(labels, average)
        self.entity_type = entity_type
        self.label_convert = label_convert

    def value(self):
        columns = ["label", "precision", "recall", "f1", "support"]
        values = []

        for label in [self.average] + sorted(self.labels):
            p, r, f, s = precision_recall_fscore_support(
                self.target, self.preds, average=self.average,
                labels=None if label == self.average else [label],
                entity_type=self.entity_type, 
                label_convert=self.label_convert,
            )
            values.append([label, p, r, f, s])
        df = pd.DataFrame(values, columns=columns)
        f1 = df[df['label'] == self.average]['f1'].item()
        return {
            f"df_{self.name()}": df, f"f1_{self.average}_{self.name()}": f1,   # for monitor
        }

    def name(self):
        name = f"{self.entity_type}_entity"
        if self.label_convert:
            name = f"{name}_{self.label_convert.__name__}"
        return name


class SequenceLabelingScoreSpan(SequenceLabelingScore):

    def update(self, 
        preds: List[List[Entity]], 
        target: List[List[Entity]],
    ):
        for pred_entities, target_entities in zip(preds, target):
            self.preds.append([])
            self.target.append([])
            for entity in pred_entities:
                spans = self.entity2spans(entity)
                self.preds[-1].extend(spans)
            for entity in target_entities:
                spans = self.entity2spans(entity)
                self.target[-1].extend(spans)
                
    def value(self):
        res = super().value()
        return {
            f"df_{self.name()}": res["df"], 
            f"f1_{self.average}_{self.name()}": res[f"f1_{self.average}"],   # for monitor
        }
        
    def entity2spans(self, entity):
        spans = []
        num_spans = len(entity)
        entity = sorted(entity, key=lambda x: (x[0], x[1]))
        for i, (start, end, label) in enumerate(entity):
            start_prefix = "B" if i == 0 else "I"
            end_prefix = "E" if i == num_spans - 1 else "I"
            prefix = start_prefix + end_prefix
            span = (start, end, prefix + "-" + label)
            spans.append(span)
        for span_a, span_b in zip(entity[:-1], entity[1:]):
            start, end = span_a[1], span_b[0]
            span = (start, end, "II" + "-" + "O")
            spans.append(span)
        return spans

    def name(self):
        return "span"
    

class Trainer(TrainerBase):

    def build_model_param_optimizer(self, model):
        '''
        若需要对不同模型赋予不同学习率，则指定`base_model_name`,
        在`transformer`模块中，默认为`base_model_name=`base_model`.
        对于base_model使用learning_rate，
        其余统一使用other_learning_rate
        '''
        msg = (f"The initial learning rate for model params : {self.opts.learning_rate} ,"
                f"and {self.opts.other_learning_rate}"
                )
        self.logger.info(msg)

        no_decay = ["bias", 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        base_model = getattr(model, self.opts.base_model_name)
        base_model_param = list(base_model.named_parameters())
        base_model_param_ids = [id(p) for n, p in base_model_param]
        other_model_param = [(n, p) for n, p in model.named_parameters() if id(p) not in base_model_param_ids]

        if self.opts.layer_wise_lr_decay is None:
            optimizer_grouped_parameters.extend(
                self._param_optimizer(base_model_param, self.opts.learning_rate, no_decay, self.opts.weight_decay))
        else:
            layer_decay_map = {
                i: self.opts.learning_rate * (
                    self.opts.layer_wise_lr_decay ** (self.model.config.num_hidden_layers - i - 1)
                ) 
                for i in range(self.model.config.num_hidden_layers)
            }
            for name, params in base_model_param:
                named_params = [(name, params)]
                if name.startswith("embeddings"):
                    optimizer_grouped_parameters.extend(
                        self._param_optimizer(named_params, layer_decay_map[0], no_decay, self.opts.weight_decay))
                elif name.startswith("encoder"):
                    layer_no = int(name.split(".")[2])
                    layer_lr = layer_decay_map[layer_no]
                    optimizer_grouped_parameters.extend(
                        self._param_optimizer(named_params, layer_lr, no_decay, self.opts.weight_decay))
                else:
                    optimizer_grouped_parameters.extend(
                        self._param_optimizer(named_params, self.opts.learning_rate, no_decay, self.opts.weight_decay))

        optimizer_grouped_parameters.extend(
            self._param_optimizer(other_model_param, self.opts.other_learning_rate, no_decay,
                                    self.opts.weight_decay))
        return optimizer_grouped_parameters

    def build_pseudo_dataloader(self, pseudo_data, num_batch_per_epoch):
        '''
        Load train datasets
        '''
        if isinstance(pseudo_data, DataLoader):
            return pseudo_data
        elif isinstance(pseudo_data, Dataset):
            batch_size = len(pseudo_data) // num_batch_per_epoch
            sampler = RandomSampler(pseudo_data) if not hasattr(pseudo_data, 'sampler') else pseudo_data.sampler
            collate_fn = pseudo_data.collate_fn if hasattr(pseudo_data, 'collate_fn') else None
            data_loader = DataLoader(pseudo_data,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     drop_last=self.opts.drop_last,
                                     num_workers=self.opts.num_workers)
            return data_loader
        else:
            raise TypeError("train_data type{} not support".format(type(pseudo_data)))

    def train_update(self):
        if self.use_amp:
            # AMP: gradients need unscaling
            self.scaler.unscale_(self.optimizer)
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(self.optimizer) if self.use_apex else self.model.parameters(),
                self.max_grad_norm)
        optimizer_was_run = True
        if self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
        else:
            self.optimizer.step()
        # >>> SWA >>>
        if optimizer_was_run:
            if self.opts.swa_enable and self.global_step > self.opts.swa_start:
                if self.swa_model.n_averaged == 0:
                    self.logger.info(f"\nStart SWA - step {self.global_step}")
                if self.global_step % self.opts.swa_freq == 0:
                    self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.scheduler.step()   # Update learning rate schedule
        # <<< SWA <<<
        self.model.zero_grad()  # Reset gradients to zero
        self.global_step += 1

    def train_step(self, step, batch, batch_pl=None, pseudo_weight=1.0):
        outputs = self.train_forward(batch)
        outputs_pl = self.train_forward(batch_pl) \
            if batch_pl is not None and pseudo_weight > 0.0 else {"loss": 0.0}
        loss = outputs['loss'] + outputs_pl["loss"] * pseudo_weight
        self.train_backward(loss)
        should_save = False
        should_logging = False
        if self.opts.adv_enable:
            self.train_adv(batch)
        if (step + 1) % self.gradient_accumulation_steps == 0 or (
                self.steps_in_epoch <= self.gradient_accumulation_steps
                and (step + 1) == self.steps_in_epoch
        ):
            self.train_update()
            should_logging = self.global_step % self.opts.logging_steps == 0
            should_save = self.global_step % self.opts.save_steps == 0
            self.records['loss_meter'].update(loss.item(), n=1)
            self.writer.add_scalar('loss/train_loss', loss.item(), self.global_step)
            if hasattr(self.scheduler, 'get_lr'):
                self.writer.add_scalar('learningRate/train_lr', self.scheduler.get_lr()[0], self.global_step)
            return outputs, should_logging, should_save
        else:
            return None, should_logging, should_save

    def unlabled_weight(self, step):
        pseudo_weight = self.opts.pseudo_weight
        if step < self.opts.pseudo_warmup_start_step:
            pseudo_weight = 0.0
        elif step > self.opts.pseudo_warmup_end_step:
            pseudo_weight = pseudo_weight
        else:
            coeff = (step - self.opts.pseudo_warmup_start_step) / \
                (self.opts.pseudo_warmup_end_step - self.opts.pseudo_warmup_start_step)
            pseudo_weight *= coeff
        return pseudo_weight

    # TODO 多机分布式训练
    def train(self, train_data, dev_data=None, pseudo_data=None, resume_path=None, start_epoch=1, state_to_save=dict()):
        train_dataloader = self.build_train_dataloader(train_data)
        pseudo_dataloader = self.build_pseudo_dataloader(pseudo_data, len(train_dataloader)) \
            if pseudo_data is not None else [None] * len(train_dataloader)
        num_training_steps = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs
        self.steps_in_epoch = len(train_dataloader)
        if self.scheduler is None:
            self.scheduler = self.build_lr_scheduler(num_training_steps)
        self.resume_from_checkpoint(resume_path=resume_path)
        self.build_model_warp()
        # >>> SWA >>>
        self.swa_train_data = None
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_model_checkpoint = None
        if self.opts.swa_enable:
            self.opts.swa_start = int(self.opts.swa_start) if self.opts.swa_start > 1.0 \
                else int(num_training_steps * self.opts.swa_start)
            self.logger.info("Initializing SWA optimization.")
            self.logger.info(f"swa_start = {self.opts.swa_start:d}")
            self.logger.info(f"swa_lr = {self.opts.swa_lr:.6f}")
            self.logger.info(f"swa_freq = {self.opts.swa_freq:d}")
            self.logger.info(f"swa_anneal_epochs = {self.opts.swa_anneal_epochs:d}")
            self.logger.info(f"swa_anneal_strategy = {self.opts.swa_anneal_strategy:s}")
            self.swa_train_data = self.build_train_dataloader(train_data)
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=self.opts.swa_lr,
                anneal_epochs=self.opts.swa_anneal_epochs,
                anneal_strategy=self.opts.swa_anneal_strategy,
            )
            self.swa_model_checkpoint = ModelCheckpoint(
                mode=self.opts.checkpoint_mode,
                monitor=f"swa_{self.opts.checkpoint_monitor}",
                ckpt_dir=self.opts.output_dir,
                verbose=self.opts.checkpoint_verbose,
                save_best=self.opts.checkpoint_save_best,
                keys_to_ignore_on_save=self.keys_to_ignore_on_checkpoint_save
            )
        # <<< SWA <<<
        self.print_summary(len(train_data), num_training_steps)
        self.optimizer.zero_grad()
        seed_everything(self.opts.seed, verbose=False)  # Added here for reproductibility (even between python 2 and 3)
        if self.opts.logging_steps < 0:
            self.opts.logging_steps = len(train_dataloader) // self.gradient_accumulation_steps
            self.opts.logging_steps = max(1, self.opts.logging_steps)
        if self.opts.save_steps < 0:
            self.opts.save_steps = len(train_dataloader) // self.gradient_accumulation_steps
            self.opts.save_steps = max(1, self.opts.save_steps)
        self.build_record_tracker()
        self.reset_metrics()
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=self.num_train_epochs)
        for epoch in range(start_epoch, int(self.num_train_epochs) + 1):
            pbar.epoch(current_epoch=epoch)
            for step, (batch, batch_pl) in enumerate(zip(train_dataloader, pseudo_dataloader)):
                outputs, should_logging, should_save = self.train_step(
                    step, batch, batch_pl, self.unlabled_weight(self.global_step))
                if outputs is not None:
                    if self.opts.ema_enable:
                        self.model_ema.update(self.model)
                    pbar.step(step, {'loss': outputs['loss'].item()})
                if (self.opts.logging_steps > 0 and self.global_step > 0) and \
                        should_logging and self.opts.evaluate_during_training:
                    self.evaluate(dev_data)
                    if self.opts.ema_enable and self.model_ema is not None:
                        self.evaluate(dev_data, prefix_metric='ema')
                    # >>> SWA >>>
                    if self.opts.swa_enable and self.swa_model is not None and self.swa_model.n_averaged > 0:
                        update_bn(self.swa_train_data, self.swa_model, device=self.swa_model.module.device)
                        model, self.model = self.model, self.swa_model
                        self.evaluate(dev_data, prefix_metric='swa', reset_records=False)
                        self.model = model; del model
                    # <<< SWA <<<
                    if hasattr(self.writer, 'save'):
                        self.writer.save()
                if (self.opts.save_steps > 0 and self.global_step > 0) and should_save:
                    # model checkpoint
                    if self.model_checkpoint:
                        state = self.build_state_object(**state_to_save)
                        if self.opts.evaluate_during_training:
                            if self.model_checkpoint.monitor not in self.records['result']:
                                msg = ("There were expected keys in the eval result: "
                                    f"{', '.join(list(self.records['result'].keys()))}, "
                                    f"but get {self.model_checkpoint.monitor}."
                                    )
                                raise TypeError(msg)
                            self.model_checkpoint.step(
                                state=state,
                                current=self.records['result'][self.model_checkpoint.monitor]
                            )
                        else:
                            self.model_checkpoint.step(
                                state=state,
                                current=None
                            )
                    # >>> SWA >>>
                    if self.opts.swa_enable and self.swa_model is not None and self.swa_model.n_averaged > 0:
                        state = {
                            "model": self.swa_model.module,
                            "opts": self.opts,
                            "global_step": self.global_step,
                        }
                        for key, value in state_to_save.items():
                            if key not in state:
                                state[key] = value
                        if self.opts.evaluate_during_training:
                            if self.swa_model_checkpoint.monitor not in self.records['result']:
                                msg = ("There were expected keys in the eval result: "
                                    f"{', '.join(list(self.records['result'].keys()))}, "
                                    f"but get {self.swa_model_checkpoint.monitor}."
                                    )
                                raise TypeError(msg)
                            self.swa_model_checkpoint.step(
                                state=state,
                                current=self.records['result'][self.swa_model_checkpoint.monitor]
                            )
                        else:
                            self.swa_model_checkpoint.step(
                                state=state,
                                current=None
                            )
                    # <<< SWA <<<
            # early_stopping
            if self.early_stopping:
                if self.early_stopping.monitor not in self.records['result']:
                    msg = ("There were expected keys in the eval result: "
                           f"{', '.join(list(self.records['result'].keys()))}, "
                           f"but get {self.early_stopping.monitor}."
                           )
                    raise TypeError(msg)
                self.early_stopping.step(
                    current=self.records['result'][self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.writer:
            self.writer.close()

    def evaluate(self, dev_data, prefix_metric=None, save_dir=None, save_result=False, file_name=None, reset_records=True):
        '''
        Evaluate the model on a validation set
        '''
        all_batch_list = []
        eval_dataloader = self.build_eval_dataloader(dev_data)
        if reset_records: self.build_record_tracker()
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, batch in enumerate(eval_dataloader):
            batch = self.predict_forward(batch)
            if batch.get('loss'):
                self.records['loss_meter'].update(batch['loss'], n=1)   # XXX: loss未通过`prefix_metric`区分

            groundtruths = batch["groundtruths"]
            start_index = eval_dataloader.batch_size * step
            end_index   = eval_dataloader.batch_size * (step + 1)
            examples = dev_data.examples[start_index: end_index]
            for i, example in enumerate(examples):
                ## raw examples -> token level examples
                for proc in dev_data.process_piplines[:-1]:
                    if proc is None: continue
                    example = proc(example)
                example = dev_data.process_piplines[-1].converter(example)
                groundtruths[i] = [
                    (start, end, label) for start, end, label, string in example["entities"]
                ]

            all_batch_list.append(batch)
            pbar.step(step)
        self.records['result']['eval_loss'] = self.records['loss_meter'].avg

        # find best thresh
        if self.opts.find_best_decode_thresh:
            best_metric_value = - np.inf
            best_decode_thresh = None
            decode_threshs = np.linspace(0.4, 0.8, 20)
            pbar = tqdm(decode_threshs, total=decode_threshs.size)
            for decode_thresh in pbar:
                self.reset_metrics()
                all_batch_list_temp = []
                for batch_list in all_batch_list:
                    all_batch_list_temp.append(dict())
                    logits = batch_list["logits"]
                    spans  = batch_list["spans" ]
                    spans_mask = batch_list["spans_mask"]
                    groundtruths = batch_list["groundtruths"]
                    predictions = self.model.decode(logits, spans, spans_mask, decode_thresh, 
                        self.model.config.label2id, self.model.config.id2label)
                    all_batch_list_temp[-1]["predictions"] = predictions
                    all_batch_list_temp[-1]["groundtruths"] = groundtruths
                self.update_metrics(all_batch_list_temp, prefix_metric)
                for metric in self.metrics:
                    if metric.name() != self.opts.checkpoint_monitor.split("_", 3)[-1]:
                        continue
                    metric_value = metric.value()
                    key = self.opts.checkpoint_monitor.split("_", 1)[1]
                    if key not in metric_value: 
                        continue
                    metric_value = metric_value[key]
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_decode_thresh = decode_thresh
                        for batch_list, batch_list_temp in zip(all_batch_list, all_batch_list_temp):
                            batch_list["predictions"] = batch_list_temp["predictions"]
                            batch_list["best_decode_thresh"] = best_decode_thresh
                pbar.set_description(f"Finding best decode threshold ({best_decode_thresh:.2f}/{best_metric_value:.6f})...")
            self.logger.info(f"Best decde threshold is {best_decode_thresh:.6f}, best metric value is {best_metric_value}.")

        self.reset_metrics()
        self.update_metrics(all_batch_list, prefix_metric)
        self.print_evaluate_result()
        if save_result:
            if file_name is None: file_name = f"dev_eval_results.pkl"
            self.save_predict_result(data=all_batch_list, file_name=file_name, save_dir=save_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_metrics(self, all_batch_list, prefix):
        preds = []; target = []
        for batch in all_batch_list:
            preds.extend(batch["predictions"])
            target.extend(batch["groundtruths"])
        prefix = "" if prefix is None else prefix + "_"
        for metric in self.metrics:
            metric.update(preds=preds, target=target)
            value = metric.value()
            if value:
                if isinstance(value, float):
                    self.records["result"][f"{prefix}eval_{metric.name()}"] = value
                elif isinstance(value, dict):
                    self.records["result"].update({f"{prefix}eval_{k}": v for k, v in value.items()})
                else:
                    raise ValueError("metric value type: expected one of (float, dict)")
            else:
                self.logger.info(f"{metric.name()} value is None")

    def print_evaluate_result(self):
        '''
        打印evaluation结果
        '''
        if len(self.records['result']) == 0:
            self.logger.warning("eval result record is empty")
        self.logger.info("***** Evaluating results of %s *****", self.opts.task_name)
        self.logger.info("  global step = %s", self.global_step)
        metric_key_value_map = dict()
        for key in sorted(self.records['result'].keys()):
            if isinstance(self.records['result'][key], pd.DataFrame):
                self.logger.info("  %s = \n%s", key, str(round(self.records['result'][key], 4)))
            else:
                self.logger.info("  %s = %s", key, str(round(self.records['result'][key], 4)))
                name = key.split("_")[1] if "_" in key else key
                self.writer.add_scalar(f"{name[0].upper() + name[1:]}/{key}", self.records['result'][key],
                                       int(self.global_step / self.opts.logging_steps))
                metric_key_value_map[key] = self.records['result'][key]
        if use_wandb:
            wandb.log(metric_key_value_map)
    
    @torch.no_grad()
    def predict_forward(self, batch):
        mc_dropout_rate = self.opts.mc_dropout_rate
        mc_dropout_times = self.opts.mc_dropout_times
        do_mc_dropout = mc_dropout_rate is not None and mc_dropout_times is not None
        
        # eval mode
        self.model.eval()
        if do_mc_dropout:
            # for module in self.model.modules():
            #     if isinstance(module, nn.Dropout):
            #         module.p = mc_dropout_rate
            #         module.training = True
            self.model.dropout.p = mc_dropout_rate
            self.model.dropout.training = True
        
        # forward
        inputs = self.build_batch_inputs(batch)
        if do_mc_dropout:
            outputs = None
            for i in range(mc_dropout_times):
                if i == 0:
                    outputs = self.model(**inputs)
                    outputs["logits"] /= mc_dropout_times
                else:
                    outputs["logits"] += self.model(**inputs)["logits"] / mc_dropout_times
            outputs["predictions"] = self.model.decode(
                outputs["logits"], inputs["spans"], inputs["spans_mask"],
                self.opts.decode_thresh, self.opts.label2id, self.opts.id2label
            )
        else:
            outputs = self.model(**inputs)
        if 'loss' in outputs and outputs['loss'] is not None:
            outputs['loss'] = outputs['loss'].mean().detach().item()

        outputs = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in
                   outputs.items()}
        batch = {key: value for key, value in dict(batch, **outputs).items() if
                 key not in self.keys_to_ignore_on_result_save}
        return batch
    
MODEL_CLASSES = {
    "bert": (BertConfig, BertForSpanClassification, BertTokenizerZh),
    "nezha": (BertConfig, NeZhaForSpanClassification, BertTokenizerZh),
    "nezhaxy": (BertConfig, NeZhaForSpanClassificationXY, BertTokenizerZh),
}

DATA_CLASSES = {
    "gaiic": (GaiicTrack2SpanClassificationDataset, GaiicTrack2ProcessExample2Feature),
}


def build_opts():
    # sys.argv.append("outputs/gaiic_nezha_nezha-ref-154k-spanv1-datav4-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3/gaiic_nezha_nezha-ref-154k-spanv1-datav4-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3_opts.json")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        opts = Argparser.parse_args_from_json(json_file=os.path.abspath(sys.argv[1]))
    else:
        parser = Argparser.get_training_parser()
        group = parser.add_argument_group(title="user-defined", description="user-defined")
        group.add_argument("--do_check", action="store_true")
        group = parser.add_argument_group(title="data-related", description="data-related")
        group.add_argument("--do_ref_tokenize", action="store_true")
        group.add_argument("--do_preprocess", action="store_true")
        group.add_argument("--context_size", type=int, default=0)
        group.add_argument("--negative_sampling", type=float, default=0.0)
        group.add_argument("--labels", nargs="+", type=str, default=None)
        group.add_argument("--max_train_examples", type=int, default=None)
        group.add_argument("--max_eval_examples", type=int, default=None)
        group.add_argument("--max_test_examples", type=int, default=None)
        group = parser.add_argument_group(title="model-related", description="model-related")
        group.add_argument("--classifier_dropout", type=float, default=0.1)
        group.add_argument("--max_span_length", type=int, default=30)
        group.add_argument("--width_embedding_size", type=int, default=128)
        group.add_argument("--extract_method", type=str, default="endpoint")
        group.add_argument("--decode_thresh", type=float, default=0.0)
        group.add_argument("--find_best_decode_thresh", action="store_true")
        group.add_argument("--use_sinusoidal_width_embedding", action="store_true")
        group.add_argument("--do_lstm", action="store_true")
        group.add_argument("--num_lstm_layers", type=int, default=1)
        group.add_argument("--do_projection", action="store_true")
        group.add_argument("--do_cln", action="store_true")
        group.add_argument("--do_co_attention", action="store_true")
        group.add_argument("--do_biaffine", action="store_true")
        group.add_argument("--use_syntactic", action="store_true")
        group.add_argument("--syntactic_upos_size", type=int, default=21)
        group.add_argument("--use_last_n_layers", type=int, default=None)
        group.add_argument("--agg_last_n_layers", type=str, default="mean")
        group.add_argument("--mc_dropout_rate", type=float, default=None)
        group.add_argument("--mc_dropout_times", type=int, default=None)
        group.add_argument("--layer_wise_lr_decay", type=float, default=None)
        group = parser.add_argument_group(title="loss function-related", description="loss function-related")
        group.add_argument("--loss_type", type=str, default="lsr", choices=["ce", "lsr", "focal"])
        group.add_argument("--label_smoothing", type=float, default=0.0)
        group.add_argument("--focal_gamma", type=float, default=2.0)
        group.add_argument("--focal_alpha", type=float, default=0.25)
        group.add_argument("--do_mixup", action="store_true")
        group.add_argument("--mixup_alpha", type=float, default=7.0)
        group.add_argument("--mixup_weight", type=float, default=0.5)
        group = parser.add_argument_group(title="R-Drop", description="R-Drop")
        group.add_argument("--do_rdrop", action="store_true")
        group.add_argument("--rdrop_weight", type=float, default=0.3)
        group = parser.add_argument_group(title="pseudo label", description="pseudo label")
        group.add_argument("--max_pseudo_examples", type=int, default=None)
        group.add_argument("--pseudo_input_file", type=str, default=None)
        group.add_argument("--pseudo_weight", type=float, default=1.0)
        group.add_argument("--pseudo_warmup_start_step", type=int, default=-1)
        group.add_argument("--pseudo_warmup_end_step", type=int, default=-1)
        group = parser.add_argument_group(title="swa", description="Stochastic Weight Averaging (SWA)")
        group.add_argument("--swa_enable", action="store_true")
        group.add_argument("--swa_start", type=float, default=0.75)
        group.add_argument("--swa_lr", type=float, default=1e-6)
        group.add_argument("--swa_freq", type=int, default=1)
        group.add_argument("--swa_anneal_epochs", type=int, default=100)
        group.add_argument("--swa_anneal_strategy", type=str, default="cos", choices=["cos", "linear"])
        opts = parser.parse_args_from_parser(parser)

    # TODO: for debug
    # opts.do_train = opts.do_eval = opts.do_predict = False
    # opts.do_eval = True
    # opts.do_check = True
    # opts.do_predict = True
    # opts.context_size = 10
    # opts.max_span_length = 50
    # opts.train_max_seq_length = 512
    # opts.max_train_examples = 100
    # opts.do_lstm = False
    # opts.device_id = 'cpu'
    # opts.do_eval = opts.do_predict = opts.do_check = False
    # opts.do_train = True
    # opts.use_syntactic = True
    # opts.syntactic_upos_size = 21
    
    return opts

def update_example_entities(tokenizer, examples, entities, processes=[]):
    assert len(examples) == len(entities)
    updated = []
    for example, entities in tqdm(zip(examples, entities), 
            total=len(examples), desc="Updating examples..."):
        example = deepcopy(example)
        for process in processes:
            if process is None:
                continue
            example = process(example)
        if isinstance(example["text"], str):
            tokens = tokenizer.tokenize(example["text"])
        else:
            tokens = example["text"]
        example["entities"] = []
        for start, end, label in entities:      # entity
            example["entities"].append(
                (start, end, label, tokens[start: end])
            )
        updated.append(example)
    return updated

def entities_to_ner_tags(sequence_length, entities):
    ner_tags = ["O"] * sequence_length
    for start, end, label, *_ in entities:
        # >>> global pointer baseline >>>
        # https://github.com/DataArk/GAIIC2022-Product-Title-Entity-Recognition-Baseline/blob/main/code/baseline.ipynb
        if 'I' in ner_tags[start]:
            continue
        if 'B' in ner_tags[start] and 'O' not in ner_tags[end - 1]:
            continue
        if 'O' in ner_tags[start] and 'B' in ner_tags[end - 1]:
            continue
        # <<< global pointer baseline <<<
        for i in range(start, end):
            if i == start:
                tag = f"B-{label}"
            else:
                tag = f"I-{label}"
            if ner_tags[i] != "O" and ner_tags[i] != tag:
                logger.info(f"Label Conflict occurs at {i}, current: {ner_tags[i]}, new: {tag}")
            ner_tags[i] = tag
    return ner_tags

def main(opts):

    logger = Logger(opts=opts)
    if use_wandb:
        # Initialize wandb
        if not wandb.run:   # for hyperparameter search
            wandb.init(
                project="gaiic2022-track2", 
                group=None,
                name=opts.experiment_code,
                tags=[opts.task_name, opts.model_type, "span"],
                config=opts, 
            )

    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]
    data_class, process_class = DATA_CLASSES[opts.task_name]

    # data processor
    logger.info("initializing data processor")
    stanza_nlp = None
    if opts.use_syntactic:
        import stanza
        stanza_nlp = stanza.Pipeline(lang="zh", processors="tokenize,mwt,pos,lemma,depparse", use_gpu=True)
        stanza_nlp.processors["tokenize"].config.update({"pretokenized": True})
    tokenizer_kwargs = {
        "do_lower_case": opts.do_lower_case,
        "do_ref_tokenize": opts.do_ref_tokenize,
    }
    try:
        tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, **tokenizer_kwargs)
    except AssertionError:
        # XXX: AssertionError: Config has to be initialized with question_encoder and generator config
        tokenizer = tokenizer_class.from_pretrained(os.path.join(opts.pretrained_model_path, "vocab.txt"), **tokenizer_kwargs)
    train_dataset = pseudo_dataset = dev_dataset = test_dataset = None
    if opts.do_train or opts.do_check:
        train_dataset = load_dataset(data_class, process_class, opts.train_input_file, opts.data_dir, "train",
                                    tokenizer, opts.train_max_seq_length, opts.context_size, opts.max_span_length, 
                                    opts.negative_sampling, stanza_nlp=stanza_nlp, labels=opts.labels, 
                                    max_examples=opts.max_train_examples, do_preprocess=opts.do_preprocess,
                                    )
        if opts.pseudo_input_file is not None and not opts.do_check:
            pseudo_dataset = load_dataset(data_class, process_class, opts.pseudo_input_file, opts.data_dir, "train",
                                         tokenizer, opts.train_max_seq_length, opts.context_size, opts.max_span_length,
                                         opts.negative_sampling, stanza_nlp=stanza_nlp, labels=opts.labels,
                                         max_examples=opts.max_pseudo_examples, do_preprocess=opts.do_preprocess,
                                         )
    if (opts.do_train and opts.evaluate_during_training) or opts.do_eval:
        dev_dataset   = load_dataset(data_class, process_class, opts.eval_input_file, opts.data_dir, "dev",
                                    tokenizer, opts.eval_max_seq_length, opts.context_size, opts.max_span_length,
                                    opts.negative_sampling, stanza_nlp=stanza_nlp, labels=opts.labels, 
                                    max_examples=opts.max_eval_examples, do_preprocess=opts.do_preprocess,
                                    )
    if opts.do_predict:
        test_dataset  = load_dataset(data_class, process_class, opts.test_input_file, opts.data_dir, "test",
                                    tokenizer, opts.test_max_seq_length, opts.context_size, opts.max_span_length,
                                    opts.negative_sampling, stanza_nlp=stanza_nlp, labels=opts.labels, 
                                    max_examples=opts.max_test_examples, do_preprocess=opts.do_preprocess,
                                    )
    if stanza_nlp is not None and train_dataset.use_cache and pseudo_dataset.use_cache and dev_dataset.use_cache and test_dataset.use_cache:
        del stanza_nlp
    opts.label2id = data_class.label2id()
    opts.id2label = data_class.id2label()
    opts.num_labels = len(opts.label2id)
    opts.xlabel2id = data_class.xlabel2id()
    opts.xid2label = data_class.xid2label()
    opts.num_xlabels = len(opts.xlabel2id)
    opts.ylabel2id = data_class.ylabel2id()
    opts.yid2label = data_class.yid2label()
    opts.num_ylabels = len(opts.ylabel2id)

    # model
    logger.info("initializing model and config")
    config, unused_kwargs = config_class.from_pretrained(
        opts.pretrained_model_path, return_unused_kwargs=True,
        num_labels=opts.num_labels, id2label=opts.id2label, label2id=opts.label2id,
        num_xlabels=opts.num_xlabels, xid2label=opts.xid2label, xlabel2id=opts.xlabel2id,
        num_ylabels=opts.num_ylabels, yid2label=opts.yid2label, ylabel2id=opts.ylabel2id,
        conditional=False,
        classifier_dropout=opts.classifier_dropout, 
        layer_wise_lr_decay=opts.layer_wise_lr_decay,
        use_last_n_layers=opts.use_last_n_layers,
        agg_last_n_layers=opts.agg_last_n_layers,
        negative_sampling=opts.negative_sampling,
        max_span_length=opts.max_span_length, 
        width_embedding_size=opts.width_embedding_size,
        do_rdrop=opts.do_rdrop,
        rdrop_weight=opts.rdrop_weight,
        loss_type=opts.loss_type,
        label_smoothing=opts.label_smoothing, 
        focal_gamma=opts.focal_gamma,
        focal_alpha=opts.focal_alpha,
        do_mixup=opts.do_mixup,
        mixup_alpha=opts.mixup_alpha,
        mixup_weight=opts.mixup_weight,
        decode_thresh=opts.decode_thresh,
        do_lstm=opts.do_lstm, 
        num_lstm_layers=opts.num_lstm_layers,
        do_projection=opts.do_projection,
        do_co_attention=opts.do_co_attention, 
        do_cln=opts.do_cln, 
        extract_method=opts.extract_method,
        do_biaffine=opts.do_biaffine,
        use_syntactic=opts.use_syntactic,
        syntactic_upos_size=opts.syntactic_upos_size,
    )
    for key, value in unused_kwargs.items():
        setattr(config, key, value)  # FIXED: 默认`from_dict`中，只有config中有键才能设置值，这里强制设置

    model = model_class.from_pretrained(opts.pretrained_model_path, config=config) \
        if opts.do_train else model_class(config=config)
    model.to(opts.device)

    if opts.use_sinusoidal_width_embedding:
        logger.info("Initializing sinusoidal width embedding")
        def _init_weight(out: nn.Parameter):
            """
            Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
            the 2nd half of the vector. [dim // 2:]
            """
            n_pos, dim = out.shape
            position_enc = np.array(
                [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
            )
            out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
            sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
            out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
            out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
            out.detach_()
            # # Trainable
            # sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
            # out.data[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
            # out.data[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
            return out
        model.head.width_embedding.weight = _init_weight(model.head.width_embedding.weight)
    if opts.use_syntactic: 
        model.syntactic_upos_embeddings.weight.data.zero_()

    # trainer
    logger.info("initializing trainer")
    metrics = [
        SequenceLabelingScoreEntity({label for label in data_class.get_labels() \
            if label not in ["O",]}, "micro", entity_type="all"),
        SequenceLabelingScoreEntity({label for label in data_class.get_xlabels() \
            if label not in ["O",]}, "micro", entity_type="all", label_convert=data_class.fx),
        SequenceLabelingScoreEntity({label for label in data_class.get_ylabels() \
            if label not in ["O",]}, "micro", entity_type="all", label_convert=data_class.fy),
        SequenceLabelingScoreEntity({"_"}, "micro", entity_type="without_label"),
    ]
    trainer = Trainer(opts=opts, model=model, metrics=metrics, logger=logger)

    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, pseudo_data=pseudo_dataset, state_to_save={"vocab": tokenizer})

    if opts.do_eval:
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        if opts.eval_all_checkpoints:
            checkpoints = find_all_checkpoints(checkpoint_dir=opts.output_dir)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.evaluate(dev_data=dev_dataset, save_result=True, save_dir=prefix)

            # 保存为样本，用于分析
            char2token = ProcessConvertLevel(tokenizer, "char2token", lang="zh")
            token2char = ProcessConvertLevel(tokenizer, "token2char", lang="zh")
            results = load_pickle(os.path.join(checkpoint, f"dev_eval_results.pkl"))

            entities = list(chain(*[batch["groundtruths"] for batch in results]))
            examples = update_example_entities(tokenizer, dev_dataset.examples, entities, dev_dataset.process_piplines[:-1])
            with open(os.path.join(checkpoint, "groundtruths.json"), "w") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            with open(os.path.join(checkpoint, "groundtruths.span.jsonl"), "w") as f:
                for example in examples:
                    text = example["text"]
                    sent_start, sent_end = example["sent_start"], example["sent_end"]
                    dummy = dict(text=text, entities=[[(sent_start, sent_end, "_", text)]])
                    example["sent_start"], example["sent_end"] = char2token(dummy)["entities"][0][0][:2]
                    example["text"] = example["text"][example["sent_start"]: example["sent_end"]]
                    text = "".join(example["text"])
                    entities = []
                    for i, (start, end, label, string) in enumerate(example["entities"]):
                        entities.append((start, end, label, text[start: end]))
                    example["text"] = text
                    example["entities"] = entities
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

            entities = list(chain(*[batch["predictions"] for batch in results]))
            examples = update_example_entities(tokenizer, dev_dataset.examples, entities, dev_dataset.process_piplines[:-1])
            with open(os.path.join(checkpoint, "evaluations.json"), "w") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            with open(os.path.join(checkpoint, "evaluations.span.jsonl"), "w") as f:
                for example in examples:
                    text = example["text"]
                    sent_start, sent_end = example["sent_start"], example["sent_end"]
                    dummy = dict(text=text, entities=[[(sent_start, sent_end, "_", text)]])
                    example["sent_start"], example["sent_end"] = char2token(dummy)["entities"][0][0][:2]
                    example["text"] = example["text"][example["sent_start"]: example["sent_end"]]
                    text = "".join(example["text"])
                    entities = []
                    for i, (start, end, label, string) in enumerate(example["entities"]):
                        entities.append((start, end, label, text[start: end]))
                    example["text"] = text
                    example["entities"] = entities
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

    if opts.do_predict:
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.predict(test_data=test_dataset, save_result=True, save_dir=prefix)

            # 保存为样本，用于分析
            results = load_pickle(os.path.join(checkpoint, f"test_predict_results.pkl"))
            entities = list(chain(*[batch["predictions"] for batch in results]))
            examples = update_example_entities(tokenizer, test_dataset.examples, entities, test_dataset.process_piplines[:-1])
            # with open(os.path.join(checkpoint, "predictions.json"), "w") as f:
            #     for example in examples:
            #         f.write(json.dumps(example, ensure_ascii=False) + "\n")
            # 保存为提交格式
            predicion_file = f"{opts.test_input_file}.predictions.txt"
            with open(os.path.join(checkpoint, predicion_file), "w") as f:
                for example_no, example in tqdm(enumerate(examples), total=len(examples), 
                        desc=f"Writing to {predicion_file}"):
                    text = test_dataset.examples[example_no]["text"]
                    ner_tags = entities_to_ner_tags(len(text), example["entities"])
                    assert len(text) == len(ner_tags)
                    for token, tag in zip(text, ner_tags):
                        f.write(f"{token} {tag}\n")
                    f.write(f"\n")

    if opts.do_check:
        # check dataset & decode function
        span_labels = []
        sequence_lengths = []
        span_lengths = []
        metric = SequenceLabelingScoreEntity({label for label in \
            data_class.get_labels() if label not in ["O", ]}, "micro", entity_type="all")
        char2token = ProcessConvertLevel(tokenizer, "char2token")
        for example_no in tqdm(range(len(train_dataset)), total=len(train_dataset)):

            # print(example_no)
            # if example_no not in [44, ]: continue

            example, feature = train_dataset.examples[example_no], train_dataset[example_no]
            span_labels.extend(feature["labels"].cpu().numpy().tolist())
            text = example["text"]
            batch = train_dataset.collate_fn([feature])
            sequence_lengths.append(feature["attention_mask"].sum().item())

            # 真实标签
            for proc in train_dataset.process_piplines[:-1]:
                if proc is None: continue
                example = proc(example)
            if isinstance(train_dataset.process_piplines[-1], ProcessExample2FeatureZh):
                tokens = example["text"]
                entities = example["entities"]
            elif isinstance(train_dataset.process_piplines[-1], ProcessExample2Feature):
                example = char2token(example)                                  # token level
                tokens = example["text"]
                entities = example["entities"]
                
            # 解码标签
            offset_mapping = feature["offset_mapping"].cpu().numpy().tolist()
            decodes = ModelForSpanClassification.decode(
                batch["labels"], batch["spans"], batch["spans_mask"],
                opts.decode_thresh, opts.label2id, opts.id2label, is_logits=False)      # token level
            decodes = decodes[0]    # batch_size = 1
            # span长度统计(token level)
            for start, end, *_ in decodes:
                span_lengths.append(end - start)
            # token -> char
            example_codec = deepcopy(example)
            entities_codec = [
                (start, end, label, tokens[start: end])
                for start, end, label in decodes
            ]
            example_codec["entities"] = entities_codec                                  # token level
            decodes = example_codec["entities"]

            sort_key = lambda x: x[:2]
            entities = sorted(entities, key=sort_key)
            decodes = sorted(decodes, key=sort_key)
            # 批次为1
            predictions = [[
                (start, end, label) 
                for start, end, label, _ in decodes
            ]]
            groundtruths = [[
                (start - example["sent_start"], end - example["sent_start"], label)
                for start, end, label, _ in entities
            ]]
            metric.update(predictions, groundtruths)
        
        # 类别统计
        label_count_map = {opts.id2label[k]: v for k, v in \
            Counter(span_labels).items() if k in opts.id2label}
        logger.info(label_count_map)
        # 序列长度统计
        seqlen2count_map = Counter(sequence_lengths)
        logger.info(sorted(seqlen2count_map.items(), key=lambda x: x[0]))
        # 片段长度统计
        spanlen2count_map = Counter(span_lengths)
        logger.info(sorted(spanlen2count_map.items(), key=lambda x: x[0]))

        # 错误用例保存为文件
        # preds, target = metric.preds, metric.target
        # predict_entities = list(chain(*list(enumerate(preds))))
        # target_entities  = list(chain(*list(enumerate(target))))
        preds = []; target = []; indices = []
        for i, (p, t) in enumerate(zip(metric.preds, metric.target)):
            if p != t:
                preds.append((i, p))
                target.append((i, t))
                indices.append(i)
        predict_entities = list(chain(*list(preds)))
        target_entities  = list(chain(*list(target)))
        with open(os.path.join(opts.output_dir, f"check-predictions.json"), "w") as f:
            for index_or_entities in predict_entities:
                if isinstance(index_or_entities, int):
                    f.write(json.dumps(index_or_entities, ensure_ascii=False) + "\n")
                else:
                    for entity in index_or_entities:
                        f.write(json.dumps(entity, ensure_ascii=False) + "\n")
        with open(os.path.join(opts.output_dir, f"check-groundtruths.json"), "w") as f:
            for index_or_entities in target_entities:
                if isinstance(index_or_entities, int):
                    f.write(json.dumps(index_or_entities, ensure_ascii=False) + "\n")
                else:
                    for entity in index_or_entities:
                        f.write(json.dumps(entity, ensure_ascii=False) + "\n")
        with open(os.path.join(opts.output_dir, f"check-examples.json"), "w") as f:
            for index in indices:
                example = train_dataset.examples[index]
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        value = metric.value()
        logger.info(value)
    
    return trainer

"""
[[Q]Exception: The wandb backend process has shutdown?](https://github.com/wandb/client/issues/3045)
"""
def hyperparameter_search_sweep(
    n_trials: int, 
    init_opts: str, 
    wandb_hp_config: str, 
    project:str = None
):
    # 读取初始化训练参数
    opts = Argparser.parse_args_from_json(json_file=init_opts)
    opts.do_train = True; opts.do_eval = True; opts.do_predict = False
    output_sub_dir = f'{opts.task_name}_{opts.model_type}_{opts.experiment_code}'
    if not opts.output_dir.endswith(output_sub_dir):
        opts.output_dir = os.path.join(opts.output_dir, output_sub_dir)
    os.makedirs(opts.output_dir, exist_ok=True)
    
    # 读取超参搜索配置
    with open(wandb_hp_config, "r") as f: 
        config = yaml.load(f)

    # 定义优化目标
    def _objective(opts):

        # Access all hyperparameter values through wandb.config
        run = wandb.run if wandb.run else wandb.init()
        opts = vars(opts)
        job_opts = vars(wandb.config)["_items"]
        opts.update(**job_opts)
        opts = argparse.Namespace(**opts)
        
        trainer = main(opts)
        metric_value = trainer.records \
            ['result'][config["metric"]["name"]]
        return metric_value
    
    # 搜索超参数
    sweep_id = wandb.sweep(config, project=project)
    wandb.agent(sweep_id, function=lambda: _objective(opts), count=n_trials)


if __name__ == "__main__":
    main(build_opts())
    exit(0)

if __name__ == "__main__":
    n_trials = 10
    wandb_hp_config = "sweep/cadec-bayes-hyperband.yaml"
    init_opts = "sweep/cadec_roberta_roberta-base-span-lr1e-5-wd0.01-dropout0.5-span15-e15-bs16x1-sinusoidal-biaffine_opts.json"
    hyperparameter_search_sweep(
        n_trials=n_trials, 
        init_opts=init_opts,
        wandb_hp_config=wandb_hp_config,
        project="gaiic2022-track2",
    )
    