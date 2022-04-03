import os
import re
import sys
import json
import yaml
import math
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import (
    BertConfig, BertPreTrainedModel, BertModel, BertTokenizerFast, BertTokenizer,
    RobertaConfig, RobertaPreTrainedModel, RobertaModel, RobertaTokenizerFast,
    PreTrainedModel, AdamW,
)
from transformers.file_utils import ModelOutput
from nezha.modeling_nezha import NeZhaPreTrainedModel, NeZhaModel

sys.path.append("TorchBlocks/")
from torchblocks.callback import ProgressBar
# from torchblocks.data.dataset import DatasetBase
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
            str(i) for i in range(55) if i not in [0, 27, 45]
        ]

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
        tokens = [ch if ch in self.tokenizer.vocab else 
            self.tokenizer.unk_token for ch in text]
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="np",)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        offset_mapping = [[i, i + 1] for i, ch in enumerate(tokens)]
        return tokens, offset_mapping


class ProcessConvertLevel(ProcessBase):

    def __init__(self, tokenizer, conversion, lang="en"):
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
        converted, entities = self.convert_func(
            example["text"], example["entities"])
        if "sent_start" in example and "sent_end" in example:
            sent_start, sent_end = example["sent_start"], example["sent_end"]
            sentence = [[(sent_start, sent_end, "_", example["text"][sent_start: sent_end])]]
            _, sentence = self.convert_func(example["text"], sentence)
            example["sent_start"], example["sent_end"] = sentence[0][0][:2]
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


class ProcessPreprocess(ProcessBase):
    pass    # TODO: 全半角转换等


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

        # encode spans
        skip_indices = [idx for idx, token in enumerate(tokens) if token in self.tokenizer.all_special_tokens_extended]
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
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_sequence_length,
            is_split_into_words=True,
            return_tensors="pt",
        )
        input_length = inputs["attention_mask"].sum().item() - 2
        pad_length = self.max_sequence_length - input_length - 2 - 1
        inputs["offset_mapping"] = torch.tensor([
            [(0, 0), ] + \
            [(i, i + 1) for i in range(input_length)] + \
            [(0, 0) for i in range(pad_length)]
        ])
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
        ProcessConvertLevel(tokenizer, "word2char") if data_class in [  # english

        ] else None,
        process_class(
            data_class.label2id(), tokenizer, max_sequence_length,
            max_span_length, negative_sampling, stanza_nlp,
        ),
    ]
    return data_class(data_name, data_dir, data_type, process_piplines, 
        context_size=context_size, use_cache=False, **kwargs)


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

    def forward_endpoint(self, sequence_output, spans):

        spans_start_embedding, spans_end_embedding = \
            self._extract_spans_embedding_endpoint(sequence_output, spans)
        spans_width_embedding = self._extract_width_embedding(spans)
        
        spans_embedding = torch.cat([
            spans_start_embedding, 
            spans_end_embedding,
            spans_width_embedding,
        ], dim=-1)  # (batch_size, num_spans, num_features)

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
            # probas, labels = logits_or_labels.softmax(dim=-1).max(dim=-1)           # (batch_size, sequence_length)
            # labels = torch.where((probas < thresh) | (labels == IGNORE_INDEX), 
            #     torch.full_like(labels, other_id), labels)
            probas = logits_or_labels.softmax(dim=-1)
            probas[..., other_id] = torch.where(probas[..., other_id] < thresh, 
                torch.zeros_like(probas[..., other_id]), probas[..., other_id])
            _, labels = probas.max(dim=-1)
            probas = 1 - probas[..., other_id]  # 是实体的概率
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

    def forward(
        self,
        logits,  # (batch_size, num_spans, num_labels)
        labels,  # (batch_size, num_spans,)
        mask,    # (batch_size, num_spans,)
    ):
        num_labels = logits.size(-1)
        loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1))
        activate_mask = mask.view(-1) == 1
        loss = loss[activate_mask]
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
        
        self.head = SpanClassificationHead(
            config.hidden_size, config.num_labels,
            config.max_span_length, config.width_embedding_size,
            config.do_projection, config.do_cln, config.do_biaffine,
            config.do_co_attention, config.extract_method,
        )

        self.loss_fct = SpanClassificationLoss(
            num_labels=config.num_labels, 
            loss_type=config.loss_type,
            label_smoothing_eps=config.label_smoothing,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            reduction="mean",
            ignore_index=IGNORE_INDEX,
        )

        if config.do_rdrop:
            self.rdrop_loss_func = SpanClassificationRDropLoss()

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
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # self.base_model.encoder(outputs["last_hidden_state"], attention_mask) # AlBert Style

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

        sequence_output = self.dropout(sequence_output)
        logits = self.head(sequence_output, spans)          # (batch_size, num_spans, num_labels)
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
        return SpanClassificationHead.decode(logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits)


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


def precision_recall_fscore_support(y_true: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    y_pred: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    *,
                                    average: Optional[str] = None,
                                    labels: Optional[List[str]] = None,
                                    entity_type: str = "all",
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

    def __init__(self, labels, average="micro", entity_type="all"):
        super().__init__(labels, average)
        self.entity_type = entity_type

    def value(self):
        columns = ["label", "precision", "recall", "f1", "support"]
        values = []

        for label in [self.average] + sorted(self.labels):
            p, r, f, s = precision_recall_fscore_support(
                self.target, self.preds, average=self.average,
                labels=None if label == self.average else [label],
                entity_type=self.entity_type
            )
            values.append([label, p, r, f, s])
        df = pd.DataFrame(values, columns=columns)
        f1 = df[df['label'] == self.average]['f1'].item()
        return {
            f"df_{self.name()}": df, f"f1_{self.average}_{self.name()}": f1,   # for monitor
        }

    def name(self):
        return f"{self.entity_type}_entity"


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

    def evaluate(self, dev_data, prefix_metric=None, save_dir=None, save_result=False, file_name=None):
        '''
        Evaluate the model on a validation set
        '''
        all_batch_list = []
        eval_dataloader = self.build_eval_dataloader(dev_data)
        self.build_record_tracker()
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, batch in enumerate(eval_dataloader):
            batch = self.predict_forward(batch)
            if batch.get('loss'):
                self.records['loss_meter'].update(batch['loss'], n=1)

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
                    if not isinstance(metric, SequenceLabelingScoreEntity):
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

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSpanClassification, BertTokenizerZh),
    "nezha": (BertConfig, NeZhaForSpanClassification, BertTokenizerZh),
}

DATA_CLASSES = {
    "gaiic": (GaiicTrack2SpanClassificationDataset, GaiicTrack2ProcessExample2Feature),
}


def build_opts():
    # sys.argv.append("outputs/gaiic_nezha_nezha-cn-base-spanv1-datav2-lr5e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0/gaiic_nezha_nezha-cn-base-spanv1-datav2-lr5e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0_opts.json")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        opts = Argparser.parse_args_from_json(json_file=os.path.abspath(sys.argv[1]))
    else:
        parser = Argparser.get_training_parser()
        group = parser.add_argument_group(title="user-defined", description="user-defined")
        group.add_argument("--do_check", action="store_true")
        group.add_argument("--labels", nargs="+", type=str, default=None)
        group.add_argument("--max_train_examples", type=int, default=None)
        group.add_argument("--max_eval_examples", type=int, default=None)
        group.add_argument("--max_test_examples", type=int, default=None)
        group.add_argument("--context_size", type=int, default=0)
        group.add_argument("--classifier_dropout", type=float, default=0.1)
        group.add_argument("--use_last_n_layers", type=int, default=None)
        group.add_argument("--agg_last_n_layers", type=str, default="mean")
        group.add_argument("--negative_sampling", type=float, default=0.0)
        group.add_argument("--max_span_length", type=int, default=30)
        group.add_argument("--width_embedding_size", type=int, default=128)
        group.add_argument("--loss_type", type=str, default="lsr", choices=["ce", "lsr", "focal"])
        group.add_argument("--label_smoothing", type=float, default=0.0)
        group.add_argument("--focal_gamma", type=float, default=2.0)
        group.add_argument("--focal_alpha", type=float, default=0.25)
        group.add_argument("--do_rdrop", action="store_true")
        group.add_argument("--rdrop_weight", type=float, default=0.3)
        group.add_argument("--decode_thresh", type=float, default=0.0)
        group.add_argument("--extract_method", type=str, default="endpoint")
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
        opts = parser.parse_args_from_parser(parser)

    # TODO: for debug
    # opts.do_train = opts.do_eval = opts.do_predict = False
    # opts.do_eval = True
    # opts.do_check = True
    # opts.context_size = 10
    # opts.max_span_length = 50
    # opts.train_max_seq_length = 512
    # opts.do_lstm = False
    # opts.device_id = 'cpu'
    # opts.do_eval = opts.do_predict = opts.do_check = False
    # opts.do_train = True
    # opts.use_syntactic = True
    # opts.syntactic_upos_size = 21
    
    return opts

def update_example_entities(tokenizer, examples, entities, process):
    assert len(examples) == len(entities)
    updated = []
    for example, entities in zip(examples, entities):
        example = deepcopy(example)
        if process is not None:
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
        stanza_nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos,lemma,depparse", use_gpu=True)
        stanza_nlp.processors["tokenize"].config.update({"pretokenized": True})
    try:
        tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    except AssertionError:
        # XXX: AssertionError: Config has to be initialized with question_encoder and generator config
        tokenizer = tokenizer_class.from_pretrained(os.path.join(opts.pretrained_model_path, "vocab.txt"), do_lower_case=opts.do_lower_case)
    if opts.do_train or opts.do_check:
        train_dataset = load_dataset(data_class, process_class, opts.train_input_file, opts.data_dir, "train",
                                    tokenizer, opts.train_max_seq_length, opts.context_size, opts.max_span_length, 
                                    opts.negative_sampling, stanza_nlp=stanza_nlp, labels=opts.labels, max_examples=opts.max_train_examples)
    if opts.do_train or opts.do_eval:
        dev_dataset   = load_dataset(data_class, process_class, opts.eval_input_file, opts.data_dir, "dev",
                                    tokenizer, opts.eval_max_seq_length, opts.context_size, opts.max_span_length,
                                    opts.negative_sampling, stanza_nlp=stanza_nlp, labels=opts.labels, max_examples=opts.max_eval_examples)
    if opts.do_predict:
        test_dataset  = load_dataset(data_class, process_class, opts.test_input_file, opts.data_dir, "test",
                                    tokenizer, opts.test_max_seq_length, opts.context_size, opts.max_span_length,
                                    opts.negative_sampling, stanza_nlp=stanza_nlp, labels=opts.labels, max_examples=opts.max_test_examples)
    if stanza_nlp is not None and train_dataset.use_cache and dev_dataset.use_cache and test_dataset.use_cache:
        del stanza_nlp
    opts.label2id = data_class.label2id()
    opts.id2label = data_class.id2label()
    opts.num_labels = len(opts.label2id)

    # model
    logger.info("initializing model and config")
    config, unused_kwargs = config_class.from_pretrained(
        opts.pretrained_model_path, return_unused_kwargs=True,
        num_labels=opts.num_labels, id2label=opts.id2label, label2id=opts.label2id,
        conditional=False,
        classifier_dropout=opts.classifier_dropout, 
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
        SequenceLabelingScoreEntity({"_"}, "micro", entity_type="without_label"),
    ]
    trainer = Trainer(opts=opts, model=model, metrics=metrics, logger=logger)

    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={"vocab": tokenizer})

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
            examples = update_example_entities(tokenizer, dev_dataset.examples, entities, dev_dataset.process_piplines[0])
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
            examples = update_example_entities(tokenizer, dev_dataset.examples, entities, dev_dataset.process_piplines[0])
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
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.predict(test_data=test_dataset, save_result=True, save_dir=prefix)

            # 保存为样本，用于分析
            results = load_pickle(os.path.join(checkpoint, f"test_predict_results.pkl"))
            entities = list(chain(*[batch["predictions"] for batch in results]))
            examples = update_example_entities(tokenizer, test_dataset.examples, entities, test_dataset.process_piplines[0])
            # with open(os.path.join(checkpoint, "predictions.json"), "w") as f:
            #     for example in examples:
            #         f.write(json.dumps(example, ensure_ascii=False) + "\n")
            # 保存为提交格式
            with open(os.path.join(checkpoint, "predictions.txt"), "w") as f:
                for example_no, example in enumerate(examples):
                    tokens = example["text"]
                    ner_tags = entities_to_ner_tags(len(example["text"]), example["entities"])
                    for token, tag in zip(tokens, ner_tags):
                        f.write(f"{token} {tag}\n")

    if opts.do_check:
        # check dataset & decode function
        span_labels = []
        sequence_lengths = []
        span_lengths = []
        metric = SequenceLabelingScoreEntity({label.split("-")[1] for label in \
            data_class.get_labels() if label not in ["O", "II-O"]}, "micro", entity_type="all")
        char2token = ProcessConvertLevel(tokenizer, "char2token")
        for example_no in tqdm(range(len(train_dataset)), total=len(train_dataset)):

            # print(example_no)
            # if example_no not in [8, ]: continue

            example, feature = train_dataset.examples[example_no], train_dataset[example_no]
            span_labels.extend(feature["labels"].cpu().numpy().tolist())
            text = example["text"]
            batch = train_dataset.collate_fn([feature])
            sequence_lengths.append(feature["attention_mask"].sum().item())

            # 真实标签
            for proc in train_dataset.process_piplines[:-1]:
                if proc is None: continue
                example = proc(example)                 # en: char level, zh: word level
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
            for span in decodes:
                for start, end, *_ in span:
                    span_lengths.append(end - start)
            # token -> char
            example_codec = deepcopy(example)
            entities_codec = [
                [(start, end, label, tokens[start: end]) for start, end, label in entity
            ] for entity in decodes]
            example_codec["entities"] = entities_codec                                  # token level
            decodes = example_codec["entities"]

            # sort_key = lambda x: (x[0][0], x[0][1] - x[0][0])   # 以每个实体第一个片段位置
            sort_key = lambda xs: list(chain(*[(x[0], x[1]) for x in xs]))  # 以实体所有片段位置
            entities = sorted(entities, key=sort_key)
            decodes = sorted(decodes, key=sort_key)
            # 批次为1
            predictions = [
                [[(start, end, label) 
                for start, end, label, _ in entity] for entity in decodes]
            ]
            groundtruths = [
                [[(start - example["sent_start"], end - example["sent_start"], label) 
                for start, end, label, _ in entity] for entity in entities]
            ]
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
    