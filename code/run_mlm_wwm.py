# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) with whole word masking on a
text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import os
import sys
import json
import math
import jieba
import torch
import random
import logging
import warnings
from typing import *
import numpy as np
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import (
    get_last_checkpoint, 
    is_main_process,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import (
    _torch_collate_batch,
    tolist,
)

from transformers import BertConfig, BertTokenizer, BertTokenizerFast
from nezha.modeling_nezha import NeZhaForMaskedLM
from tokenization_bert_zh import BertTokenizerZh
from run_chinese_ref import is_chinese

logger = logging.getLogger(__name__)
# MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_CLASSES = {
    "default": (AutoConfig, AutoModelForMaskedLM, AutoTokenizer),
    "nezha": (BertConfig, NeZhaForMaskedLM, BertTokenizerZh),
}
MODEL_TYPES = tuple(MODEL_CLASSES.keys())

WORD_SYNONYMS_MAP_FILE = "../data/tmp_data/word_synonyms_map.wv.json"
word_synonyms_map = None    # 单例
def get_synonym(word, default=None, invariable_length=False):
    global word_synonyms_map
    if word_synonyms_map is None:
        logging.info(f"Initializing word_synonyms_map...")
        with open(WORD_SYNONYMS_MAP_FILE, "r") as f:
            word_synonyms_map = json.load(f)
    synonyms = word_synonyms_map.get(word, [])
    if invariable_length:
        synonyms = [
            [synonym, score] for synonym, score in synonyms 
            if len(synonym) == len(word) and is_chinese(synonym)
        ]
    if len(synonyms) == 0:
        if default: return default
        return None
    synonym = random.choice(synonyms)[0]
    return synonym

@dataclass
class DataCollatorForNGramWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    .. note::

        This collator relies on details of the implementation of subword tokenization by
        :class:`~transformers.BertTokenizer`, specifically that subword tokens are prefixed with `##`. For tokenizers
        that do not adhere to this scheme, this collator will produce an output that is roughly equivalent to
        :class:`.DataCollatorForLanguageModeling`.
    """
    max_ngram: int = 1
    mlm_as_correction: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.example_count = 0
        self.chinese_tokens = [
            k for k in self.tokenizer.vocab.keys() if is_chinese(k)
        ]

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]
        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        batch_synonyms = None
        if self.mlm_as_correction:
            batch_synonyms = []
            for input_ids in batch_input:
                try:
                    input_length = torch.sum(input_ids != self.tokenizer.pad_token_id)
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                    tokens = tokens[: input_length][1: -1]
                    string = self.tokenizer.convert_tokens_to_string(tokens)
                    words = jieba.lcut(string)
                    synonyms = [self._synonym(word) for word in words]
                    synonym_ids = self.tokenizer(
                        "".join(synonyms),
                        padding="max_length", 
                        truncation=True, 
                        max_length=input_ids.size(-1),
                        return_tensors="pt"
                    )["input_ids"][0]
                    assert synonym_ids.size(0) == input_ids.size(0) and \
                        torch.sum(synonym_ids != self.tokenizer.pad_token_id) == input_length
                except Exception as e:
                    logger.info(e)
                    logger.info(f"MLM as correction Error, use `random_ids` as `synonym_ids`")
                    synonym_ids = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
                batch_synonyms.append(synonym_ids)
            batch_synonyms = torch.stack(batch_synonyms, dim=0)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if (not getattr(self.tokenizer, "do_ref_tokenize", False)) and ("chinese_ref" in e):
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask, batch_synonyms)

        if self.example_count < 5: # for debug
            for a, b in zip(inputs, labels):
                a = self.tokenizer.convert_ids_to_tokens(a)
                b = self.tokenizer.convert_ids_to_tokens(b)
                s = f"#{self.example_count}\t" + " ".join(f"{i}/{j if j != self.tokenizer.unk_token else '_'}" for i, j in zip(a, b))
                logger.info(s)
                self.example_count += 1
                if self.example_count > 5:
                    break

        return {"input_ids": inputs, "labels": labels}

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any, candidates: Any = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        if candidates is not None:
            inputs[indices_replaced] = candidates[indices_replaced]    # for MacBERT
        else:
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForNGramWholeWordMask is only suitable for BertTokenizer-like tokenizers."
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        # prepare
        ngrams = np.arange(1, self.max_ngram + 1, dtype=np.int64)
        # pvals = 1. / np.arange(1, self.max_ngram + 1)   # SpanBERT
        pvals = np.arange(self.max_ngram, 0, -1) * 1.   # MacBERT
        pvals /= pvals.sum(keepdims=True)
        cand_ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            cand_ngram_indexes.append(ngram_index)
        np.random.shuffle(cand_ngram_indexes)

        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for ngram_index_set in cand_ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # Choose ngram to mask
            if not ngram_index_set:
                continue
            for index_set in ngram_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue
            n = np.random.choice(
                ngrams[: len(ngram_index_set)],
                p=pvals[: len(ngram_index_set)] / \
                  pvals[: len(ngram_index_set)].sum(keepdims=True)
            )
            index_set = sum(ngram_index_set[n - 1], [])
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(ngram_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _synonym(self, word):
        if not is_chinese(word):
            return word
        # 尝试获取同义词，若不存在同义词，则返回随机
        synonym = get_synonym(word, invariable_length=True)
        if synonym is None:
            synonym = "".join(np.random.choice(
                self.chinese_tokens, size=len(word)))
        assert len(synonym) == len(word)
        return synonym

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    do_ref_tokenize: bool = field(
        default=False,
        metadata={
            "help": "For `~tokenization_bert_zh.BertTokenizerZh`"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    max_ngram: int = field(
        default=1, metadata={"help": "For n-gram mask."}
    )
    mlm_as_correction: bool = field(
        default=False, 
        metadata={"help": "For MacBERT"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def add_chinese_references(dataset, ref_file):
    with open(ref_file, "r", encoding="utf-8") as f:
        refs = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    assert len(dataset) == len(refs)

    dataset_dict = {c: dataset[c] for c in dataset.column_names}
    dataset_dict["chinese_ref"] = refs
    return Dataset.from_dict(dataset_dict)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_args.model_type = "default" if model_args.model_type is None else model_args.model_type
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = config_class.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = config_class.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "do_ref_tokenize": model_args.do_ref_tokenize,
    }
    if model_args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        try:
            tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        except AssertionError:
            # XXX: AssertionError: Config has to be initialized with question_encoder and generator config
            tokenizer = tokenizer_class.from_pretrained(
                os.path.join(model_args.model_name_or_path, "vocab.txt"), **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(examples["text"], padding=padding, truncation=True, max_length=data_args.max_seq_length)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Add the chinese references if provided
    if data_args.train_ref_file is not None:
        tokenized_datasets["train"] = add_chinese_references(
            tokenized_datasets["train"], data_args.train_ref_file
        )
    if data_args.validation_ref_file is not None:
        tokenized_datasets["validation"] = add_chinese_references(
            tokenized_datasets["validation"], data_args.validation_ref_file
        )
    # If we have ref files, need to avoid it removed by trainer
    has_ref = data_args.train_ref_file or data_args.validation_ref_file
    if has_ref:
        training_args.remove_unused_columns = False

    # Data collator
    # This one will take care of randomly masking the tokens.
    # data_collator = DataCollatorForWholeWordMask(
    #     tokenizer=tokenizer, 
    #     mlm_probability=data_args.mlm_probability,
    # )
    data_collator = DataCollatorForNGramWholeWordMask(
        tokenizer=tokenizer, 
        mlm_probability=data_args.mlm_probability,
        max_ngram=data_args.max_ngram,
        mlm_as_correction=data_args.mlm_as_correction,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm_wwm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
    exit(0)

if __name__ == "__main__":
    import os
    from run_chinese_ref import prepare_ref
    from tokenization_bert_zh import BertTokenizerZh

    corpus = \
        """
        OPPO闪充充电器 X9070 X9077 R5 快充头通用手机数据线 套餐【2.4充电头+数据线 】 安卓 1.5m
        OWIN净水器家用厨房欧恩科技反渗透纯水机O-50CSC
        教学教具磁条贴磁性条背胶(3M)软磁条 黑板软磁铁贴吸铁石磁贴片 宽50x厚1.5mm 1件=2米
        【限时促销】笔记本文具创意复古古风本子线装记事本学生小清新彩页日记本 寒江吟
        20本装a5笔记本子文具学生B5软抄本子记事本日记本软面抄批发简约加厚商务工作大学生笔记本办公用品手 大号B5-80张【10本装】随机颜色
        小米黑鲨2代钢化膜2pro电竞游戏专用二代手机保护膜全屏磨砂贴膜3代por黑沙2p防指纹抗蓝光玻璃防黑鲨2代Pro-KPL游戏膜【超清款】单片装4
        新款蜡烛香薰机家用迷你加湿器智能家居 磨砂灰 英规适配器
        冷藏展示柜保鲜柜立式单门双门商用冰柜超市冰箱冷柜饮料柜 三门豪华铝合金款（风冷无霜1200GLF）
        新科广场舞播放器音响户外大音量便携式小型手提拉杆移动蓝牙音箱低音炮大功率超大带无线话筒演出家用K歌 M100户外音响+有线话筒+16G优盘
        听雨轩80支装中性笔芯0.5mm全针管水笔芯0.38学生用考试黑色笔芯女签字替芯办公教师碳素蓝色水文 80支经典(蓝色0.5mm)
        神舟战神Z8-CR7N1游戏笔记本外壳保护贴膜15.6英寸电脑机身炫彩贴纸Z8-CT7N DIY来图定制/发图给客服/备注图代号 ABC面+磨砂防反光屏幕膜+键盘膜
        惠普暗影精灵4键盘膜15.6寸笔记本键盘贴纸按键贴光影精灵4代pro plus 2/3代银河舰队畅游 暗影精灵4代pro （透明轻薄专用TPU键盘膜）
        2020新版考研答题卡英语一二数学一二政治联考答题卡纸 自命题(课)B卡
        龙视安500万poe监控器设备套装高清网络摄像头一体机商用室外工厂 无 20路
        爱普生（EPSON）投影仪 办公商务 高清高亮度 大型工程投影机 CB-G7800（8000流明 标清XGA）官方标配
        MECHENA5可外放mp3mp4音乐播放器学生随身听女生款超薄便携式p3插卡有屏p4学英语mp6深空灰8G外放版+蓝牙
        韩版创意文具糖果包装点心蛋糕造型橡皮擦幼儿园节日奖品开学礼物 1641糖果袋橡皮
        1.5米德国进口榉木画架木制支架式画板架写生油画架广告架1.7实木素描架套装楼盘木质展览架展示架多功 1.5米榉木画架+2K画板
        小米手环2/3/4腕带替换带3代4代金属手环带链式不锈钢防水智能运动真皮表带二代手环4NFC版男女款 【三珠加强款-黑色】 小米手环2腕带
        20本餐馆餐饮饭店点菜单一联单层二联三联点菜本无碳复写开单本点单本不复写2联3联加饭店菜单记账单 (20本常规)1联/每本40张/不复写
        长虹同款3节能省电大厦式取暖器家用宿舍立式摇头电暖器 大厦2
        毛笔字帖水写布文房四宝毛笔套装水写字帖书法练习仿宣纸初学者毛笔高档礼品文具套装小学生成人书法字帖精 中国红套装
        4本加厚笔记本子学生课堂笔记本日记本作业本记事本子韩版小清新好看的本子女生款少女心 A5线圈4本/60张/樱花之夏
        Sonoely 手机壳360°全包防摔PC保护套薄外壳 适用于vivox9s/X20/X21/R15 深邃蓝 华为P20
        Choseal秋叶原qs8113 hdmi线高清线2.0版数据线4k高清线电脑电视数据连接线3D电视入门级 12米
        索歌（SUOGE）鸭脖保鲜柜风冷鸭货展示柜冷藏熟食柜水果卤菜凉菜冒菜烧烤展示柜卧式商用大理石点菜柜 大理石直角【风冷款】【送除雾加湿器】1.2米【进口丹佛斯压缩机】
        华为畅享5S手机壳TAG-AL00保护套tag-t100软胶cloo防摔TL创意指环-保时捷【挂绳+指环】
        显示器屏增高架台式电脑办公桌面收纳底座托架抽屉创意置物架子竹 【小鹿2号爱心单抽】雕刻增高架【爱心抽屉】
        金正看戏机老人唱戏机高清多功能大屏幕老年广场舞视频播放器收音 19机皇护眼版+送32GU盘+送老人视频资源
        笔筒 简约ins北欧风金属笔筒铁艺玫瑰金笔筒 学生桌面文具收纳 办公室桌面摆件化妆品收纳桶 圆柱镂空 四方形铁艺笔筒 -浅金色
        圆珠笔0.7mm蓝色单支装办公学生文具用品原子笔芯按动油笔经典简约批发 60只装（笔杆颜色随机）送笔筒
        【非原厂物料】苹果iPhone华为手机电池屏幕上门/到店安装后壳摄像头尾插按键更换服务ipad换电池手机屏幕上门安装服务
        紫光唱片车载车用CD光盘黑胶CD-R音乐无损碟片空白黑胶mp3刻录盘紫光光碟黑胶片CD黑碟空白光盘空音乐风 CD5 片 + 光盘袋5 个 + 黑
        2020新年华为mate30手机壳女mate30Pro玻璃防摔硅胶全包个性情侣款鼠年本命年暴富手机mate30【好运连连+玻璃】送钢化膜+挂绳
        坚果 JmGO SU（含88英寸硬屏套餐） 4K激光电视投影仪 投影机家用 （4K超高清 2400ANSI流明 杜比音响） 
        安卓王者荣耀走位利器游戏手柄手机吃鸡遥控辅助器A9绝地火线 Type-C版【1条装】+供电线 其他
        日照鑫 镭射玫瑰和纸胶带小清新diy手账胶带日记相册装饰贴纸 5个装 圆形烫玫瑰金
        纠错本子考研大学生初高中生纠正科目简约错题集 橙色4本装 胡萝卜北系列
        """
    lines = [
        line.strip() for line in corpus.split("\n") if len(line.strip()) > 0
    ]

    model_name_or_path = "/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base"
    vocab_file = os.path.join(model_name_or_path, "vocab.txt")
    tokenizer = BertTokenizerZh.from_pretrained(vocab_file, do_ref_tokenize=False)
    seg_res, ref_ids = prepare_ref(lines, None, tokenizer)

    tokenizer = BertTokenizerZh.from_pretrained(vocab_file, do_ref_tokenize=False)
    examples = [
        tokenizer(line, padding=False, truncation=True, max_length=128)
        for line in lines
    ]
    for example, refs in zip(examples, ref_ids):
        example["chinese_ref"] = refs

    data_collator = DataCollatorForNGramWholeWordMask(
        tokenizer=tokenizer, 
        mlm_probability=0.15, 
        max_ngram=4, 
        mlm_as_correction=True,
    )
    data_collator(examples)
