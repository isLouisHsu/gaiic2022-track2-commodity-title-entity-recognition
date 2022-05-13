# %%
import os
import json
from utils import LABEL2MEANING_MAP
from run_span_classification_v1 import (
    MODEL_CLASSES, 
    DATA_CLASSES, 
    load_dataset,
)

LABEL2MEANING_MAP = {k: v[:2] + v[4:] for k, v in LABEL2MEANING_MAP.items()}
LABEL2ID = {k: i for i, (k, v) in enumerate(sorted(LABEL2MEANING_MAP.items(), key=lambda x: int(x[0])))}
LABEL2MEANING_MAP["O"] = "非实体"

#%%
data_class, process_class = DATA_CLASSES["gaiic"]
config_class, model_class, tokenizer_class = MODEL_CLASSES["nezha"]
model_name_or_path = "../data/pretrain_model/nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2/checkpoint-100000/"
tokenizer = tokenizer_class.from_pretrained(os.path.join(model_name_or_path, "vocab.txt"), do_lower_case=True)
dataset = load_dataset(
    data_class=data_class, 
    process_class=process_class, 
    data_name="train.all.jsonl", 
    data_dir="../data/tmp_data/v3/", 
    data_type="train", 
    tokenizer=tokenizer, 
    max_sequence_length=512, 
    context_size=0, 
    max_span_length=35,
    negative_sampling=0.0,
)

#%%
tags = []
for example in dataset.examples:
    entities = sorted(example["entities"], key=lambda x: x[:2])
    tags.append([entity[2] for entity in entities])

#%%
from smoothnlp.algorithm.phrase.phrase_extraction import extract_phrase
from smoothnlp.algorithm.phrase.ngram_utils import sentence_split_by_punc

vocab = list(tokenizer.vocab.items())
strs = ["".join([vocab[int(i) + 1000][0] for i in x]) for x in tags]
phrases = extract_phrase(
    strs, 200, min_n=2, max_n=10, min_freq=10
)
labels = [
    [tokenizer.convert_tokens_to_ids(ch) - 1000 for ch in phrase]
    for phrase in phrases
]
"""
[[20, 20],
 [47, 47],
 [20, 21],
 [41, 41],
 [49, 49],
 [6, 6],
 [15, 1],
 [15, 15],
 [15, 1, 1],
 [20, 21, 36, 36, 31],
 [21, 36, 36, 36, 31],
 [41, 41, 41],
 [21, 36, 22, 36, 31],
 [21, 36, 36, 31],
 [20, 36, 36, 31],
 [20, 20, 21],
 [20, 20, 20],
 [20, 21, 21],
 [20, 36, 20],
 [31, 36, 31],
 [47, 47, 47],
 [19, 20],
 [31, 36, 21],
 [30, 22, 30],
 [49, 49, 49],
 [6, 6, 6],
 [34, 34],
 [31, 22, 31],
 [21, 21],
 [30, 30],
 [15, 1, 1, 2],
 [15, 15, 1],
 [21, 20],
 [31, 31],
 [31, 21],
 [52, 49],
 [34, 31],
 [47, 40, 47],
 [15, 1, 2],
 [49, 40, 47],
 [30, 31],
 [47, 41],
 [47, 49],
 [41, 47],
 [6, 4, 6],
 [49, 47],
 [6, 1, 1],
 [6, 1]]
"""
[
    [LABEL2MEANING_MAP[str(l)] for l in label]
    for label in labels
]
"""
[['配件-系列', '配件-系列'],
 ['其他-功能', '其他-功能'],
 ['配件-系列', '配件-型号'],
 ['其他-用途', '其他-用途'],
 ['其他-样式', '其他-样式'],
 ['主体-时间', '主体-时间'],
 ['主体-产地', '主体-品牌'],
 ['主体-产地', '主体-产地'],
 ['主体-产地', '主体-品牌', '主体-品牌'],
 ['配件-系列', '配件-型号', '配件-尺寸', '配件-尺寸', '配件-样式'],
 ['配件-型号', '配件-尺寸', '配件-尺寸', '配件-尺寸', '配件-样式'],
 ['其他-用途', '其他-用途', '其他-用途'],
 ['配件-型号', '配件-尺寸', '配件-名称', '配件-尺寸', '配件-样式'],
 ['配件-型号', '配件-尺寸', '配件-尺寸', '配件-样式'],
 ['配件-系列', '配件-尺寸', '配件-尺寸', '配件-样式'],
 ['配件-系列', '配件-系列', '配件-型号'],
 ['配件-系列', '配件-系列', '配件-系列'],
 ['配件-系列', '配件-型号', '配件-型号'],
 ['配件-系列', '配件-尺寸', '配件-系列'],
 ['配件-样式', '配件-尺寸', '配件-样式'],
 ['其他-功能', '其他-功能', '其他-功能'],
 ['配件-品牌', '配件-系列'],
 ['配件-样式', '配件-尺寸', '配件-型号'],
 ['配件-材料', '配件-名称', '配件-材料'],
 ['其他-样式', '其他-样式', '其他-样式'],
 ['主体-时间', '主体-时间', '主体-时间'],
 ['配件-颜色', '配件-颜色'],
 ['配件-样式', '配件-名称', '配件-样式'],
 ['配件-型号', '配件-型号'],
 ['配件-材料', '配件-材料'],
 ['主体-产地', '主体-品牌', '主体-品牌', '主体-系列'],
 ['主体-产地', '主体-产地', '主体-品牌'],
 ['配件-型号', '配件-系列'],
 ['配件-样式', '配件-样式'],
 ['配件-样式', '配件-型号'],
 ['其他-颜色', '其他-样式'],
 ['配件-颜色', '配件-样式'],
 ['其他-功能', '其他-名称', '其他-功能'],
 ['主体-产地', '主体-品牌', '主体-系列'],
 ['其他-样式', '其他-名称', '其他-功能'],
 ['配件-材料', '配件-样式'],
 ['其他-功能', '其他-用途'],
 ['其他-功能', '其他-样式'],
 ['其他-用途', '其他-功能'],
 ['主体-时间', '主体-名称', '主体-时间'],
 ['其他-样式', '其他-功能'],
 ['主体-时间', '主体-品牌', '主体-品牌'],
 ['主体-时间', '主体-品牌']]
"""
