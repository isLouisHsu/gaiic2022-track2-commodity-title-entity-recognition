import os
import argparse
from transformers import BertConfig, BertForMaskedLM
from tokenization_bert_zh import BertTokenizerZh
from nezha.modeling_nezha import NeZhaForMaskedLM
from run_chinese_ref import _is_chinese_char

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizerZh),
    "nezha": (BertConfig, NeZhaForMaskedLM, BertTokenizerZh),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
model = model_class.from_pretrained(args.model_name_or_path)

old2new_token_map = {
    token: "##" + token
    for token in tokenizer.vocab
    if len(token) == 1 and _is_chinese_char(ord(token))
}

num_added_tokens = tokenizer.add_tokens(list(old2new_token_map.values()))
model.resize_token_embeddings(len(tokenizer))
print(f"Add {num_added_tokens} tokens")

old2new_id_map = {
    tokenizer.convert_tokens_to_ids(old): \
        tokenizer.convert_tokens_to_ids(new)
    for old, new in old2new_token_map.items()
}

for old_id, new_id in old2new_id_map.items():
    model.base_model.embeddings.word_embeddings.weight[new_id] = \
        model.base_model.embeddings.word_embeddings.weight[old_id]

tokenizer.save_pretrained(args.save_path)
model.save_pretrained(args.save_path)
print(f"Model saved in {args.save_path}")
