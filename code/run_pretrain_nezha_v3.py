import os
import logging
import warnings
import random
import torch
import pdb
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from itertools import chain
from transformers import TrainingArguments, Trainer
from nezha.modeling_nezha import NeZhaConfig, NeZhaForMaskedLM
from transformers import BertTokenizerFast
from packages import seed_everything
from torch.utils.data import Dataset, DataLoader

import os
import time
import logging

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WANDB_DISABLED"] = "true"
# logger = logging.getLogger(__name__)
kongge_replace = ['\x08',
                  '\t',
                  ' ',
                  '\x7f',
                  '\xa0',
                  '\u2002',
                  '\u2006',
                  '\u200b',
                  '\u200d',
                  '\u3000',
                  '️',
                  '\ufeff']
kongge2str = '^'
# ---
# train_data_path = '../data/contest_data/train_data/train.txt'
# #unlabeled_train_data_path = '../data/contest_data/train_data/unlabeled_train_data.txt'
# #pre_test_a_data_path = '../data/contest_data/preliminary_test_a/sample_per_line_preliminary_A.txt'
# #pre_test_b_data_path = '../data/contest_data/preliminary_test_b/sample_per_line_preliminary_B.txt'
# pretrain_data_path = '../data/contest_data'
# pretrained_model_path = '../data/pretrain_model/checkpoint-33000'
# record_save_path = '../data/submission'
# ---
train_data_path = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data/train_data/train.txt'
# unlabeled_train_data_path = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data/train_data/unlabeled_train_data.txt'
# pre_test_a_data_path = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data/preliminary_test_a/sample_per_line_preliminary_A.txt'
# pre_test_b_data_path = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data/preliminary_test_b/sample_per_line_preliminary_B.txt'
pretrain_data_path = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data'
pretrained_model_path = '/home/mw/input/gmodel_pretrainv2_3600027674/'
record_save_path = '/home/mw/project/data/pretrain_model/nezha_pretrain_v3'
# ---
output_pretrain_data_path = os.path.join(record_save_path, 'pretrain_data.txt')
mlm_probability = 0.15
num_train_epochs = 50
seq_length = 128
batch_size = 128
learning_rate = 5e-5
save_steps = 600
ckpt_save_limit = 10
logging_steps = 200
seed = 42
opts = {
    'task_name': 'gaiic',
    'model_type': 'nezha',
    'experiment_code': 'v3',
    'output_dir': record_save_path,
    'do_pretrain': True,
}

seed_everything(seed)
os.makedirs(os.path.dirname(record_save_path), exist_ok=True)
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path)
model_config = NeZhaConfig.from_pretrained(pretrained_model_path)
model = NeZhaForMaskedLM.from_pretrained(pretrained_model_path, config=model_config)

train_sentences = []
sentence_counter = 0
with open(train_data_path, encoding="utf-8") as f:
    lines = f.readlines()
current_words = []
current_labels = []
for row in lines:
    row = row.rstrip("\n")
    if row != "":
        token, label = row[0], row[2:]
        current_words.append(token)
        current_labels.append(label)
    else:
        if not current_words:
            continue
        assert len(current_words) == len(current_labels), "word len doesn't match label length"
        sentence = "".join(current_words)
        for s in kongge_replace:
            sentence = sentence.replace(s, kongge2str)
        sentence_counter += 1
        current_words = []
        current_labels = []
        train_sentences.append(sentence)
print(f'train pretrain data : {len(train_sentences)}.')

#pre_test_a_sentences = []
#with open(pre_test_a_data_path, "r") as fr:
#    for sentence in fr:
#        sentence = sentence.strip("\n")
#        for s in kongge_replace:
#            sentence = sentence.replace(s, kongge2str)
#        pre_test_a_sentences.append(sentence)
#print(f'Test A pretrain data : {len(pre_test_a_sentences)}.')
#
#pre_test_b_sentences = []
#with open(pre_test_b_data_path, "r") as fr:
#    for sentence in fr:
#        sentence = sentence.strip("\n")
#        for s in kongge_replace:
#            sentence = sentence.replace(s, kongge2str)
#        pre_test_b_sentences.append(sentence)
#print(f'Test B pretrain data : {len(pre_test_b_sentences)}.')#
#
#unlabeled_train_sentences = []
#with open(unlabeled_train_data_path, "r") as fr:
#    for sentence in fr:
#        sentence = sentence.strip("\n")
#        for s in kongge_replace:
#            sentence = sentence.replace(s, kongge2str)
#        unlabeled_train_sentences.append(sentence)
#print(f'unlabeled train pretrain data : {len(unlabeled_train_sentences)}.')

#all_pretrain_sentences = train_sentences + pre_test_a_sentences + pre_test_b_sentences + unlabeled_train_sentences
#print(f'total pretrain data : {len(all_pretrain_sentences)}.')
#all_pretrain_sentences = list(set(all_pretrain_sentences))
#print(f'set pretrain data : {len(all_pretrain_sentences)}.')

#with open(output_pretrain_data_path, 'w', encoding='utf-8') as f:
#    for i in all_pretrain_sentences:
#        f.writelines(i + '\n')
#print(f'process data has been written to {output_pretrain_data_path}.')
all_pretrain_sentences = train_sentences
inputs = []
for row in tqdm(all_pretrain_sentences, desc='', total=len(all_pretrain_sentences)):
    sentence = row
    inputs_dict = tokenizer.encode_plus(sentence,
                                        add_special_tokens=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    input = [row, inputs_dict['input_ids'], inputs_dict['token_type_ids'], inputs_dict['attention_mask']]
    inputs.append(input)


class GaiicDataset(Dataset):
    def __init__(self, data: List):
        super(Dataset, self).__init__()
        self.data = data

    def __getitem__(self, index: int) -> tuple:
        item = (self.data[index][1],
                self.data[index][2],
                self.data[index][3])
        return item

    def __len__(self) -> int:
        return len(self.data)


class GaiicDataCollator:
    def __init__(self, max_seq_len: int, tokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def _ngram_mask(self, input_ids, max_seq_len):
        cand_indexes = []
        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])
        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))
        if len(input_ids) <= 32:
            max_ngram = 2
        else:
            max_ngram = 3
        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        np.random.shuffle(ngram_indexes)
        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue
            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            if len(covered_indexes) + len(index_set) > num_to_predict:
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

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def ngram_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:

        labels = inputs.clone()
        probability_matrix = mask_labels

        bs = inputs.shape[0]
        # word struct prediction
        for i in range(bs):
            tmp = []
            tmp_pro = []
            tmp_pro.extend([1] * 3)
            now_input = inputs[i]
            now_probability_matrix = probability_matrix[i]
            now_probability_matrix = now_probability_matrix.cpu().numpy().tolist()
            now_input = now_input.cpu().numpy().tolist()
            for j in range(len(now_input)):
                if now_input[j] == self.tokenizer.sep_token_id:
                    sep_index = j
            # we don't choose cls_ids, sep_ids, pad_ids
            choose_range = now_input[1:sep_index - 2]
            if len(choose_range) == 0:
                choose_range = now_input[1:5]
            rd_token = np.random.choice(choose_range)
            token_idx = now_input.index(rd_token)
            tmp.extend(now_input[token_idx:token_idx + 3])
            np.random.shuffle(tmp)
            now_input[token_idx:token_idx + 3] = tmp
            now_probability_matrix[token_idx:token_idx + 3] = tmp_pro
            now_input = torch.tensor(now_input)
            now_probability_matrix = torch.tensor(now_probability_matrix)
            inputs[i] = now_input
            probability_matrix[i] = now_probability_matrix
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        batch_mask = self.ngram_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }
        return data_dict

data_collator = GaiicDataCollator(max_seq_len=seq_length, tokenizer=tokenizer, mlm_probability=mlm_probability)
train_MLM_data = GaiicDataset(inputs)

training_args = TrainingArguments(
    output_dir=record_save_path,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    save_steps=save_steps,
    save_total_limit=ckpt_save_limit,
    logging_steps=logging_steps,
    seed=seed,
    gradient_accumulation_steps=2,
    learning_rate=learning_rate,
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_MLM_data,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(record_save_path)

