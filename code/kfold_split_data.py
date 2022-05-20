import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import KFold
from packages import seed_everything

# FILE_PATH = '../data/contest_data/train_data/train.txt'
# UNLABLED_FILE_PATH = '../data/contest_data/train_data/unlabeled_train_data.txt'
# OUTPUT_PATH = '../data/tmp_data/10_folds_data/'
FILE_PATH = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data/train_data/train.txt'
UNLABLED_FILE_PATH = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data/train_data/unlabeled_train_data.txt'
OUTPUT_PATH = '/home/mw/temp/10_folds_data/'
SEED = 42
K_FOLDS = 10
NUM_UNLABELED = 40000
seed_everything(SEED)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

sentences = []
sentence_counter = 0
with open(FILE_PATH, encoding="utf-8") as f:
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
        sentence = {
                "id": str(sentence_counter),
                "tokens": current_words,
                "ner_tags": current_labels
            }
        sentence_counter += 1
        current_words = []
        current_labels = []
        sentences.append(sentence)

with open(os.path.join(OUTPUT_PATH, f"train.all.jsonl"), "w") as f:
    for example in sentences:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

kf = KFold(n_splits=K_FOLDS, shuffle=True)
for fold_no, (train_index, dev_index) in enumerate(kf.split(sentences)):
    print(f"split={fold_no}, #train={len(train_index)}, #dev={len(dev_index)}")
    with open(os.path.join(OUTPUT_PATH, f"train.{fold_no}.jsonl"), "w") as f:
        for index in train_index:
            example = sentences[index]
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    with open(os.path.join(OUTPUT_PATH, f"dev.{fold_no}.jsonl"), "w") as f:
        for index in dev_index:
            example = sentences[index]
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
 
 # 无标签数据
with open(UNLABLED_FILE_PATH, "r") as f:
    unlabeled_texts = [line.strip() for line in f.readlines() if len(line) < 128]

# class BM25(object):
#     def __init__(self, documents_list, k1=2, k2=1, b=0.5):
#         self.documents_list = documents_list
#         self.documents_number = len(documents_list)
#         self.avg_documents_len = sum([
#             len(document) for document in documents_list
#         ]) / self.documents_number
#         self.f = []
#         self.idf = {}
#         self.k1 = k1
#         self.k2 = k2
#         self.b = b
#         self.init()
 
#     def init(self):
#         df = {}
#         for document in tqdm(self.documents_list, total=len(self.documents_list)):
#             temp = {}
#             for word in document:
#                 temp[word] = temp.get(word, 0) + 1
#             self.f.append(temp)
#             for key in temp.keys():
#                 df[key] = df.get(key, 0) + 1
#         for key, value in df.items():
#             self.idf[key] = np.log(
#                 (self.documents_number - value + 0.5) / (value + 0.5)
#             )
 
#     def get_score(self, index, query):
#         score = 0.0
#         document_len = len(self.f[index])
#         qf = Counter(query)
#         for q in query:
#             if q not in self.f[index]:
#                 continue
#             score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
#                 self.f[index][q] + self.k1 * (
#                     1 - self.b + self.b * document_len / self.avg_documents_len
#                 ))) * (qf[q] * (self.k2 + 1) / (qf[q] + self.k2))
#         return score
 
#     def get_documents_score(self, query):
#         score_list = []
#         for i in range(self.documents_number):
#             score_list.append(self.get_score(i, query))
#         return score_list

# bm25 = BM25(unlabeled_texts, k1=2, k2=1, b=0.5)
# index2indices_map = dict()
# train_texts = ["".join(s["tokens"]) for s in sentences]
# for idx, text in tqdm(enumerate(train_texts), total=len(train_texts)):
#     score = torch.tensor(bm25.get_documents_score(text))
#     _, indices = torch.topk(score, k=3)
#     indices = indices.numpy().tolist()
#     index2indices_map[idx] = indices
#     # max([len(t) for t in unlabeled_texts])
#     # temp = [unlabeled_texts[i] for i in indices]
# indices = torch.tensor(list(index2indices_map.values())).view(-1)
# print(indices.size())
# indices = torch.unique(indices)
# print(indices.size())

np.random.shuffle(unlabeled_texts)
unlabeled_texts = unlabeled_texts[: NUM_UNLABELED]
print(f"#unlabeled={len(unlabeled_texts)}")
with open(os.path.join(OUTPUT_PATH, f"unlabeled.txt"), "w") as f:
    for text in unlabeled_texts:
        f.write(text + "\n")
