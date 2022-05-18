import os
import json
from sklearn.model_selection import KFold

# FILE_PATH = '../data/contest_data/train_data/train.txt'
# OUTPUT_PATH = '../data/tmp_data/10_folds_data/'
FILE_PATH = '/home/mw/input/gaiic_contest8627/gaiic2022_track2_contest_data/contest_data/train_data/train.txt'
OUTPUT_PATH = '/home/mw/project/data/tmp_data/10_folds_data/'
SEED = 42
K_FOLDS = 10

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


