import os
import json

# TRAIN_PATH = "../data/tmp_data/stage2-gp/train.0.jsonl"
# SEMI_PATH = "../data/model_data/gaiic_nezha_experiment_bert_base_fold0_gp_v2_pre_v73/checkpoint-0.81642-22500/unlabeled_results.txt"
# OUTPUT_PATH = '../data/tmp_data/stage2-gp/'

with open(TRAIN_PATH, "r") as f:
    examples = [json.loads(line) for line in f.readlines()]

sentences = []
sentence_counter = len(examples)
with open(SEMI_PATH, encoding="utf-8") as f:
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

examples.extend(sentence)
with open(os.path.join(OUTPUT_PATH, f"train.0.semi.jsonl"), "w") as f:
    for example in sentences:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")