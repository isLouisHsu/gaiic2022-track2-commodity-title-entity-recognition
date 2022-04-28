# %%
import os
import json
from utils import LABEL2MEANING_MAP
from run_span_classification_v1 import entities_to_ner_tags

LABEL2MEANING_MAP = {k: v[:2] + v[4:] for k, v in LABEL2MEANING_MAP.items()}
LABEL2ID = {k: i for i, (k, v) in enumerate(sorted(LABEL2MEANING_MAP.items(), key=lambda x: int(x[0])))}
LABEL2MEANING_MAP["O"] = "非实体"

#%% 错误分析
eval_path = "/home/xuyaobin/gaiic2022-track2-commodity-title-entity-recognition/data/model_data/20220428/gaiic_nezhaxy_nezhaxy-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs6x2-sinusoidal-biaffine-fgm1.0/checkpoint-eval_f1_micro_all_entity-best/evaluations.span.jsonl"
gt_path = "/home/xuyaobin/gaiic2022-track2-commodity-title-entity-recognition/data/model_data/20220428/gaiic_nezhaxy_nezhaxy-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs6x2-sinusoidal-biaffine-fgm1.0/checkpoint-eval_f1_micro_all_entity-best/groundtruths.span.jsonl"

with open(eval_path, "r") as f:
    eval_examples = [json.loads(line) for line in f.readlines()]
with open(gt_path, "r") as f:
    gt_examples = [json.loads(line) for line in f.readlines()]

eval_entities = []
for example in eval_examples:
    eval_entities.extend(sorted(example["entities"], key=lambda x: x[:2]))
gt_entities = []
for example in gt_examples:
    gt_entities.extend(sorted(example["entities"], key=lambda x: x[:2]))

#%%
with open(os.path.join(os.path.dirname(eval_path), "evaluations.csv"), "w") as f:
    for start, end, label, string in eval_entities:
        label = LABEL2MEANING_MAP[label]
        f.write(f"{start},{end},{label},{string}\n")
with open(os.path.join(os.path.dirname(gt_path), "groundtruths.csv"), "w") as f:
    for start, end, label, string in gt_entities:
        label = LABEL2MEANING_MAP[label]
        f.write(f"{start},{end},{label},{string}\n")

f = open(os.path.join(os.path.dirname(gt_path), "char_gt_eval.csv"), "w")
for gt_example, ev_example in zip(gt_examples, eval_examples):
    sequence_length = len(gt_example["text"])
    gt_ner_tags = entities_to_ner_tags(sequence_length, gt_example["entities"])
    ev_ner_tags = entities_to_ner_tags(sequence_length, ev_example["entities"])
    for char, gt_tag, ev_tag in zip(gt_example["text"], gt_ner_tags, ev_ner_tags):
        gt_tag = (gt_tag[:2] + LABEL2MEANING_MAP[gt_tag[2:]]) if gt_tag != "O" else gt_tag
        ev_tag = (ev_tag[:2] + LABEL2MEANING_MAP[ev_tag[2:]]) if ev_tag != "O" else ev_tag
        f.write(f"{'x' if gt_tag != ev_tag else ' '} {char} {gt_tag} {ev_tag}\n")
    f.write("\n")
f.close()
