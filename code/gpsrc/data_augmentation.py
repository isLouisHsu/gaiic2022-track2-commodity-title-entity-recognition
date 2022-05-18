from  copy import deepcopy
import numpy as np
import json
import os
from utils import get_entity_biob
from collections import defaultdict
input_file = '../data/tmp_data/10_folds_data/train.0.jsonl'
OUTPUT_PATH = '../data/tmp_data/10_folds_data/'
lines = []
with open(input_file) as fr:
    for line in fr.readlines():
        line = json.loads(line)
        lines.append(line)

def _shuffle_same_label_entities(entities):
    label2entities_map = defaultdict(list)
    for index, entity in enumerate(entities):
        label2entities_map[entity[2]].append((index, entity))
    idx2entity_map = dict()
    for label, indexed_entities in label2entities_map.items():
        indices = list(range(len(indexed_entities)))
        np.random.shuffle(indices)
        shffled_entities = [indexed_entities[index][1] for index in indices]
        for (idx, entity), shffled_entity in zip(indexed_entities, shffled_entities):
            start, end, label, string = entity
            shffled_start, shffled_end, shffled_label, shffled_string = shffled_entity
            idx2entity_map[idx] = [start, start + len(shffled_string), label, shffled_string]
    return idx2entity_map

def _replace_entities(text, entities, idx2entity_map):
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

new_examples = []
for example in lines:
    labels = example['ner_tags']
    text = example['tokens']
    entities = []
    for _type, _start_idx, _end_idx in get_entity_biob(labels, None):
        entities.append([_start_idx,_end_idx+1,_type,text[_start_idx:_end_idx + 1]])
    idx2entity_map = _shuffle_same_label_entities(entities)
    _text, _entities, idx2entity_map = _replace_entities(
        text, entities, idx2entity_map)
    new_example = {}
    new_example["tokens"] = _text
    new_example["entities"] = [[x[0],x[1]-1,x[2],x[3]] for x in _entities]
    new_examples.append(new_example)

with open(os.path.join(OUTPUT_PATH, f"train.0.aug.jsonl"), "w") as f:
    for index in new_examples:
        f.write(json.dumps(index, ensure_ascii=False) + "\n")

