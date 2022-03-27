import os
import sys
import json
from typing import *
from argparse import ArgumentParser
from sklearn.model_selection import KFold

sys.path.append("TorchBlocks/")
from torchblocks.utils.seed import seed_everything
from torchblocks.metrics.sequence_labeling.scheme import get_scheme

def generate_examples(data_path):
    sentence_counter = 0
    with open(data_path, encoding="utf-8") as f:
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
            sentence = (
                sentence_counter,
                {
                    "id": str(sentence_counter),
                    "tokens": current_words,
                    "ner_tags": current_labels,
                },
            )
            sentence_counter += 1
            current_words = []
            current_labels = []
            yield sentence

    # if something remains:
    if current_words:
        sentence = (
            sentence_counter,
            {
                "id": str(sentence_counter),
                "tokens": current_words,
                "ner_tags": current_labels,
            },
        )
        yield sentence

def create_examples(data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
    examples = []
    get_entities = get_scheme("BIO")
    for (i, line) in enumerate(data):
        guid = f"{data_type}-{i}"
        tokens = line[1]["tokens"]
        entities = None
        if data_type != "test":
            entities = []
            for label, start, end in get_entities(line[1]["ner_tags"]):
                entities.append((start, end + 1, label, tokens[start: end + 1]))
        examples.append(dict(guid=guid, text=tokens, entities=entities, sent_start=0, sent_end=len(tokens)))
    return examples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--labeled_files", type=str, nargs="+", default=[
        "data/raw/train_data/train.txt",
    ])
    parser.add_argument("--test_files", type=str, nargs="+", default=[
        "data/raw/preliminary_test_a/word_per_line_preliminary_A.txt",
    ])
    parser.add_argument("--output_dir", type=str, default="data/processed/")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # prepare
    seed_everything(args.seed)
    args.output_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(args.output_dir, exist_ok=True)

    split train & dev
    labeled_examples = []
    for labeled_file in args.labeled_files:
        labeled_examples.extend(create_examples(generate_examples(labeled_file), "train"))
    kf = KFold(n_splits=args.n_splits, shuffle=True)
    for fold_no, (train_index, dev_index) in enumerate(kf.split(labeled_examples)):
        print(f"split={fold_no}, #train={len(train_index)}, #dev={len(dev_index)}")
        with open(os.path.join(args.output_dir, f"train.{fold_no}.jsonl"), "w") as f:
            for index in train_index:
                example = labeled_examples[index]
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        with open(os.path.join(args.output_dir, f"dev.{fold_no}.jsonl"), "w") as f:
            for index in dev_index:
                example = labeled_examples[index]
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    for test_file in args.test_files:
        test_examples = create_examples(generate_examples(test_file), "test")
        print(f"test file: {test_file}, #test={len(test_examples)}")
        basename, ext = os.path.splitext(os.path.basename(test_file))
        with open(os.path.join(args.output_dir, f"{basename}.jsonl"), "w") as f:
            for example in test_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        