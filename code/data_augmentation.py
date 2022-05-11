import os
import copy
import json
import argparse
from run_span_classification_v1 import (
    AugmentRandomMask,
    AugmentExchangeEntity,
    AugmentExchangeSegments,
)
from packages import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/tmp_data/stage2-v1/dev.0.jsonl")
    parser.add_argument("--output_file", type=str, default="data/tmp_data/stage2-v1/dev.0.aug.jsonl")
    parser.add_argument("--augment_times", type=int, default=1)
    parser.add_argument("--do_random_mask_augment", action="store_true")
    parser.add_argument("--do_exchange_entity_augment", action="store_true")
    parser.add_argument("--do_exchange_segments_augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    examples = []
    with open(args.input_file, "r") as f:
        data_type = "train"
        for i, line in enumerate(f.readlines()):
            line = json.loads(line)
            guid = f"{data_type}-{i}"
            tokens = line["text"]
            entities = None
            if data_type != "test":
                entities = line["entities"]
            examples.append(dict(guid=guid, text=tokens, entities=entities, sent_start=0, sent_end=len(tokens)))

    augment_pipline = [
        AugmentRandomMask(p=1.0, mask_mode="non-entity", mask_proba=0.15, mask_token="[MASK]"
            ) if args.do_random_mask_augment else None,
        AugmentExchangeEntity(p=1.0) if args.do_exchange_entity_augment else None,
        AugmentExchangeSegments(p=1.0, beta=5) if args.do_exchange_segments_augment else None,
    ]
    augmented_examples = []
    for example in examples:
        for i in range(args.augment_times):
            augmented = copy.deepcopy(example)
            for process in augment_pipline:
                if process is None:
                    continue
                augmented = process(augmented)
            augmented_examples.append(augmented)

    # 查看样例
    for example_no, (example, augmented) in enumerate(zip(examples, augmented_examples)):
        print(example)
        print(augmented)
        print()
        if example_no > 10: break

    with open(args.output_file, "w") as f:
        for example in examples + augmented_examples:
            if "status" in example:
                example.pop("status")
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
