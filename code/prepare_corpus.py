import os
import sys
import random
from itertools import chain
from argparse import ArgumentParser
from prepare_data import generate_examples, create_examples
from utils import get_spans_bio

sys.path.append("TorchBlocks/")
from torchblocks.utils.seed import seed_everything

def generate_examples_from_lines(input_file):
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            yield (i, dict(tokens=list(line)))

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--version", type=str, default="pretrain-v0")
    parser.add_argument("--output_dir", type=str, default="data/processed/")
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # prepare
    seed_everything(args.seed)
    args.output_dir = os.path.join(args.output_dir, args.version)    
    os.makedirs(args.output_dir, exist_ok=True)

    corpus = []
    entities = set()
    for example in chain(
        generate_examples("../data/contest_data/train_data/train.txt"),
        generate_examples("../data/contest_data/preliminary_test_a/word_per_line_preliminary_A.txt"),
        generate_examples("../data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt"),
        generate_examples_from_lines("../data/contest_data/train_data/unlabeled_train_data.txt"),
    ):
        text = "".join(example[1]["tokens"])
        # 保存实体词典
        ner_tags = example[1]["ner_tags"] if "ner_tags" in example[1] else None
        if ner_tags is not None:
            for label, start, end in get_spans_bio(ner_tags):
                entities.add(text[start: end + 1])
            # import jieba
            # a = jieba.lcut(text)
            # jieba.load_userdict("data/processed/pretrain-v0/entities.txt")
            # b = jieba.lcut(text)
            # print()
        length = len(text)
        if length < args.min_length or length > args.max_length:
            continue
        corpus.append(text)
    print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")

    with open(os.path.join(args.output_dir, "entities.txt"), "w", encoding="utf-8") as f:
        f.writelines([entity + "\n" for entity in entities])

    random.shuffle(corpus)
    corpus = list(map(lambda x: x + "\n", corpus))
    with open(os.path.join(args.output_dir, "corpus.txt"), "w", encoding="utf-8") as f:
        f.writelines(corpus)

    if args.train_ratio is not None:
        num_corpus_train = int(len(corpus) * args.train_ratio)
        corpus_train = corpus[: num_corpus_train]
        corpus_valid = corpus[num_corpus_train: ]
        with open(os.path.join(args.output_dir, "corpus.train.txt"), "w", encoding="utf-8") as f:
            f.writelines(corpus_train)
        with open(os.path.join(args.output_dir, "corpus.valid.txt"), "w", encoding="utf-8") as f:
            f.writelines(corpus_valid)
