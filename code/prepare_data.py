import os
import sys
import json
import random
from typing import *
from argparse import ArgumentParser
from sklearn.model_selection import KFold

# sys.path.append("TorchBlocks/")
# from torchblocks.utils.seed import seed_everything
# from torchblocks.metrics.sequence_labeling.scheme import get_scheme
from packages import seed_everything
from utils import get_spans_bio

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
    # get_entities = get_scheme("BIO")  # FIXED: 该函数无法提取由"B-X"标记的单个token实体
    get_entities = get_spans_bio
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

def find_most_similar_unlabeled(labeled_examples, unlabeled_examples, topn_per_unlabeld, reduce_dim=64, batch_size=256):
    import jieba
    from tqdm import tqdm
    from gensim import corpora, models, similarities

    # 对输入语料分词
    labeled_texts = ["".join(example["text"]) for example in tqdm(labeled_examples, total=len(labeled_examples))]
    unlabeled_texts = ["".join(example["text"]) for example in tqdm(unlabeled_examples, total=len(unlabeled_examples))]
    # labeled_texts = [jieba.lcut(text) for text in tqdm(labeled_texts, total=len(labeled_texts))]
    # unlabeled_texts = [jieba.lcut(text) for text in tqdm(unlabeled_texts, total=len(unlabeled_texts))]
    labeled_texts = [list(text) for text in tqdm(labeled_texts, total=len(labeled_texts))]
    unlabeled_texts = [list(text) for text in tqdm(unlabeled_texts, total=len(unlabeled_texts))]
    all_texts = labeled_texts + unlabeled_texts

    # 生成词典
    dictionary = corpora.Dictionary(all_texts)
    # 通过doc2bow稀疏向量生成语料库
    labeled_corpus = [dictionary.doc2bow(text) for text in labeled_texts]
    unlabeled_corpus = [dictionary.doc2bow(text) for text in unlabeled_texts]
    corpus = labeled_corpus + unlabeled_corpus
    # 通过TF模型算法，计算出tf值
    tfidf = models.TfidfModel(corpus)

    # 通过token2id得到特征数（字典里面的键的个数）
    num_features = len(dictionary.token2id.keys())
    # 计算稀疏矩阵相似度，建立一个索引
    labeled_index = similarities.MatrixSimilarity(tfidf[labeled_corpus], num_features=num_features)
    unlabeled_index = similarities.MatrixSimilarity(tfidf[unlabeled_corpus], num_features=num_features)
    labeled_matrix = labeled_index.index; unlabeled_matrix = unlabeled_index.index

    # 用PCA降维，防止计算量过大
    print("reducing dimension...")
    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA(n_components=reduce_dim)
    pca.fit(np.concatenate([labeled_matrix, unlabeled_matrix], axis=0))
    print(pca.explained_variance_ratio_)
    labeled_matrix = pca.transform(labeled_matrix)
    unlabeled_matrix = pca.transform(unlabeled_matrix)
    print("reducing dimension done")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    similarities = []
    labeled_matrix = torch.tensor(labeled_matrix).cuda()
    unlabeled_matrix = torch.tensor(unlabeled_matrix)
    unlabeled_dataset = TensorDataset(unlabeled_matrix)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
    for batch_no, unlabeled_matrix_batch in tqdm(enumerate(unlabeled_dataloader), total=len(unlabeled_dataloader)):
        unlabeled_matrix_batch = unlabeled_matrix_batch[0].cuda()
        similarities_batch = F.cosine_similarity(labeled_matrix.unsqueeze(1),
            unlabeled_matrix_batch.unsqueeze(0), dim=-1)    # (num_labeled, num_batch)
        similarities.append(similarities_batch.cpu())
    similarities = torch.cat(similarities, dim=-1)

    # _, indices = torch.topk(similarities, topn_per_unlabeld, dim=-1)
    indices = []
    for sim in similarities:
        indices.append(torch.topk(sim, topn_per_unlabeld))
    indices = torch.stack(indices, dim=0)
    print("total: ", indices.view(-1).size(0))
    indices = torch.unique(indices)
    print("unique: ", indices.view(-1).size(0))

    return indices.cpu().numpy().tolist()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--labeled_files", type=str, nargs="+", default=[
        "data/raw/train_data/train.txt",
    ])
    parser.add_argument("--pseudo_files", type=str, nargs="+", default=None)
    parser.add_argument("--unlabeled_files", type=str, nargs="+", default=None)
    parser.add_argument("--num_unlabeled_most_similar", type=int, default=None)
    parser.add_argument("--test_files", type=str, nargs="+", default=[
        "data/raw/preliminary_test_a/word_per_line_preliminary_A.txt",
    ])
    parser.add_argument("--start_unlabeled_files", type=int, default=None)
    parser.add_argument("--end_unlabeled_files", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="data/processed/")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # prepare
    seed_everything(args.seed)
    args.output_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(args.output_dir, exist_ok=True)

    # split train & dev
    labeled_examples = []
    for labeled_file in args.labeled_files:
        labeled_examples.extend(create_examples(generate_examples(labeled_file), "train"))
    if args.n_splits > 1:
        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle)
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
    else:
        print(f"split=all, #train={len(labeled_examples)}, #dev=0")
        with open(os.path.join(args.output_dir, f"train.all.jsonl"), "w") as f:
            for example in labeled_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    if args.pseudo_files is not None:
        pseudo_examples = []
        for pseudo_file in args.pseudo_files:
            pseudo_examples.extend(create_examples(generate_examples(pseudo_file), "pseudo"))
        print(f"#pseudo={len(pseudo_examples)}")
        with open(os.path.join(args.output_dir, f"pseudo.jsonl"), "w") as f:
            for example in pseudo_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    if args.unlabeled_files is not None:
        unlabeled_examples = []; count = 0
        for unlabeled_file in args.unlabeled_files:
            with open(unlabeled_file, "r") as f:
                for line in f.readlines():
                    unlabeled_examples.append(dict(
                        guid=f"semi-{count}", text=list(line.strip()),
                        entities=None, sent_start=0, sent_end=len(line)
                    ))
        semi_file = os.path.join(args.output_dir, "semi.all.jsonl")
        if args.start_unlabeled_files is not None and args.end_unlabeled_files is not None:
            # random.shuffle(unlabeled_examples)
            unlabeled_examples = unlabeled_examples[
                args.start_unlabeled_files: args.end_unlabeled_files]
            semi_file = os.path.join(args.output_dir, f"semi.{args.start_unlabeled_files}:{args.end_unlabeled_files}.jsonl")
        if args.num_unlabeled_most_similar is not None:
            most_similar_unlabeled_indices = find_most_similar_unlabeled(
                labeled_examples, unlabeled_examples, args.num_unlabeled_most_similar)
            unlabeled_examples = [unlabeled_examples[i] for i in most_similar_unlabeled_indices]
        print(f"#semi={len(unlabeled_examples)}")
        with open(semi_file, "w") as f:
            for example in unlabeled_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    for test_file in args.test_files:
        test_examples = create_examples(generate_examples(test_file), "test")
        print(f"test file: {test_file}, #test={len(test_examples)}")
        basename, ext = os.path.splitext(os.path.basename(test_file))
        with open(os.path.join(args.output_dir, f"{basename}.jsonl"), "w") as f:
            for example in test_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        