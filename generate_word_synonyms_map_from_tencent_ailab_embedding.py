#%%
import os
import json
from tqdm import tqdm
from collections import defaultdict
from gensim.models import KeyedVectors
from prepare_data import create_examples, generate_examples

#%%
wv_file = "/home/louishsu/NewDisk/Garage/weights/gensim/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
wv_from_text = KeyedVectors.load_word2vec_format(wv_file, binary=False)

# %%
labeled_file = "data/raw/train_data/train.txt"
examples = create_examples(generate_examples(labeled_file), "train")

#%%
words = set()
for example in examples:
    for entity in example["entities"]:
        word = "".join(entity[-1])
        if word in words:
            continue
        try:
            wv_from_text[word]
        except KeyError:
            continue
        words.add(word)

#%%
word_synonyms_map = defaultdict(list)
for word in tqdm(words, total=len(words)):
    synonyms = wv_from_text.most_similar_cosmul(positive=[word], negative=None, topn=20)
    word_synonyms_map[word].extend(synonyms)

#%%
with open("data/word_synonyms_map.wv.json", "w") as f:
    json.dump(word_synonyms_map, f, ensure_ascii=False, indent=4)
