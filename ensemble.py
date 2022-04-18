#%%
import os
import sys
import glob
import torch
from tqdm import tqdm
from itertools import chain

sys.path.append("TorchBlocks/")
from torchblocks.utils.paths import load_pickle
from torchblocks.utils.options import Argparser
from run_span_classification_v1 import (
    GaiicTrack2SpanClassificationDataset, 
    GaiicTrack2ProcessExample2Feature,
    ModelForSpanClassification, 
    update_example_entities,
    entities_to_ner_tags,
    load_dataset,
)
from tokenization_bert_zh import BertTokenizerZh

#%%
# dirnames = glob.glob("outputs/gaiic_*datav2.*-span35*")
dirnames = glob.glob("outputs/gaiic_nezha_nezha-100k-*span35*-bs32x1-*")

# %%
filenames = []
for dirname in dirnames:
    filename = os.path.join(dirname, "checkpoint-eval_f1_micro_all_entity-best", "test_predict_results.pkl")
    if os.path.exists(filename):
        filenames.append(filename)
print(len(filenames))

#%%
label2id = GaiicTrack2SpanClassificationDataset.label2id()
id2label = GaiicTrack2SpanClassificationDataset.id2label()

# %%
num_models = len(filenames)
logits = None; spans = None; spans_mask = None
for fileno, filename in tqdm(enumerate(filenames), total=num_models):
    outputs = load_pickle(filename)
    if fileno == 0:
        logits = [output["logits"] / num_models for i, output in enumerate(outputs)]
        spans = [output["spans"] for output in outputs]
        spans_mask = [output["spans_mask"] for output in outputs]
    else:
        logits = [logits[i] + (output["logits"] / num_models) for i, output in enumerate(outputs)]

#%%
predictions = []
for l, s, m in zip(logits, spans, spans_mask):
    predictions.append(ModelForSpanClassification.decode(l, s, m, 0.0, label2id, id2label))

#%%
json_file = glob.glob(os.path.join(dirnames[0], "*.json"))[0]
opts = Argparser.parse_args_from_json(json_file=json_file)
tokenizer_kwargs = {
    "do_lower_case": opts.do_lower_case,
    "do_ref_tokenize": opts.do_ref_tokenize,
}
tokenizer = BertTokenizerZh.from_pretrained(
    os.path.join(opts.pretrained_model_path, "vocab.txt"), **tokenizer_kwargs)

#%%
test_dataset  = load_dataset(
    GaiicTrack2SpanClassificationDataset, GaiicTrack2ProcessExample2Feature, 
    opts.test_input_file, opts.data_dir, "test",
    tokenizer, opts.test_max_seq_length, opts.context_size, opts.max_span_length,
    opts.negative_sampling, stanza_nlp=None, labels=opts.labels, 
    max_examples=opts.max_test_examples, do_preprocess=opts.do_preprocess,
)

#%%
entities = list(chain(*[p for p in predictions]))
examples = update_example_entities(tokenizer, test_dataset.examples, entities, test_dataset.process_piplines[:-1])
# with open(os.path.join(checkpoint, "predictions.json"), "w") as f:
#     for example in examples:
#         f.write(json.dumps(example, ensure_ascii=False) + "\n")

predicion_file = f"{opts.test_input_file}.predictions.txt"
with open(os.path.join("./", predicion_file), "w") as f:
    for example_no, example in tqdm(enumerate(examples), total=len(examples), 
            desc=f"Writing to {predicion_file}"):
        text = test_dataset.examples[example_no]["text"]
        ner_tags = entities_to_ner_tags(len(text), example["entities"])
        assert len(text) == len(ner_tags)
        for token, tag in zip(text, ner_tags):
            f.write(f"{token} {tag}\n")
        f.write(f"\n")
