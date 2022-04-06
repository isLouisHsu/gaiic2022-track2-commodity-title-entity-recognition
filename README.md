## 优化方向

- 物品共现关系统计

## 更新

### 2022/4/5

1. `large`微调，全量数据，FGM0.5，`nezha-large-100k-spanv1-datav3-lr2e-5-wd0.01-dropout0.1-span35-e6-bs16x2-sinusoidal-biaffine-fgm0.5`，线上0.8084194384827382；
2. 新增半监督训练`run_span_classification_ssl`。

### 2022/4/4
1. `large`微调，全量数据，FGM0.5，`nezha-large-100k-spanv1-datav3-lr2e-5-wd0.01-dropout0.1-span35-e6-bs16x2-sinusoidal-biaffine-fgm0.5`，待训练。

TODO:
1. MacBERT/PERT；
2. 标签清洗，用K折，待large模型微调；
3. 半监督：伪标签/在线；

### 2022/4/4
1. 同义词替换增强，版本`nezha-50k-spanv1-datav2-augv1-lr3e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0`，线下0.8076；
2. 继续预训练，总步数100k步，版本`nezha-cn-base-wwm-seq128-lr3e-5-mlm0.15-100k-warmup30k-bs64x2`

### 2022/4/3
1. 预训练50k步，`nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-50k-warmup30k-bs64x2`，MLM损失`"eval_loss": 1.778834581375122`；
2. 用50k步模型训练，版本`nezha-50k-spanv1-datav2-lr3e-5-wd0.01-dropout0.1-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0`，线下0.8093，线上0.809930004779276；
3. 尝试R-Drop，版本`nezha-50k-spanv1-datav2-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8098，线上0.8108433916489609；
    **为什么有效？总体召回率较高、精确率较低。**

### 2022/4/1
1. 优化`tokenization_bert_zh.py/BasicTokenizerZh`；
2. 优化预训练：
    - `prepare_corpus.py`训练语料添加测试集；
    - `run_chinese_ref.py`换用`BasicTokenizerZh`，采用`LTP`分词；
    - 生成预训练数据`data/processed/pretrain-v1/`；
    - 预训练`nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-30k-warmup1k-bs32x2`；
3. 新增数据增强
    - `generate_word_synonyms_map_from_tencent_ailab_embedding.py`；
    - `run_span_classification_v1.py`新增`AugmentSynonymReplace`，功能待实现；

### 2022/3/31
1. 新增数据增强相关，待测试：
    - `ProcessBaseDual`
    - `ProcessConcateExamplesRandomly`
    - `ProcessDropRandomEntity`
    - `ProcessMaskRandomEntity`
    - `ProcessRandomMask`(未实现)
    - `ProcessSynonymReplace`(未实现)
        用[腾讯预训练词向量](https://ai.tencent.com/ailab/nlp/zh/embedding.html)求近义词
2. `Nezha`代码完善

### 2022/3/30

1. 尝试去重策略
    基于版本：`nezha-cn-base-span-v1-lr3e-5-wd0.01-dropout0.5-span15-e5-bs16x1-sinusoidal-biaffine-fgm1.0`
    - 用`drop_overlap_baseline`对实体去重，线下0.8092，线上0.8061317373961586
    - 用`drop_overlap_nms`对实体去重，线下0.8097，线上0.8067522940214945
    - 如果`drop_overlap_nms`有效，考虑添加分类器，输出“为实体的概率”，以该得分进行nms
2. 修改`utils.py/get_spans_bio`，修复单个字符实体遗漏问题，重新生成数据集`v2`
    - 注：划分与`v0`一致，因为目前第1折实验上是同步的，先实验调参，后续再增加训练数据
3. 实验：`nezha-cn-base-spanv1-datav2-lr3e-5-wd0.01-dropout0.5-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0`
    - 线下0.8079，线上0.8073288575268414
4. `torchblocks`定义`scheduler`可通过`Trainer`传入，但定义时缺少`num_training_steps`信息？
5. TODO: 
   - `NeZha-base/large-wwm`
   - `scheduler`
   - `PGD`
   - 用少量训练数据进行快速验证（如10000条），通过`--max_train/eval/test_examples`指定

### 2022/3/29
1. 尝试[dbiir/UER-py](https://github.com/dbiir/UER-py)下其他模型，如`MixedCorpus+BertEncoder(xlarge)+MlmTarget`,`MixedCorpus+BertEncoder(xlarge)+BertTarget(WWM)`
2. 模型部署时，可参考[ELS-RD/transformer-deploy](https://github.com/ELS-RD/transformer-deploy)
