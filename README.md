## 优化方向

- 物品共现关系统计

## 更新

### 2022/4/12
1. 实验
   - `nezha-100k-spanv1-datav4-lr3e-5-wd0.01-dropout0.1-span50-e6-bs12x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8117；
   - `nezha-100k-spanv1-datav4-lr3e-5-wd0.01-dropout0.1-span50-e6-bs12x2-sinusoidal-fgm1.0-rdrop0.3`，线下0.8142，线上0.8123719204143346；
   - `nezha-100k-spanv1-datav4-lr3e-5-wd0.01-dropout0.1-span50-w64-e6-bs12x2-sinusoidal-fgm1.0-rdrop0.3`，线下0.814；
2. 新增`~run_mlm_wwwm.DataCollatorForNGramWholeWordMask`，支持ngram mask，以及macbert（同义词待实现）；
3. `~tokenization_bert_zh.BertTokenizerZh`将space token移除special tokens，并重写`tokenize`函数；

### 2022/4/11
1. 完善`~tokenization_bert_zh.BertTokenizerZh`：
   - 新增`do_ref_tokenize`参数，并重写`_tokenize_chinese_chars`，可实现带词汇信息的中文分词（基于`jieba`）；
   - 重写`_batch_encode_plus`与`_batch_prepare_for_model`，支持批量分词（在`run_mlm_wwm.py`中需要）；
2. 带词汇信息的预训练：
   - 新增`extend_chinese_ref_embeddings.py`，新增`##X`并初始化对应词向量（复制`X`的词向量）；
   - `run_mlm_wwm.py`新增`do_ref_tokenize`配置项，可进行带词汇信息的中文预训练；
   - `nezha-cn-base-wwm-word-seq128-lr2e-5-mlm0.15-200k-warmup3k-bs64x2`；
3. 微调，`nezha-100k-spanv2-datav3-lr2e-5-wd0.01-cos-dp0.1-span35-e6-bs16x2-fgm0.5-rdrop0.5-lstmx1-last4mean-tklv`，线上0.8112401066614894。

### 2022/4/10
1. 以wordpiece级别进行下游任务微调（之前为char级别），因为预训练截断以wordpiece粒度进行，`run_span_classification_v2`；
2. 为支持wordpeice级别，完善`BasicTokenizerZh`，具体：
   - 实现`WordpieceTokenizerZh`，支持返回`offsets_mapping`，并解决`unk`偏移量问题；
   - 重写`_tokenize`、`_encode_plus`、`prepare_for_model`等函数，支持返回`offsets_mapping`；
   - 新增`is_pre_tokenized`分词参数，支持将已分词的序列输入分词器，`_batch_encode_plus`待实现；
3. 修复`ProcessExample2Feature`中`skip_indices`，仅跳过对`cls`、`sep`、`pad`token；
4. 微调，`nezha-100k-spanv2-datav3-lr3e-5-wd0.01-dropout0.3-span20-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-tklv`，已进行数据校验，部分由于分词问题导致的实体存在偏差(F1=0.999686)，线上0.8134645317760053。

TODO:
1. cosine学习率
2. SWA
3. 参考[ccks2021 中文NLP地址要素解析 冠军方案 - 知乎](https://zhuanlan.zhihu.com/p/449676168)内Kaggle优化策略；
4. K折：1)数据清洗？2）伪标签；
5. Albert-style；
6. [Weighted Layer Pooling](https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook)
7. `run_span_classfication_v1`未作数据校验，可能存在问题；

### 2022/4/9
1. grouped llrd，`nezha-100k-spanv1-datav3-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-llrd0.95`，线上0.8134104896855145；
2. 伪标签，`nezha-100k-spanv1-datav6-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线上0.8135514749209326；
3. Labeltxt代码优化：1)复制实体时越界问题；2)未打开文件时不支持界面点击;

### 2022/4/8
1. 继续预训练100k步，`nezha-cn-base-wwm-seq128-lr2e-5-mlm0.15-100k-warmup3k-bs64x2`，最终MLM损失1.6174659729003906；
2. `nezha-100k-spanv1-datav3-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线上0.8136793661222608；
3. 完善半监督代码`run_span_classification_ssl.py`，实验效果不佳；
4. BERT中文分词器`_clean_text`函数支持控制字符分词；
5. huggingface上传`nezha-cn-wwm-base`及`nezha-cn-wwm-large`(private)。
6. NMS尝试以类别概率为评分：效果差别不大，召回率略高、精确率略低；
7. MC-Dropout，效果一般，召回率较低。

### 2022/4/7
1. `nezha-50k-spanv1-datav3-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线上0.8132536474956525。

### 2022/4/5

1. `large`微调，全量数据，FGM0.5，`nezha-large-100k-spanv1-datav3-lr2e-5-wd0.01-dropout0.1-span35-e6-bs16x2-sinusoidal-biaffine-fgm0.5`，线上0.8084194384827382；
2. 新增半监督训练`run_span_classification_ssl`。

### 2022/4/4
1. `large`微调，全量数据，FGM0.5，`nezha-large-100k-spanv1-datav3-lr2e-5-wd0.01-dropout0.1-span35-e6-bs16x2-sinusoidal-biaffine-fgm0.5`，待训练。

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
