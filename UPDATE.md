## 优化方向

- 物品共现关系统计

## 更新

### 2022/5/8
1. 线上预训练加微调
   - `run_pretrain_nezha_v2`、`run_pretrain_nezha_v3`
   - `experiment_bert_base_fold0_gp_v2_pre_v62`

### 2022/5/17
1. 提交线上
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0`，即`gmodel_spancls_2022051617829`，0.8104775463912551
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp0.5-rdrop1.0`，即`gmodel_spancls_2022051629391`，0.815385819848358

### 2022/5/16
1. 提交线上，`nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0-pseuv1`，即`gmodel_spancls_2022051528877`，0.8154189358838884；
2. 线上微调
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0¶`
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp0.5-rdrop1.0`

### 2022/5/15
1. 线下微调，`nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0-cosine`,0.8155;
2. 提交线上，`nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0-cosine`，即`gmodel_spancls_2022051515130`，0.8153156273245536

### 2022/5/14

1. 准备蒸馏/伪标签，目前线上/线下模型相近的为
   线上
   - **2022/5/10/1** nezha-100k-spanv1-datas2v0-lr3e-5-wd0.001-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4 - 0.8144165048602052
   - **2022/5/12/1** nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0 - 0.8154596708823378
   - 2022/5/12/2 nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0-augv0
   - **2022/5/12/3** nezha-100k-spanv1gp-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0 - 0.814864726037745
   - 2022/5/13/1 nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.5-awp1.0-rdrop2.0
   线下
   - nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-lsr0.01 - 0.8144
   - nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0 - 0.8152
   - nezha-100k-spanv1gp-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0 - 0.8154
2. 线下微调
   - `nezha-100k-spanv1-datas2v2.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0-pseuv0`，TODO:
   - `nezha-100k-spanv1-datas2v2.0-lr4e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0-pseuv0`，0.8156
3. 线上微调
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0-pseuv0`
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp2.0-rdrop1.0`
4. 提交线上，`nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp2.0-rdrop1.0`，即`gmodel_spancls_202205142`，	0.8153485305142973

TODO: 
1. ~~AWP参数调整~~
2. 列一下线上/线下模型及分数，选择相近的线下调蒸馏/伪标签；
3. SWA单独微调；

### 2022/5/13

1. 线下微调
   - `nezha-100k-spanv1-datas2v0.0-augv0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0`，0.8127；
   - `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgme1.0-rdrop0.4-ploy1_ce2.0`，0.8118；
   - `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-pgde1.0-rdrop0.4-ploy1_ce2.0`，0.8136；
   - `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0`，0.8152；
   - `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop2.0-ploy1_ce2.5`，0.8151;
   - `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce2.0`，不佳；
2. 提交线上
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0`，即`gmodel_spancls_2022051219618`，0.8154596708823378；
   - `nezha-100k-spanv1gp-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0`，即`gmodel_gp_2022051235705`，0.814864726037745；
3. 线上微调
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.5-awp1.0-rdrop2.0`

TODO: 
1. ~~静态数据增强；~~
2. ~~用新词发现分析标签；~~
3. ~~实体库用作分词；~~

### 2022/5/12
1. 线上导出模型`gmodel_nezha_cn_base_entity`，MLM最终损失1.5178041458129883；
2. 线上微调
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0`，；
   - `nezha-100k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0-augv0`；
   - `nezha-100k-spanv1gp-datas2v0-lr3e-5-wd0.01-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-ploy1_ce2.0-awp1.0-rdrop1.0`；

### 2022/5/11
2. 提交线上，`nezha-100k-spanv1-datas2v0-lr3e-5-wd0.001-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4`，`gmodel_spancls_202205101483`，0.8144165048602052；
3. 提交离线任务`nezha-cn-base-seq128-lr3e-5-mlm0.15-100k-warmup10k-bs64x2`，数据源`gdata_pretrain_v31104`；
4. 线下微调，`nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-ploy1_ce0.1`，0.8149，Ploy1Loss有效；
5. 修复静态数据增强概率计算问题，已生成数据，待微调；

### 2022/5/10
1. 修改`run_span_classification_v1.SpanClassificationMixin.drop_overlap_rule`；
2. 新增`run_span_classification_v1.set_extra_defaults`；
3. 为实体增强MLM实现：`prepare_corpus.py`新增保存实体列表，`run_chinese_ref.py`支持jieba导入用户词典；
4. 提交线上`2022/5/9 nezha-200k-spanv1-datas2v0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop1.0-swav1`，0.8131485592934022；
5. 新增`data_augmentation.py`，用于静态数据增强。
6. 线上微调，`nezha-100k-spanv1-datas2v0-lr3e-5-wd0.001-dropout0.5-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4`；
7. 提交离线任务，构造数据源`gdata_pretrain_v3`；

TODO:
1. ~~线上微调对比AWP效果，用mlm损失1.392，待导出模型提交；~~
2. ~~实体增强MLM，数据处理中，待导出后预训练，学习率由1.结果确定(2e-5或3e-5)；~~

### 2022/5/9
1. 新增`AugmentRandomMask`、`AugmentExchangeEntity`、`AugmentExchangeSegments`；
2. `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1.0-rdrop0.4-lsr0.01`，线下0.8144；

### 2022/5/8
1. 提交线上`gmodel_spancls_202205077213`，0.8135973553679232；
2. 线上预训练`gmodel_nezha_cn_base_gaiic_1.392`；
3. `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01`，线下0.8134，精确率与召回率相近；
4. `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.001-last4mean-lstm2-dropout0.3-span128-e8-bs16x1-fgm1.0-rdrop0.4`，线下0.8127；
5. 提交线上`gmodel_gpswa_202205089550`，0.8135456792962686；

### 2022/5/7
1. `nezha-100k-spanv1-datas2v1.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01-pseuv0`，线下0.8136；
2. `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01-swav0`，线下0.8132，swa0.8135；
3. 提交线上`gmodel_spancls_20220505065060/`，0.8132664117368327；
4. `nezha-finetune-spanv1-datas2v0.0-lr1e-5-wd0.01-dropout0.3-span35-e1-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01-swav0`，线下0.813，swa0.8129；
5. 线上预训练`gmodel_nezha_cn_base_gaiic_1.477`；

### 2022/5/6
1. 新增伪标签学习/蒸馏；
2. `nezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.4-lsr0.01`线下0.8136；

### 2022/5/5
1. 排查线上提交错误问题，成功提交，修复问题包括：
   1. 新增packages包，用于部署模型服务（不支持勾选文件夹问题）；
   2. 解决transformers==4.17.0与datasets==1.8.0两个包依赖冲突（huggingface-hub）；
   3. 修改submit_results.py：
      1. 不安装apex，并设置opts.fp17 = False；
      2. 线上pred_BIO传入绝对路径，无需添加"../"前缀；
      3. 修改opts.test_input_file文件名为线上测试集文件；
      4. run_span_classification_v1.py中do_predict阶段预测文件不落盘（/home/mw/project/容量仅3G）；
2. 线上继续预训练
   1. run_chinese_ref.py修改LTP导入位置，并修改默认args.ltp参数为None；
   2. 解决ipynb修改环境变量问题，用`%env KEY=VALUE`命令设置

### 2022/5/1
1. `nezha-4gram-200k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs8x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8108；

### 2022/4/30
1. 预训练，`nezha-cn-base-wwm-4gram-seq128-lr2e-5-mlm0.15-200k-warmup5k-bs64x2`，MLM最终损失1.674683690071106；
2. 预训练，`nezha-cn-base-wwm-4gram-seq128-lr3e-5-mlm0.15-100k-warmup1k-bs64x2`，MLM最终损失1.7672427892684937；
3. `nezha-4gram-200k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0`，线下0.812；
4. `nezha-4gram-200k-spanv1xy-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0`，w/ share_fc，线下0.811；

### 2022/4/28
1. 新增X, Y分类实现；

### 2022/4/26
1. 新增Mixup损失；
2. 新增X, Y维度评估；
3. 新增X, Y分类，具体待实现。

### 2022/4/25
1. 预训练，`nezha-cn-base-mac-seq128-lr2e-5-mlm0.15-4gram-100k-bs64x2`，MLM损失1.924962043762207；
2. `macnezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8121；
3. `nezha-100k-spanv1-datas2v0.0-lr5e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8121。

### 2022/4/24
1. `~run_mlm_wwm.DataCollatorForNGramWholeWordMask`修改`mlm_as_correction`实现，但速度奇慢，待优化。

### 2022/4/23
1. 按规范整理代码结构，并提交；
2. 搭建复赛基线(10折，无伪标签，初赛最优参数)，`ezha-100k-spanv1-datas2v0.0-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8131；
3. 尝试MAC预训练：
   - 添加初赛B榜测试集数据（10k条）
   - 最短文本10，训练集比例0.95，总计1059869条
   - 用ltp分词
   - MacBERT(4-gram + mlm_as_correction)
   - 调整超参数，无warmup

### 2022/4/21
1. `nezha-100k-spanv1-datav6-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-pseu0.4`，B榜，线上0.8139140781337324；
2. `nezha-100k-spanv1-datav7-lr3e-5-wd0.01-dropout0.3-span35-e8-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-pseu0.4-swa`，利用B榜结果作伪标签，并加入SWA，线上0.813454025383709

### 2022/4/20
1. `nezha-100k-spanv1-datav3-pre-lr3e-5-wd0.01-dropout0.3-span35-e6-bs32x1-sinusoidal-biaffine-awp1.0-rdrop0.3`，线上0.8091197920427468；
2. 新增伪标签，`nezha-100k-spanv1-datav6-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3-pseu0.4`，线上0.813717708182999；

### 2022/4/19
1. 以[2022/4/18](#2022418)结果作为伪标签训练，`nezha-100k-base-spanv1-datav7-pre-lr3e-5-wd0.01-dropout0.3-span35-e8-bs32x1-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线上0.8132154960622345；

### 2022/4/18
1. K折训练，`nezha-100k-spanv1-datav2.${k}-pre-lr3e-5-wd0.01-dropout0.3-span35-e6-bs32x1-sinusoidal-biaffine-fgm1.0`，每折线下0.8098(线上0.8107454805205765)，0.8096，0.8087，0.8126，0.8112，均值集成线上0.813503921456291；
2. `nezha-100k-spanv1-datav3-pre-lr3e-5-wd0.01-dropout0.3-span50-e6-bs32x1-sinusoidal-fgm1.0-rdrop0.3-lsr0.1`，线上0.8122967064554684；

### 2022/4/17
1. 新增`~torchblocks.callback.adversarial.awp.AWP`；
2. `nezha-100k-spanv1-datav4-lr2e-5-wd0.01-dp0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8123；
3. `nezha-100k-spanv1-datav4-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-awp1-rdrop0.3`，线下0.8109；

### 2022/4/16
1. `nezha-100k-spanv1-datav4-lr3e-5-wd0.01-dropout0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8135，线上 0.812235943611719；
2. `nezha-100k-spanv1-datav4-lr3e-5-wd0.01-wup0.0-schecos-dp.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8129；
3. `nezha-100k-spanv1-datav4-lr2e-5-wd0.01-dp0.3-span35-e6-bs16x2-sinusoidal-biaffine-fgm1.0-rdrop0.3`，线下0.8123；

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
