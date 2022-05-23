# 代码说明

## 环境配置

基础镜像：GAIIC 比赛镜像 (腾讯云专用)
Python：镜像默认版本
Pytorch：安装1.7.0
额外安装包：
``` txt
torch==1.7.0
tensorboardX==2.1
transformers==4.18.0
jieba==0.42.1
scikit-learn==0.23.2
pandas==1.1.4
numpy==1.19.4
synonyms==3.16.0
gensim==4.1.2
ltp==4.1.3.post1
torchmetrics==0.7.2
pytorch-lightning==1.2.6
datasets==2.1.0
```

## 数据
未使用公开数据

## 预训练模型
- 使用nezha-cn-base，路径存放于`data/pretrain_model/nezha-cn-base`，可由[lonePatient/NeZha_Chinese_PyTorch - Github](https://github.com/lonePatient/NeZha_Chinese_PyTorch)仓库中网盘链接获取。

## 算法

### 整体思路介绍（必选）
**预训练**

分为两阶段：
1. 无标注语料预训练
	- 语料：无标注数据（100W）、初赛A/B榜测试集（2W）
	- 任务：N-gram mask language modeling
	- 模型：nezha-cn-base
	- 参数：见`run_pretrain_nezha_v2.py`
2. 在1基础上，用训练语料继续预训练
	- 语料：训练集（4W）
	- 任务：N-gram mask language modeling
	- 模型：步骤1输出的模型
	- 参数：见`run_pretrain_nezha_v3.py`

**微调**
- 模型：采用GlobalPointer命名实体提取模型
- 数据：采用全量训练集（4W）
- 后处理：该模型可提取嵌套实体，设定规则去重

### 方法的创新点（可选）
略

### 网络结构（必选）
NeZha -> LSTM -> GlobalPointer

### 损失函数（必选）
交叉熵损失，计算每个片段的分类损失

### 数据增广（可选）
无

### 模型集成（可选）
根据本赛题规则，只允许训练阶段进行模型集成，不允许预测阶段进行模型集成

### 算法的其他细节（可选）
略

## 训练流程
见train.ipynb

## 测试流程
见test.ipynb

## 其他注意事项
无
