import os
import torch
import json
import numpy as np
from torch import nn
from torch.nn import Module
from transformers import BertTokenizerFast
from gpsrc.dataset import DatasetBase
from gpsrc.utils import get_entity_biob, FGM, PGD
from gpsrc.seed import seed_everything
from gpsrc.device import prepare_device
from gpsrc.options import Argparser
from gpsrc.logger import Logger
import copy
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
from gpsrc.lr_scheduler import get_lr_scheduler
from gpsrc.adamw import AdamW
from gpsrc.progressbar import ProgressBar
from torch.optim import Adam
from torch.nn import functional as F
from gpsrc.nezha.modeling_nezha import NeZhaConfig, NeZhaModel, NeZhaPreTrainedModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gpsrc.layers import GlobalPointer, SpatialDropout, LayerNorm, ConvolutionLayer, WeightedLayerPooling, MLP, Biaffine
from gpsrc.polyloss import Poly1CrossEntropyLoss, Poly1FocalLoss

try:
    from torch.optim.swa_utils import (
        AveragedModel, update_bn, SWALR
    )
    SWA_AVAILABLE = True
except ImportError:
    SWA_AVAILABLE = False

class GlobalPointerNeZha(NeZhaPreTrainedModel):
    '''
    gp方案
    '''
    def __init__(self, config):
        super(GlobalPointerNeZha, self).__init__(config)
        self.num_labels = config.num_labels
        self.inner_dim = config.inner_dim
        self.use_rope = config.use_rope
        self.hidden_size = config.hidden_size
        self.do_lstm = config.do_lstm
        self.use_last_n_layers = config.use_last_n_layers
        self.bert = NeZhaModel(config)
        self.global_pointer = GlobalPointer(self.num_labels, self.inner_dim, self.hidden_size, self.use_rope)
        if self.do_lstm:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2,
                                num_layers=config.num_lstm_layers, batch_first=True, bidirectional=True)
        self.embedding_dropout = SpatialDropout(config.embed_dropout)
        self.do_weight_layer_pooling = config.do_weight_layer_pooling
        if self.do_weight_layer_pooling:
            self.WLP = WeightedLayerPooling(
                config.num_hidden_layers,
                layer_start=config.layer_start, layer_weights=None
            )
        self.post_lstm_dropout = nn.Dropout(config.post_lstm_dropout)
        self.init_weights()
        # self._init_weights(self.global_pointer)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, grid_mask2d=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=None,
            inputs_embeds=None,
        )
        sequence_output = outputs[0]
        # 加权
        if self.do_weight_layer_pooling:
            sequence_output = self.WLP(all_hidden_states=outputs[2])
        else:
            # 平均
            if self.use_last_n_layers is not None:
                last_hidden_state = outputs[2]
                last_hidden_state = torch.stack(last_hidden_state[- self.use_last_n_layers:], dim=-1)
                last_hidden_state = torch.mean(last_hidden_state, dim=-1)
                sequence_output = last_hidden_state
        sequence_output = self.embedding_dropout(sequence_output)
        if self.do_lstm:
            self.lstm.flatten_parameters()
            sequence_lengths = attention_mask.sum(dim=1)
            packed_sequence_output = pack_padded_sequence(
                sequence_output, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_sequence_output, _ = self.lstm(packed_sequence_output)
            unpacked_sequence_output, _ = pad_packed_sequence(
                packed_sequence_output, batch_first=True, total_length=sequence_lengths.max())
            sequence_output = sequence_output + unpacked_sequence_output
        sequence_output = self.post_lstm_dropout(sequence_output)
        logits = self.global_pointer(sequence_output, mask=attention_mask)
        outputs = {}
        outputs['logits'] = logits
        return outputs

class ResidualGatedConv1D(nn.Module):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1):
        super(ResidualGatedConv1D, self).__init__()
        self.filters = filters  # 输出维度
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True
        self.padding = self.dilation_rate*(self.kernel_size - 1)//2
        self.conv1d = nn.Conv1d(filters, 2*filters, self.kernel_size, padding=self.padding, dilation=self.dilation_rate)
        self.layernorm = nn.LayerNorm(self.filters)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        input_cov1d = inputs.permute([0, 2, 1])
        outputs = self.conv1d(input_cov1d)
        outputs = outputs.permute([0, 2, 1])
        gate = torch.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs

class GlobalPointerNeZhaV4(NeZhaPreTrainedModel):
    '''
    gp方案
    '''
    def __init__(self, config):
        super(GlobalPointerNeZhaV4, self).__init__(config)
        self.num_labels = config.num_labels
        self.inner_dim = config.inner_dim
        self.use_rope = config.use_rope
        self.hidden_size = config.hidden_size
        self.do_lstm = config.do_lstm
        self.use_last_n_layers = config.use_last_n_layers
        self.bert = NeZhaModel(config)
        self.global_pointer = GlobalPointer(self.num_labels, self.inner_dim, self.hidden_size, self.use_rope)
        self.embedding_dropout = SpatialDropout(config.embed_dropout)
        self.do_weight_layer_pooling = config.do_weight_layer_pooling
        if self.do_weight_layer_pooling:
            self.WLP = WeightedLayerPooling(
                config.num_hidden_layers,
                layer_start=config.layer_start, layer_weights=None
            )
        self.post_lstm_dropout = nn.Dropout(config.post_lstm_dropout)
        self.ResidualGatedConv1D_1 = ResidualGatedConv1D(self.hidden_size, 3, dilation_rate=config.dilation[0])
        self.ResidualGatedConv1D_2 = ResidualGatedConv1D(self.hidden_size, 3, dilation_rate=config.dilation[1])
        self.ResidualGatedConv1D_3 = ResidualGatedConv1D(self.hidden_size, 3, dilation_rate=config.dilation[2])
        self.ResidualGatedConv1D_4 = ResidualGatedConv1D(self.hidden_size, 3, dilation_rate=config.dilation[3])
        self.ResidualGatedConv1D_5 = ResidualGatedConv1D(self.hidden_size, 3, dilation_rate=config.dilation[4])
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, grid_mask2d=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=None,
            inputs_embeds=None,
        )
        sequence_output = outputs[0]
        # 加权
        if self.do_weight_layer_pooling:
            sequence_output = self.WLP(all_hidden_states=outputs[2])
        else:
            # 平均
            if self.use_last_n_layers is not None:
                last_hidden_state = outputs[2]
                last_hidden_state = torch.stack(last_hidden_state[- self.use_last_n_layers:], dim=-1)
                last_hidden_state = torch.mean(last_hidden_state, dim=-1)
                sequence_output = last_hidden_state
        sequence_output = self.embedding_dropout(sequence_output)
        sequence_output = self.post_lstm_dropout(sequence_output)
        sequence_output = self.ResidualGatedConv1D_1(sequence_output)
        sequence_output = self.ResidualGatedConv1D_2(sequence_output)
        sequence_output = self.ResidualGatedConv1D_3(sequence_output)
        sequence_output = self.ResidualGatedConv1D_4(sequence_output)
        sequence_output = self.ResidualGatedConv1D_5(sequence_output)
        logits = self.global_pointer(sequence_output, mask=attention_mask)
        outputs = {}
        outputs['logits'] = logits
        return outputs

def read_pseudo(pseudo_file_path):
    sentences = []
    sentence_counter = 0
    with open(pseudo_file_path, encoding="utf-8") as f:
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
            sentence = {
                    "id": str(sentence_counter),
                    "tokens": current_words,
                    "ner_tags": current_labels
                }
            sentence_counter += 1
            current_words = []
            current_labels = []
            sentences.append(sentence)
    return sentences

######################################## 数据处理
class GaiicDataset(DatasetBase):
    keys_to_truncate_on_dynamic_batch = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'grid_mask2d']
    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines,
                 pseudo_file_path=None,
                 **kwargs):
        super().__init__(data_name, data_dir, data_type, process_piplines,**kwargs)

    @classmethod
    def get_labels(cls):
        return [str(i) for i in range(55) if i not in [27, 45]]

    def read_data(self, input_file):
        lines = []
        with open(input_file) as fr:
            for line in fr.readlines():
                line = json.loads(line)
                lines.append(line)
        return lines
    def create_examples(self, data, data_type, **kwargs):
        examples = []
        for line in data:
            guid = f"{data_type}-{line['id']}"
            tokens = line['tokens']
            labels = line['ner_tags']
            examples.append(dict(guid=guid, tokens=tokens, labels=labels))
        # if data_type == 'train':
        #    pseudo_data = read_pseudo("../data/contest_data/train_data/pseudo_label_data_8w.txt")[:40000]
        #    for line in pseudo_data:
        #        guid = f"pseudo-{line['id']}"
        #        tokens = line['tokens']
        #        labels = line['ner_tags']
        #        examples.append(dict(guid=guid, tokens=tokens, labels=labels))
        print("all data size: ", len(examples))
        return examples

    def collate_fn(self, features):
        batch = {}
        first = features[0]
        max_input_length = first['input_ids'].size(0)
        if self.collate_dynamic:
            max_input_length = max([torch.sum(f["attention_mask"]) for f in features])
        if "labels" in first and first["labels"] is not None:
            batch["labels"] = torch.stack([f["labels"] for f in features])
        for k, v in first.items():
            if k != "labels" and v is not None and not isinstance(v, str):
                bv = torch.stack([f[k] for f in features]) if isinstance(v, torch.Tensor) \
                    else torch.tensor([f[k] for f in features])
                batch[k] = bv
        if self.collate_dynamic:
            for k in self.keys_to_truncate_on_dynamic_batch:
                if k in batch:
                    if k in ['labels', 'grid_mask2d']:
                        batch[k] = batch[k][:, :max_input_length, :max_input_length]
                    elif batch[k].dim() >= 2:
                        batch[k] = batch[k][:, : max_input_length]
        return batch

class ProcessExample2Feature:
    def __init__(self, label2id, tokenizer, max_sequence_length, do_mask, mask_p):
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.do_mask = do_mask
        self.mask_p = mask_p
        self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    def __call__(self, example):
        tokens = example['tokens']
        labels = example['labels']
        ## 处理空格以及异常符号
        new_tokens = []
        for i, word in enumerate(tokens):
            tokenizer_word = self.tokenizer.tokenize(word)
            if len(tokenizer_word) == 0:
                new_tokens.append("^")
            else:
                new_tokens.append(word)
        # 获取实体的span
        entity_spans = []
        for _type, _start_idx, _end_idx in get_entity_biob(labels, None):
            entity_spans.append({  # 左闭右闭
                'start_idx': _start_idx,
                'end_idx': _end_idx,
                'type': self.label2id[_type],
                'entity': "".join(tokens[_start_idx:_end_idx + 1])
            })
        # 重新生成text
        text = "".join(new_tokens)  # mapping
        token2char_span_mapping = self.tokenizer(text,
                                                 return_offsets_mapping=True,
                                                 max_length=self.max_sequence_length,
                                                 truncation=True)['offset_mapping']
        # 映射start和end的关系
        start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}

        encoder_txt = self.tokenizer.encode_plus(text, max_length=self.max_sequence_length, truncation=True,
                                                 padding="max_length", return_tensors="pt")
        encoder_txt = {k: v.squeeze(0) for k, v in encoder_txt.items()}
        global_labels = torch.zeros((self.max_sequence_length, self.max_sequence_length), dtype=torch.long)
        # 真实的长度，包含【CLS】【SEP】
        bert_input_length = torch.sum(encoder_txt["attention_mask"])
        # 构建2维mask矩阵
        grid_mask2d = torch.zeros((self.max_sequence_length, self.max_sequence_length), dtype=torch.bool)
        grid_mask2d[:bert_input_length, :bert_input_length] = True
        grid_mask2d = torch.triu(grid_mask2d)  # 上三角
        no_mask = []
        for span in entity_spans:
            start = span['start_idx']
            end = span['end_idx']
            label_id = span['type']
            entity = span['entity']
            if start in start_mapping and end in end_mapping:
                new_start = start_mapping[start]
                new_end = end_mapping[end]
                if new_start > new_end or entity == '':
                    continue
                global_labels[new_start, new_end] = label_id
                no_mask.append(new_start)
                no_mask.append(new_end)
        if self.do_mask:
            if np.random.random() >0.5:
                ix = torch.rand(size=(len(encoder_txt['input_ids']),)) < self.mask_p
                for i in no_mask:
                    ix[i] = False
                encoder_txt['input_ids'][ix] = self.mask_token
        inputs = {
            "input_ids": encoder_txt["input_ids"],
            'token_type_ids': encoder_txt["token_type_ids"],
            'attention_mask': encoder_txt["attention_mask"],
            'grid_mask2d': grid_mask2d,
            'labels': global_labels
        }
        return inputs


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length, do_mask, mask_p,pseudo_file_path=None, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            GaiicDataset.label2id(), tokenizer, max_sequence_length, do_mask, mask_p),
    ]
    return GaiicDataset(data_name, data_dir, data_type, process_piplines,pseudo_file_path, **kwargs)

def _param_optimizer(params, learning_rate, no_decay, weight_decay):
    _params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay,
         'lr': learning_rate},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': learning_rate},
    ]
    return _params

def get_optimizer_grouped_parameters(
        model,
        model_type='bert',
        learning_rate=5e-5,
        weight_decay=0.01,
        layerwise_learning_rate_decay=0.9
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


def get_optimizer(optimizer_grouped_parameters, opts):
    if opts.optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=opts.adam_epsilon,
            correct_bias=not opts.use_bertadam
        )
    elif opts.optimizer_type == "madgrad":
        pass

    elif opts.optimizer_type == 'adam':
        optimizer = Adam(optimizer_grouped_parameters,
                         eps=opts.adam_epsilon,
                         weight_decay=opts.weight_decay)
    else:
        optimizer = ''
    return optimizer


def get_loss_fct(opts):
    if opts.loss_type == 'ce':
        return nn.CrossEntropyLoss(reduction='mean')
    elif opts.loss_type == 'lsr':
        pass
        # return LabelSmoothingCE(eps=opts.lsr_eps)
    elif opts.loss_type == 'focal':
        pass

    elif opts.loss_type == 'pcl':
        return Poly1CrossEntropyLoss(opts.num_labels, epsilon=opts.pcl_epsilon, alpha=opts.pcl_alpha,reduction='mean')
    elif opts.loss_type == 'pfcl':
        return Poly1FocalLoss(num_classes=opts.num_labels,
                              epsilon=1.0,
                              alpha=0.25,
                              gamma=2.0,
                              reduction="mean")
    else:
        return ''


def rdrop_loss_fun(p, q, mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()
    loss = (p_loss + q_loss) / 2
    return loss

mse =nn.MSELoss()
def fuzhu_loss(logits,target,mask):
    y_pred = torch.argmax(logits, -1)
    y_pred[:, [0, -1]] = 0
    y_pred[:, :, [0, -1]] = 0
    y_pred = y_pred[mask]
    y_true = target[mask]
    l1 = (y_pred>0).float()
    l2 = (y_true > 0).float()
    return mse(l1,l2)

def train(opts, model, tokenizer, train_dataset, dev_dataset, logger):
    """ Train the model """
    opts.train_batch_size = opts.per_gpu_train_batch_size * max(1, opts.device_num)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=opts.train_batch_size,
                                  drop_last=opts.train_drop_last,
                                  collate_fn=train_dataset.collate_fn)

    t_total = len(train_dataloader) // opts.gradient_accumulation_steps * opts.num_train_epochs
    no_decay = ["bias", 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    if hasattr(model, opts.base_model_name) and opts.other_learning_rate != 0.0:
        msg = (f"The initial learning rate for model params : {opts.learning_rate} ,"
               f"and {opts.other_learning_rate}"
               )
        logger.info(msg)
        base_model = getattr(model, opts.base_model_name)
        base_model_param = list(base_model.named_parameters())

        if opts.grouped_parameters:
            optimizer_grouped_parameters.extend(
                get_optimizer_grouped_parameters(model=base_model, model_type='bert', layerwise_learning_rate_decay=0.9,
                                                 weight_decay=opts.weight_decay))
        else:
            optimizer_grouped_parameters.extend(
                _param_optimizer(base_model_param, opts.learning_rate, no_decay, opts.weight_decay))
        base_model_param_ids = [id(p) for n, p in base_model_param]
        other_model_param = [(n, p) for n, p in model.named_parameters() if
                             id(p) not in base_model_param_ids]

        optimizer_grouped_parameters.extend(
            _param_optimizer(other_model_param, opts.other_learning_rate, no_decay,
                             opts.weight_decay))
    else:
        all_model_param = list(model.named_parameters())
        optimizer_grouped_parameters.extend(
            _param_optimizer(all_model_param, opts.learning_rate, no_decay, opts.weight_decay))

    opts.warmup_steps = int(t_total * opts.warmup_proportion)
    optimizer = get_optimizer(optimizer_grouped_parameters, opts)
    scheduler_function = get_lr_scheduler(opts.scheduler_type)
    scheduler = scheduler_function(optimizer=optimizer, num_warmup_steps=opts.warmup_steps, num_training_steps=t_total)
    if opts.do_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer, swa_lr=opts.swa_learning_rate,
            anneal_epochs=opts.anneal_epochs,
            anneal_strategy=opts.anneal_strategy
        )
        logger.info(
            f"Total Training Steps: {t_total}, Total Warmup Steps: {opts.warmup_steps}, SWA Start Step: {opts.swa_start}")
    if opts.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=opts.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if opts.device_num > 1:
        model = torch.nn.DataParallel(model)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Options = %s", opts)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", opts.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", opts.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", opts.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    f1 = 0.0
    model.zero_grad()
    seed_everything(opts.seed)  # Added here for reproductibility (even between python 2 and 3)
    if opts.do_fgm:
        fgm = FGM(model, emb_name=opts.fgm_name, epsilon=opts.fgm_epsilon)
    if opts.do_pgd:
        pgd = PGD(model, emb_name='word_embeddings.', epsilon=opts.pgd_epsilon, alpha=opts.pgd_alpha)
    if opts.do_awp:
        from gpsrc.utils import AWP
        awp = AWP(model,
                  optimizer,
                  adv_lr=opts.awp_lr,
                  adv_eps=opts.awp_eps,
                  start_epoch=1
                  )
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=opts.num_train_epochs)
    if opts.save_steps == -1 and opts.logging_steps == -1:
        opts.logging_steps = len(train_dataloader) // opts.gradient_accumulation_steps
        opts.save_steps = len(train_dataloader) // opts.gradient_accumulation_steps
    loss_fct = get_loss_fct(opts)
    for epoch in range(int(opts.num_train_epochs)):
        pbar.epoch(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {key: value.to(opts.device) if key != 'grid_mask2d' else value for key, value in batch.items()}
            grid_mask2d = inputs['grid_mask2d'].clone()
            if opts.do_rdrop and (epoch+1) >opts.rdrop_epoch:
                outputs1 = model(**inputs)
                outputs2 = model(**inputs)
                loss1 = loss_fct(outputs1['logits'][grid_mask2d], inputs['labels'][grid_mask2d])
                loss2 = loss_fct(outputs2['logits'][grid_mask2d], inputs['labels'][grid_mask2d])
                loss3 = rdrop_loss_fun(outputs1['logits'][grid_mask2d], outputs2['logits'][grid_mask2d])
                loss_weight = (1 - opts.rdrop_weight) / 2
                loss = loss_weight * loss1 + loss_weight * loss2 + opts.rdrop_weight * loss3
            else:
                outputs = model(**inputs)
                loss = loss_fct(outputs['logits'][grid_mask2d], inputs['labels'][grid_mask2d])
                # fu_loss = fuzhu_loss(outputs['logits'], inputs['labels'],grid_mask2d)
                # loss = loss+fu_loss
                # import pdb
                # pdb.set_trace()
            if opts.device_num > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if opts.gradient_accumulation_steps > 1:
                loss = loss / opts.gradient_accumulation_steps
            if opts.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if opts.do_fgm:
                fgm.attack()
                adv_outputs = model(**inputs)
                loss_adv = loss_fct(adv_outputs['logits'][grid_mask2d], inputs['labels'][grid_mask2d])
                loss_adv = loss_adv.mean()
                if opts.fp16:
                    with amp.scale_loss(loss_adv, optimizer) as adv_scaled_loss:
                        adv_scaled_loss.backward()
                else:
                    loss_adv.backward()
                fgm.restore()
            if opts.do_pgd:
                pgd.backup_grad()
                # 对抗训练
                for t in range(opts.pgd_k):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != opts.pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_outputs = model(**inputs)
                    loss_adv = loss_fct(adv_outputs['logits'][grid_mask2d], inputs['labels'][grid_mask2d])
                    if opts.fp16:
                        with amp.scale_loss(loss_adv, optimizer) as adv_scaled_loss:
                            adv_scaled_loss.backward()
                    else:
                        loss_adv.backward()
                pgd.restore()  # 恢复embedding参
            pbar.step(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if opts.do_awp:
                if opts.awp_epoch>0:
                    if epoch > opts.awp_epoch:
                        awp.attack_backward(inputs, loss_fct, grid_mask2d, epoch + 1,opts.fp16)
                else:
                    if f1 > opts.awp_f1:#806
                        awp.attack_backward(inputs,loss_fct,grid_mask2d, epoch+1,opts.fp16)
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                if opts.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opts.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                optimizer.step()
                if not opts.do_swa:
                    scheduler.step()  # Update learning rate schedul
                else:
                    if (epoch + 1) < opts.swa_start:
                        scheduler.step()
                model.zero_grad()
                global_step += 1
                if opts.logging_steps > 0 and global_step % opts.logging_steps == 0:
                    result = evaluate(opts, model, dev_dataset, logger)
                    f1 = result
                if opts.save_steps > 0 and global_step % opts.save_steps == 0:
                    if opts.save_best:
                        output_dir = os.path.join(opts.output_dir, "checkpoint-best")
                    else:
                        output_dir = os.path.join(opts.output_dir, "checkpoint-{}-{}".format(round(result, 5), global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(opts, os.path.join(output_dir, "training_opts.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        if opts.do_swa and (epoch + 1) >= opts.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_result = evaluate(opts, swa_model, dev_dataset, logger, prefix='swa')
            output_dir_swa = os.path.join(opts.output_dir,
                                          "checkpoint-{}-{}-swa".format(round(swa_result, 5), global_step))
            if not os.path.exists(output_dir_swa) and opts.do_swa:
                os.makedirs(output_dir_swa)
            # model_to_save.save_pretrained(output_dir)
            torch.save(swa_model.module.state_dict(), f"{output_dir_swa}/pytorch_model.bin")
        if 'cuda' in str(opts.device):
            torch.cuda.empty_cache()

## metric计算
class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true,p_v):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, start, end in zip(*np.where(y_pred > p_v)):
            pred.append((b, y_pred[b,start,end], start, end))
        for b, start, end in zip(*np.where(y_true > p_v)):
            true.append((b,y_true[b,start,end], start, end))
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

def evaluate(opts, model, dev_dataset, logger, prefix="", save=False):
    metric = MetricsCalculator()
    opts.eval_batch_size = opts.per_gpu_eval_batch_size * max(1, opts.device_num)
    eval_sampler = SequentialSampler(dev_dataset)
    eval_dataloader = DataLoader(dev_dataset,
                                 sampler=eval_sampler,
                                 batch_size=opts.eval_batch_size,
                                 collate_fn=dev_dataset.collate_fn)
    eval_loss = 0.0
    nb_eval_steps = 0
    total_f1_, total_precision_, total_recall_ = 0., 0., 0.
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    loss_fct = get_loss_fct(opts)
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        with torch.no_grad():
            inputs = {key: value.to(opts.device) if key != 'grid_mask2d' else value for key, value in batch.items()}
            grid_mask2d = inputs['grid_mask2d'].clone()
            outputs = model(**inputs)
        tmp_eval_loss = loss_fct(outputs['logits'][grid_mask2d], inputs['labels'][grid_mask2d])
        logits = outputs['logits']
        if opts.device_num > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        predict_labels = torch.argmax(logits, -1)
        # 把predict_label的下三角排除
        mask = torch.tril(torch.ones_like(predict_labels), diagonal=-1)
        predict_labels = predict_labels - mask * 1e12
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        f1, p, r = metric.get_evaluate_fpr(predict_labels, inputs['labels'],0.0)
        total_f1_ += f1
        total_precision_ += p
        total_recall_ += r
        pbar.step(step)
    avg_f1 = total_f1_ / (len(eval_dataloader))
    avg_precision = total_precision_ / (len(eval_dataloader))
    avg_recall = total_recall_ / (len(eval_dataloader))
    eval_loss = eval_loss / nb_eval_steps
    logger.info("***** Eval results %s *****", prefix)
    logger.info(f"  %s = %s", 'f1', str(round( avg_f1, 5)))
    logger.info(f"  %s = %s", 'acc', str(round(avg_precision, 5)))
    logger.info(f"  %s = %s", 'recall', str(round(avg_recall, 5)))
    logger.info(f"  %s = %s", 'loss', str(round(eval_loss, 5)))
    return  avg_f1

def is_nested(chunk1, chunk2):
    a,b = chunk1, chunk2
    return min(a[1], b[1]) - max(a[0], b[0]) > 0

def detect_nested(chunks):
    for i, ck1 in enumerate(chunks):
        for ck2 in chunks[i+1:]:
            if is_nested(ck1, ck2):
                return True
    return False

def detect_not_nested(chunks):
    for i, ck1 in enumerate(chunks):
        for ck2 in chunks[i+1:]:
            if is_nested(ck1, ck2):
                continue
            else:
                return ck1,ck2

def detect_nested2(chunks):
    nest = []
    for i, ck1 in enumerate(chunks):
        for ck2 in chunks[i+1:]:
            if is_nested(ck1, ck2):
                if len(nest) ==0 :
                    nest.append([])
                    nest[-1].append(ck1)
                    nest[-1].append(ck2)
                    continue
                if ck1 in nest[-1]:
                    nest[-1].append(ck2)
                else:
                    nest.append([])
                    nest[-1].append(ck1)
                    nest[-1].append(ck2)
    return nest

def predict_one_sample(opts, model, tokens, tokenizer, threshold=0.0, prefix="", model_extra=None):
    ## 处理空格
    new_tokens = []
    for i, word in enumerate(tokens):
        tokenizer_word = tokenizer.tokenize(word)
        if len(tokenizer_word) == 0:
            new_tokens.append("^")
        else:
            new_tokens.append(word)
    # 重新生成text
    text = "".join(new_tokens)
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
    encoder_txt = tokenizer.encode_plus(text, padding="max_length", return_tensors="pt")
    inputs = {
        "input_ids": encoder_txt["input_ids"],
        'token_type_ids': encoder_txt["token_type_ids"],
        'attention_mask': encoder_txt["attention_mask"]
    }
    model.eval()
    with torch.no_grad():
        inputs = {key: value.to(opts.device) for key, value in inputs.items()}
        logits = model(**inputs)['logits'].cpu()
    if model_extra is not None:
        for extra in model_extra:
            extra.eval()
            with torch.no_grad():
                logits += extra(**inputs)['logits'].cpu()
        logits = logits / float(len(model_extra) + 1)
    logits[:, [0, -1]] -= np.inf
    logits[:, :, [0, -1]] -= np.inf
    y_pred = torch.argmax(logits, -1).cpu().numpy()
    y_pred[:, [0, -1]] = 0
    y_pred[:, :, [0, -1]] = 0
    entities = []

    for b, start, end in zip(*np.where(y_pred > 0)):
        # if end-start>=20: # 伪标签出错
        #     continue
        category = y_pred[b, start, end]
        label_logits = logits[b,start,end,category]
        if end - 1 > token2char_span_mapping[-2][-1]:
            break
        if token2char_span_mapping[start][0] <= token2char_span_mapping[end][-1]:
            # 左闭右闭
            entitie_ = [token2char_span_mapping[start][0], token2char_span_mapping[end][-1] - 1,
                        opts.id2label[category],
                        text[token2char_span_mapping[start][0]: token2char_span_mapping[end][-1]]
                        ]
            if entitie_[-1] == '':
                continue
            entities.append(entitie_)
    # if '可擦钢笔魔力擦暗尖包尖学生专用热可擦刚笔细笔尖小学生三年级男生晶蓝色墨蓝黑色热敏可擦墨囊可擦笔摩擦笔^可擦钢笔【3支纹理' in text:
    #     import pdb
    #     pdb.set_trace()
    chunks = copy.deepcopy(entities)
    overlaps = detect_nested2(chunks)
    if len(overlaps) == 0:
        return entities
    else:
        for op in overlaps:
            for z in op:
                chunks.pop(chunks.index(z))
            ts = list(set([x[2] for x in op]))
            if len(op) == 2:
                if len(ts) == 1:
                    ls = [x[1] - x[0] + 1 for x in op]
                    index = ls.index(min(ls))
                    chunks.append(op[index])
                if len(ts) == 2:
                    ls = [x[1] - x[0] + 1 for x in op]
                    index = ls.index(max(ls))
                    chunks.append(op[index])
            if len(op) == 3:
                ck1, ck2 = detect_not_nested(op)
                chunks.append(ck1)
                chunks.append(ck2)
        chunks = sorted(chunks, key=lambda x: (x[0], x[1]))
        return chunks

MODEL_CLASSES = {
    "nezha": (NeZhaConfig, GlobalPointerNeZha, BertTokenizerFast),
}

def main():
    parser = Argparser.get_training_parser()
    group = parser.add_argument_group(title="global pointer", description="Global pointer")
    group.add_argument('--inner_dim', default=64, type=int)
    group.add_argument('--use_rope', action='store_true')

    group.add_argument("--embed_dropout", type=float, default=0.0)
    group.add_argument("--post_lstm_dropout", type=float, default=0.0)
    group.add_argument("--conv_dropout", type=float, default=0.0)
    group.add_argument("--mlp_dropout", type=float, default=0.0)
    group.add_argument("--out_dropout", type=float, default=0.0)
    group.add_argument("--s_e_dropout", type=float, default=0.0)

    group.add_argument("--use_last_n_layers", type=int, default=None)
    group.add_argument("--do_lstm", action="store_true")
    group.add_argument("--num_lstm_layers", type=int, default=1)
    group.add_argument("--do_fgm", action="store_true",
                       help="Whether to adversarial training.")

    group.add_argument('--fgm_epsilon', default=1.0, type=float,
                       help="Epsilon for adversarial.")
    group.add_argument('--fgm_name', default='word_embeddings', type=str,
                       help="name for adversarial layer.")

    group.add_argument("--do_swa", action="store_true")
    group.add_argument("--optimizer_type", default='adamw', choices=['adamw', 'madgrad', 'adam'])
    group.add_argument("--use_bertadam", action="store_true")
    group.add_argument('--swa_learning_rate', default=1.0, type=float,
                       help="Epsilon for adversarial.")
    group.add_argument('--anneal_epochs', default=3, type=int,
                       help="Epsilon for adversarial.")
    group.add_argument('--swa_start', default=3, type=int,
                       help="Epsilon for adversarial.")
    group.add_argument('--anneal_strategy', default='cos', type=str,
                       help="Epsilon for adversarial.")

    group.add_argument("--do_multi_drop", action="store_true")
    group.add_argument('--grouped_parameters', action="store_true")
    group.add_argument('--do_weight_layer_pooling', action="store_true")
    group.add_argument('--layer_start', default=9, type=int)
    group.add_argument("--do_pgd", action="store_true",
                       help="Whether to adversarial training.")
    group.add_argument('--pgd_epsilon', default=1.0, type=float,
                       help="Epsilon for adversarial.")
    group.add_argument('--pgd_k', default=3, type=int,
                       help="Epsilon for adversarial.")
    group.add_argument('--pgd_alpha', default=0.3, type=float,
                       help="Epsilon for adversarial.")
    group.add_argument('--do_rdrop', action="store_true")
    group.add_argument('--rdrop_weight', type=float, default=0.1)
    group.add_argument('--rdrop_epoch',default=1,type=int)
    group.add_argument('--train_drop_last', action="store_true")
    group.add_argument('--conv_hid_size', default=80, type=int)
    group.add_argument('--ffnn_hid_size', default=128, type=int)
    group.add_argument('--biaffine_size', default=512, type=int)
    group.add_argument('--loss_type', default='ce', type=str)
    group.add_argument('--lsr_eps', default=0.1, type=float)
    group.add_argument('--focal_gamma', default=2.0, type=float)
    group.add_argument('--focal_alpha', default=0.25, type=float)
    group.add_argument('--merge_mode', default='zero', type=str)
    group.add_argument('--do_mask', action="store_true")
    group.add_argument('--mask_p', default=0.15, type=float)
    group.add_argument('--eval_checkpoint_path', type=str)
    group.add_argument('--eval_checkpoint_path_extra', type=str, nargs="+", default=None)
    group.add_argument('--pcl_epsilon',type=float,default=1.0)
    group.add_argument('--pcl_alpha',type=float,default=0.0)
    group.add_argument('--do_awp', action="store_true")
    group.add_argument('--awp_lr', type=float, default=0.0005)
    group.add_argument('--awp_eps', type=float, default=0.001)
    group.add_argument('--awp_epoch',default=0,type=int)
    group.add_argument('--awp_f1',default=0.810,type=float)
    group.add_argument('--do_post_swa', action="store_true")
    group.add_argument('--swa_model_dir',default='',type=str)
    group.add_argument('--do_predict_test',action='store_true')
    group.add_argument('--submit_file_path',default='',type=str)
    group.add_argument('--save_best', action="store_true")
    group.add_argument('--do_pseudo',action='store_true')
    group.add_argument('--pseudo_data_file',default='',type=str)
    group.add_argument('--pseudo_label_file', default='', type=str)
    opts = parser.parse_args_from_parser(parser)
    logger = Logger(opts=opts)
    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]
    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    if not opts.do_predict_test:
        train_dataset = load_data(opts.train_input_file, opts.data_dir, "train", tokenizer, opts.train_max_seq_length,
                                do_mask=opts.do_mask, mask_p=opts.mask_p)
        dev_dataset = load_data(opts.eval_input_file, opts.data_dir, "dev", tokenizer, opts.eval_max_seq_length,
                                do_mask=False, mask_p=opts.mask_p)
    opts.label2id = GaiicDataset.label2id()
    opts.id2label = GaiicDataset.id2label()
    opts.num_labels = len(GaiicDataset.get_labels())
    # model
    logger.info("initializing model and config")
    config, unused_kwargs = config_class.from_pretrained(opts.pretrained_model_path,
                                                         return_unused_kwargs=True,
                                                         use_rope=opts.use_rope,
                                                         num_labels=opts.num_labels,
                                                         id2label=opts.id2label,
                                                         label2id=opts.label2id,
                                                         output_hidden_states=True,
                                                         do_lstm=opts.do_lstm,
                                                         do_multi_drop=opts.do_multi_drop,
                                                         num_lstm_layers=opts.num_lstm_layers,
                                                         use_last_n_layers=opts.use_last_n_layers,
                                                         post_lstm_dropout=opts.post_lstm_dropout,
                                                         embed_dropout=opts.embed_dropout,
                                                         do_weight_layer_pooling=opts.do_weight_layer_pooling,
                                                         layer_start=opts.layer_start,
                                                         do_rdrop=opts.do_rdrop,
                                                         rdrop_weight=opts.rdrop_weight,
                                                         train_drop_last=opts.train_drop_last,
                                                         conv_hid_size=opts.conv_hid_size,
                                                         dilation=[1, 2,4,1,1],
                                                         ffnn_hid_size=opts.ffnn_hid_size,
                                                         merge_mode=opts.merge_mode,
                                                         biaffine_size=opts.biaffine_size,
                                                         conv_dropout=opts.conv_dropout,
                                                         mlp_dropout=opts.mlp_dropout,
                                                         out_dropout=opts.out_dropout,
                                                         s_e_dropout=opts.s_e_dropout,
                                                         inner_dim=opts.inner_dim)
    # FIXED: 默认`from_dict`中，只有config中有键才能设置值，这里强制设置
    for key, value in unused_kwargs.items(): setattr(config, key, value)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    # do train
    if opts.do_train:
        train(opts=opts, model=model, tokenizer=tokenizer, logger=logger, train_dataset=train_dataset, dev_dataset=dev_dataset)
    if opts.do_eval:
        model = model_class.from_pretrained(opts.eval_checkpoint_path, config=config)
        model.to(opts.device)
        evaluate(opts, model, dev_dataset, logger=logger, save=True)
    # 预测dev定位为主
    if opts.do_predict:
        model = model_class.from_pretrained(opts.eval_checkpoint_path, config=config)
        model.to(opts.device)
        from tqdm import tqdm
        predict_results = []
        with open(os.path.join(opts.data_dir, opts.eval_input_file), 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                line = json.loads(line)
                tokens = line['tokens']
                true_entity = line['ner_tags']
                pre_entity = predict_one_sample(opts, model, tokens, tokenizer)
                entity_spans = []
                for _type, _start_idx, _end_idx in get_entity_biob(true_entity, None):
                    entity_spans.append([_start_idx, _end_idx, _type, "".join(tokens[_start_idx:_end_idx + 1])])
                tmp_dict = {}
                tmp_dict['text'] = tokens
                tmp_dict['true'] = entity_spans
                tmp_dict['pred'] = pre_entity
                predict_results.append(tmp_dict)
        torch.save(predict_results, os.path.join(opts.output_dir, opts.eval_input_file + ".predict"))

    if opts.do_post_swa:
        import copy
        def get_model_path_list(base_dir):
            """
            从文件夹中获取 model.pt 的路径
            """
            model_lists = []
            for root, dirs, files in os.walk(base_dir):
                for _file in files:
                    if 'pytorch_model.bin' == _file:
                        model_lists.append(os.path.join(root, _file))
            model_lists = [x for x in model_lists if 'checkpoint-' in x and 'swa' not in x]
            model_lists = sorted(model_lists,
                                 key=lambda x: float(x.split('/')[-2].split('-')[-2]),reverse=False)
            return model_lists
        model_path_list = get_model_path_list(opts.swa_model_dir)
        # assert 1 <= swa_start < len(model_path_list) - 1, \
        #     f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'
        swa_model = copy.deepcopy(model)
        swa_n = 0.
        model_num = len(model_path_list[-opts.swa_start:])
        with torch.no_grad():
            for _ckpt in model_path_list[-opts.swa_start:]:
                logger.info(f'Load model from {_ckpt}')
                # import pdb
                # pdb.set_trace()
                # model.load_state_dict(torch.load(_ckpt))
                # tmp_para_dict = dict(model.named_parameters())
                tmp_para_dict = dict(torch.load(_ckpt))
                alpha = 1. / (swa_n + 1.)
                for name, para in swa_model.named_parameters():
                    para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
                swa_n += 1
        # with torch.no_grad():
        #     for _ckpt in model_path_list[-opts.swa_start:]:
        #         logger.info(f'Load model from {_ckpt}')
        #         # model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
        #         tmp_para_dict = dict(torch.load(_ckpt))
        #         if swa_n == 0:
        #             for name, para in swa_model.named_parameters():
        #                 para.copy_(tmp_para_dict[name].data.clone() / model_num)
        #         else:
        #             for name, para in swa_model.named_parameters():
        #                 para.copy_(tmp_para_dict[name].data.clone() / model_num + para.data.clone())
        #         swa_n += 1
        swa_f1 = evaluate(opts,swa_model,dev_dataset,logger,prefix='swa')
        # use 100000 to represent swa to avoid clash
        swa_model_dir = os.path.join(opts.swa_model_dir, f'checkpoint-swa-{swa_f1}-{opts.swa_start}')
        if not os.path.exists(swa_model_dir):
            os.mkdir(swa_model_dir)
        logger.info(f'Save swa model in: {swa_model_dir}')
        swa_model.save_pretrained(swa_model_dir)
        #swa_model_path = os.path.join(swa_model_dir, 'pytorch_model.bin')
        #torch.save(swa_model.state_dict(), swa_model_path)

    if opts.do_predict_test:
        from tqdm import tqdm
        model = model_class.from_pretrained(opts.eval_checkpoint_path, config=config)
        model.to(opts.device)
        model_extra = None
        if opts.eval_checkpoint_path_extra is not None:
            logger.warning(f'Using the same model config!!!')
            model_extra = []
            for checkpoint_path in opts.eval_checkpoint_path_extra:
                model_extra.append(model_class.from_pretrained(checkpoint_path, config=config))
                model_extra[-1].to(opts.device)
        predict_results = []
        with open(opts.test_input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) == 80000: lines = lines[: 40000] # FIXED: 71333解码报错，用4W条 ValueError: [29, 39, '38', 'X20X21x23z3'] is not in list
            for line in tqdm(lines, total=len(lines)):
                line = line.strip("\n")
                tokens = list(line)
                label = len(tokens) * ['O']
                entitys = predict_one_sample(opts,model,tokens,tokenizer,model_extra=model_extra)
                for _preditc in entitys:
                    label[_preditc[0]] = 'B-' + _preditc[2]
                    label[(_preditc[0] + 1): (_preditc[1]+1)] = (_preditc[1] - _preditc[0]) * [('I-' + _preditc[2])]
                predict_results.append([line, label])
        with open(opts.submit_file_path, 'w', encoding='utf-8') as f:
            for _result in predict_results:
                for word, tag in zip(_result[0], _result[1]):
                    if word == '\n':
                        continue
                    f.write(f'{word} {tag}\n')
                f.write('\n')
    if opts.do_pseudo:
        model = model_class.from_pretrained(opts.eval_checkpoint_path, config=config)
        model.to(opts.device)
        predict_results = []
        from tqdm import tqdm
        with open(opts.pseudo_data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip("\n")
                tokens = list(line)
                label = len(tokens) * ['O']
                entitys = predict_one_sample(opts,model,tokens,tokenizer)
                for _preditc in entitys:
                    label[_preditc[0]] = 'B-' + _preditc[2]
                    label[(_preditc[0] + 1): (_preditc[1]+1)] = (_preditc[1] - _preditc[0]) * [('I-' + _preditc[2])]
                predict_results.append([line, label])
        with open(opts.pseudo_label_file, 'w', encoding='utf-8') as f:
            for _result in predict_results:
                for word, tag in zip(_result[0], _result[1]):
                    if word == '\n':
                        continue
                    f.write(f'{word} {tag}\n')
                f.write('\n')


if __name__ == "__main__":
    main()
