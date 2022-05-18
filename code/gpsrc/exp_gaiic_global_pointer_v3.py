import os
import torch
import json
import numpy as np
from torch import nn
from torch.nn import Module
from transformers import BertTokenizerFast
from dataset import DatasetBase
from utils import get_entity_biob, FGM, PGD
from seed import seed_everything
from device import prepare_device
from options import Argparser
from logger import Logger
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
from lr_scheduler import get_lr_scheduler
from adamw import AdamW
from progressbar import ProgressBar
from torch.optim import Adam
from torch.nn import functional as F
from nezha.modeling_nezha import NeZhaConfig, NeZhaModel, NeZhaPreTrainedModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from layers import GlobalPointer, SpatialDropout, LayerNorm, ConvolutionLayer, WeightedLayerPooling, MLP, Biaffine
from polyloss import Poly1CrossEntropyLoss, Poly1FocalLoss
from ema import EMA
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


class GlobalPointerNeZhaV2(NeZhaPreTrainedModel):
    '''
    gp+conv_ffn方案
    '''
    def __init__(self, config):
        super(GlobalPointerNeZhaV2, self).__init__(config)
        self.num_labels = config.num_labels
        self.inner_dim = config.inner_dim
        self.use_rope = config.use_rope
        self.hidden_size = config.hidden_size
        self.use_last_n_layers = config.use_last_n_layers
        self.do_lstm = config.do_lstm
        self.conv_hid_size = config.conv_hid_size
        self.conv_dropout = config.conv_dropout
        self.mlp_dropout = config.mlp_dropout
        self.out_dropout = config.out_dropout
        self.bert = NeZhaModel(config)
        self.global_pointer = GlobalPointer(self.num_labels, self.inner_dim, self.hidden_size, self.use_rope)
        if self.do_lstm:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=config.num_lstm_layers,
                                batch_first=True, bidirectional=True)
        self.embedding_dropout = SpatialDropout(config.embed_dropout)
        self.do_weight_layer_pooling = config.do_weight_layer_pooling
        if self.do_weight_layer_pooling:
            self.WLP = WeightedLayerPooling(
                config.num_hidden_layers,
                layer_start=config.layer_start, layer_weights=None
            )
        self.post_lstm_dropout = nn.Dropout(config.post_lstm_dropout)
        self.convLayer = ConvolutionLayer(config.hidden_size, config.conv_hid_size, config.dilation, self.conv_dropout)
        self.cln = LayerNorm(config.hidden_size, config.hidden_size, conditional=True)
        self.mlp_rel = MLP(self.conv_hid_size * len(config.dilation), config.ffnn_hid_size, dropout=self.mlp_dropout)
        self.linear = nn.Linear(config.ffnn_hid_size, self.num_labels)
        self.dropout = nn.Dropout(self.out_dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, grid_mask2d=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=None,
            inputs_embeds=None,
        )
        sequence_output = outputs[0]
        if self.do_weight_layer_pooling:
            sequence_output = self.WLP(all_hidden_states=outputs[2])
        else:
            if self.use_last_n_layers is not None:
                last_hidden_state = outputs[2]
                last_hidden_state = torch.stack(last_hidden_state[- self.use_last_n_layers:], dim=-1)
                last_hidden_state = torch.mean(last_hidden_state, dim=-1)
                sequence_output = last_hidden_state
        sequence_output = self.embedding_dropout(sequence_output)
        if self.do_lstm:
            sequence_lengths = attention_mask.sum(dim=1)
            packed_sequence_output = pack_padded_sequence(
                sequence_output, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_sequence_output, _ = self.lstm(packed_sequence_output)
            unpacked_sequence_output, _ = pad_packed_sequence(
                packed_sequence_output, batch_first=True, total_length=sequence_lengths.max())
            sequence_output = sequence_output + unpacked_sequence_output
        sequence_output = self.post_lstm_dropout(sequence_output)

        cln = self.cln(sequence_output.unsqueeze(2), sequence_output)
        conv_inputs = torch.masked_fill(cln, grid_mask2d.eq(0).unsqueeze(-1).to(input_ids.device), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1).to(input_ids.device),
                                         0.0)  # (2,16,16,240)
        ffn_outputs = self.dropout(self.mlp_rel(conv_outputs))
        ffn_outputs = self.linear(ffn_outputs)
        gp_logits = self.global_pointer(sequence_output, mask=attention_mask)
        logits = gp_logits + ffn_outputs
        outputs = {}
        outputs['logits'] = logits
        return outputs


class GlobalPointerNeZhaV3(NeZhaPreTrainedModel):
    '''
    gp+conv_ffn+biaffine方案
    '''
    def __init__(self, config):
        super(GlobalPointerNeZhaV3, self).__init__(config)
        self.num_labels = config.num_labels
        self.inner_dim = config.inner_dim
        self.use_rope = config.use_rope
        self.hidden_size = config.hidden_size
        self.use_last_n_layers = config.use_last_n_layers
        self.do_lstm = config.do_lstm
        self.conv_hid_size = config.conv_hid_size
        self.conv_dropout = config.conv_dropout
        self.mlp_dropout = config.mlp_dropout
        self.out_dropout = config.out_dropout
        self.s_e_dropout = config.s_e_dropout
        self.bert = NeZhaModel(config)
        self.global_pointer = GlobalPointer(self.num_labels, self.inner_dim, self.hidden_size, self.use_rope)
        if self.do_lstm:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=config.num_lstm_layers,
                                batch_first=True, bidirectional=True)
        self.embedding_dropout = SpatialDropout(config.embed_dropout)
        self.do_weight_layer_pooling = config.do_weight_layer_pooling
        if self.do_weight_layer_pooling:
            self.WLP = WeightedLayerPooling(
                config.num_hidden_layers,
                layer_start=config.layer_start, layer_weights=None
            )
        self.post_lstm_dropout = nn.Dropout(config.post_lstm_dropout)
        self.convLayer = ConvolutionLayer(config.hidden_size, config.conv_hid_size, config.dilation, self.conv_dropout)
        self.cln = LayerNorm(config.hidden_size, config.hidden_size, conditional=True)
        self.mlp_rel = MLP(self.conv_hid_size * len(config.dilation), config.ffnn_hid_size, dropout=self.mlp_dropout)
        self.linear = nn.Linear(config.ffnn_hid_size, self.num_labels)
        self.dropout = nn.Dropout(self.out_dropout)
        self.mlp1 = MLP(n_in=self.hidden_size, n_out=config.biaffine_size, dropout=self.s_e_dropout)
        self.mlp2 = MLP(n_in=self.hidden_size, n_out=config.biaffine_size, dropout=self.s_e_dropout)
        self.biaffine = Biaffine(n_in=config.biaffine_size, n_out=self.num_labels, bias_x=True, bias_y=True)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, grid_mask2d=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=None,
            inputs_embeds=None,
        )
        sequence_output = outputs[0]
        if self.do_weight_layer_pooling:
            sequence_output = self.WLP(all_hidden_states=outputs[2])
        else:
            if self.use_last_n_layers is not None:
                last_hidden_state = outputs[2]
                last_hidden_state = torch.stack(last_hidden_state[- self.use_last_n_layers:], dim=-1)
                last_hidden_state = torch.mean(last_hidden_state, dim=-1)
                sequence_output = last_hidden_state
        sequence_output = self.embedding_dropout(sequence_output)
        if self.do_lstm:
            sequence_lengths = attention_mask.sum(dim=1)
            packed_sequence_output = pack_padded_sequence(
                sequence_output, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_sequence_output, _ = self.lstm(packed_sequence_output)
            unpacked_sequence_output, _ = pad_packed_sequence(
                packed_sequence_output, batch_first=True, total_length=sequence_lengths.max())
            sequence_output = sequence_output + unpacked_sequence_output
        sequence_output = self.post_lstm_dropout(sequence_output)

        cln = self.cln(sequence_output.unsqueeze(2), sequence_output)
        conv_inputs = torch.masked_fill(cln, grid_mask2d.eq(0).unsqueeze(-1).to(input_ids.device), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1).to(input_ids.device),
                                         0.0)  # (2,16,16,240)
        ffn_outputs = self.dropout(self.mlp_rel(conv_outputs))
        ffn_outputs = self.linear(ffn_outputs)
        gp_logits = self.global_pointer(sequence_output, mask=attention_mask)

        ent_start = self.dropout(self.mlp1(sequence_output))  # (2,16,512)
        ent_end = self.dropout(self.mlp2(sequence_output))  # (2,16,512)

        biaff_logits = self.biaffine(ent_start, ent_end)
        logits = (gp_logits + ffn_outputs + biaff_logits) / 3
        outputs = {}
        outputs['logits'] = logits
        return outputs


######################################## 数据处理
class GaiicDataset(DatasetBase):
    keys_to_truncate_on_dynamic_batch = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'grid_mask2d']

    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines,
                 **kwargs):
        super().__init__(data_name, data_dir, data_type, process_piplines, **kwargs)

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
        # if data_type=='train':
        #     i = 0
        #     with open('../data/tmp_data/10_folds_data/train.0.aug.jsonl') as fr:
        #         for line in fr.readlines():
        #             guid = f"{data_type}-aug-{i}"
        #             line = json.loads(line)
        #             examples.append(dict(guid=guid, tokens=line['tokens'], labels=line['entities']))
        #             i = i+1
        print("all data size: ",len(examples))
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
        guid = example['guid']
        ## 处理空格以及异常符号
        new_tokens = []
        for i, word in enumerate(tokens):
            tokenizer_word = self.tokenizer.tokenize(word)
            if len(tokenizer_word) == 0:
                new_tokens.append("^")
            else:
                new_tokens.append(word)
        # 获取实体的span
        if 'aug' in guid:
            entity_spans = []
            for ent in labels:
                entity_spans.append({  # 左闭右闭
                    'start_idx': ent[0],
                    'end_idx': ent[1],
                    'type': self.label2id[ent[2]],
                    'entity': "".join(tokens[ent[0]:ent[1] + 1])
                })
        else:
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


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length, do_mask, mask_p, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            GaiicDataset.label2id(), tokenizer, max_sequence_length, do_mask, mask_p),
    ]
    return GaiicDataset(data_name, data_dir, data_type, process_piplines, **kwargs)


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
        return nn.CrossEntropyLoss()
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


def train(opts, model, teacher_model,train_dataset, dev_dataset, logger):
    """ Train the model """
    teacher_model.model.eval()
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
    model.zero_grad()
    seed_everything(opts.seed)  # Added here for reproductibility (even between python 2 and 3)
    if opts.do_fgm:
        fgm = FGM(model, emb_name=opts.fgm_name, epsilon=opts.fgm_epsilon)
    if opts.do_pgd:
        pgd = PGD(model, emb_name='word_embeddings.', epsilon=opts.pgd_epsilon, alpha=opts.pgd_alpha)
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
            if opts.do_rdrop:
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
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参
            pbar.step(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                if opts.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opts.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                optimizer.step()
                teacher_model.update_params(model)
                teacher_model.apply_shadow()
                if not opts.do_swa:
                    scheduler.step()  # Update learning rate schedul
                else:
                    if (epoch + 1) < opts.swa_start:
                        scheduler.step()
                model.zero_grad()
                global_step += 1
                if opts.logging_steps > 0 and global_step % opts.logging_steps == 0:
                    result = evaluate(opts, model, dev_dataset, logger)
                    result = evaluate(opts, teacher_model, dev_dataset, logger,prefix='ema')
                if opts.save_steps > 0 and global_step % opts.save_steps == 0:
                    output_dir = os.path.join(opts.output_dir, "checkpoint-{}-{}".format(round(result, 5), global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
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
            torch.save(swa_model.module.state_dict(), f"{output_dir_swa}/pytorch_model.bin")
        if 'cuda' in str(opts.device):
            torch.cuda.empty_cache()

## metric计算
from collections import Counter

class MetricsCalculator2(object):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def update(self, y_pred, y_true, logits):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, start, end in zip(*np.where(y_pred > 0)):
            pred.append((y_pred[b, start, end], start, end))
        for b, start, end in zip(*np.where(y_true > 0)):
            true.append((y_true[b, start, end], start, end))
        self.origins.extend(true)
        self.founds.extend(pred)
        self.rights.extend([pre_entity for pre_entity in pred if pre_entity in true])

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

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

def evaluate2(opts, model, dev_dataset, logger, prefix="", save=False):
    metric = MetricsCalculator()
    opts.eval_batch_size = opts.per_gpu_eval_batch_size * max(1, opts.device_num)
    eval_sampler = SequentialSampler(dev_dataset)
    eval_dataloader = DataLoader(dev_dataset,
                                 sampler=eval_sampler,
                                 batch_size=opts.eval_batch_size,
                                 collate_fn=dev_dataset.collate_fn)
    eval_loss = 0.0
    nb_eval_steps = 0
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
        metric.update(predict_labels, inputs['labels'], logits)
        pbar.step(step)
    eval_result, class_info = metric.result()
    eval_loss = eval_loss / nb_eval_steps
    logger.info("***** Eval results %s *****", prefix)
    logger.info(f"  %s = %s", 'f1', str(round(eval_result['f1'], 5)))
    logger.info(f"  %s = %s", 'acc', str(round(eval_result['acc'], 5)))
    logger.info(f"  %s = %s", 'recall', str(round(eval_result['recall'], 5)))
    logger.info(f"  %s = %s", 'loss', str(round(eval_loss, 5)))
    return eval_result['f1']

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
        # metric.update(predict_labels, inputs['labels'], logits)
        f1, p, r = metric.get_evaluate_fpr(predict_labels, inputs['labels'],0.0)
        total_f1_ += f1
        total_precision_ += p
        total_recall_ += r
        pbar.step(step)

        # pbar.step(step)
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

def predict_one_sample(opts, model, tokens, tokenizer, threshold=0.0, prefix=""):
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
    logits[:, [0, -1]] -= np.inf
    logits[:, :, [0, -1]] -= np.inf
    y_pred = torch.argmax(logits, -1).cpu().numpy()
    entities = []
    for b, start, end in zip(*np.where(y_pred > 0)):
        category = y_pred[b, start, end]
        if end - 1 > token2char_span_mapping[-2][-1]:
            break
        if token2char_span_mapping[start][0] <= token2char_span_mapping[end][-1]:
            # 左闭右闭
            entitie_ = [token2char_span_mapping[start][0], token2char_span_mapping[end][-1] - 1,
                        opts.id2label[category],
                        text[token2char_span_mapping[start][0]: token2char_span_mapping[end][-1]],
                        ]
            if entitie_[-1] == '':
                continue
            entities.append(entitie_)
    return entities


MODEL_CLASSES = {
    "nezha": (NeZhaConfig, GlobalPointerNeZha, BertTokenizerFast),
    "nezhav2": (NeZhaConfig, GlobalPointerNeZhaV2, BertTokenizerFast),
    "nezhav3": (NeZhaConfig, GlobalPointerNeZhaV3, BertTokenizerFast),
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
    group.add_argument('--pcl_epsilon',type=float,default=1.0)
    group.add_argument('--pcl_alpha',type=float,default=0.0)
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
    train_dataset = load_data(opts.train_input_file, opts.data_dir, "train", tokenizer, opts.train_max_seq_length,
                              do_mask=opts.do_mask, mask_p=opts.mask_p)
    dev_dataset = load_data(opts.eval_input_file, opts.data_dir, "dev", tokenizer, opts.eval_max_seq_length,
                            do_mask=False, mask_p=opts.mask_p)
    opts.num_labels = train_dataset.num_labels
    opts.label2id = GaiicDataset.label2id()
    opts.id2label = GaiicDataset.id2label()
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
                                                         dilation=[1, 2, 3],
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
    teacher_model = EMA(model)
    # trainer
    logger.info("initializing traniner")
    # do train
    if opts.do_train:
        train(opts=opts, model=model, teacher_model=teacher_model,logger=logger, train_dataset=train_dataset, dev_dataset=dev_dataset)
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


if __name__ == "__main__":
    main()
