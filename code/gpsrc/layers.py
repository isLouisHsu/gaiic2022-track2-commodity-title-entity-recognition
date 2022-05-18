import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x

class SinusoidalPositionEmbedding(Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
            self,
            output_dim,
            merge_mode='add',
            custom_position_ids=False
    ):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        input_shape = inputs.shape
        _, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)

def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)

def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits

class GlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, head_size, inner_dim, hidden_size, RoPE=True):
        super(GlobalPointer, self).__init__()
        self.inner_dim = inner_dim
        self.head_size = head_size  # num_label
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size, self.head_size * self.inner_dim * 2)

    def forward(self, inputs, mask=None):
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.inner_dim * 2, dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.inner_dim], inputs[..., self.inner_dim:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(inputs)
            cos_pos = pos[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        logits = add_mask_tril(logits, mask)
        logits / self.inner_dim ** 0.5
        logits = logits.permute(0, 2, 3, 1)  # (bs,sequence,sequence,label_num)
        # scale返回
        return logits

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
        self.initialize_weights()
    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)
    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            # nn.GELU(),
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)
        x = F.gelu(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)
        return s
# weight layer pooling
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_hidden_states = torch.stack(all_hidden_states)
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        # self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        # x = self.activation(x)
        x = F.gelu(x)
        return x


# class MultiSampleDropout(nn.Module):
#     def __init__(self, n_drops=None, p_drops=None):
#         super().__init__()
#
#         self.q_0 = 0.10
#         self.q_1 = 0.50 - self.q_0
#
#         self.n_drops = n_drops or cfg.N_DROPS
#         self.p_drops = (p_drops or cfg.P_DROPS) or self.gen_dropout_probas()
#
#         self.drop_modules = nn.ModuleList([nn.Dropout(p_drop) for p_drop in self.p_drops])
#
#     def gen_dropout_probas(self):
#         assert self.n_drops >= 0
#
#         if self.n_drops == 0:
#             return []
#         elif self.n_drops == 1:
#             return [self.q_0]
#         else:
#             return [self.q_0 + self.q_1 * n / (self.n_drops - 1) for n in range(self.n_drops)]
#
#     def forward(self, x):
#         if not self.training or not self.n_drops:
#             return x[:, None]
#
#         res = []
#         for drop_module in self.drop_modules:
#             res.append(drop_module(x))
#         res = torch.stack(res, dim=1)
#         return res
#
#     def mask_input_ids(self, input_ids):
#
#         p_mask_size = np.random.uniform(cfg.P_MASK_SIZE_LOW, cfg.P_MASK_SIZE_HIGH)
#
#         n_masked = int(p_mask_size * len(input_ids))
#
#         if n_masked > 0:
#             index = np.random.choice(len(input_ids), size=n_masked, replace=False)
#             input_ids[index] = self.mask_token_id
#
#         return input_ids


class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        self.LSTM=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1=nn.Dropout(0.2)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.2)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x)

''' B,L=x.shape
        x=x.reshape(B,L,-1)
        x=self.lstm(x.permute(1,0,2)).permute(1,0,2)

'''
