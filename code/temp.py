class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

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
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2

class SpanClassificationHeadW2NER(SpanClassificationMixin):

    def __init__(self, hidden_size, num_labels, 
        dist_emb_size, type_emb_size, 
        conv_hid_size, conv_dilation, conv_dropout, 
        ffnn_hid_size, ffnn_hid_dropout,
        **kwargs
    ):
        super().__init__()
        self.num_labels = num_labels
        self.dist_embeddings = nn.Embedding(512, dist_emb_size)
        self.type_embeddings = nn.Embedding(3, type_emb_size)

        conv_input_size = hidden_size * 2 + dist_emb_size + type_emb_size
        self.conv = ConvolutionLayer(conv_input_size, conv_hid_size, conv_dilation, conv_dropout)
        self.predictor = CoPredictor(num_labels, hidden_size, hidden_size // 4,
                                     conv_hid_size * len(conv_dilation), ffnn_hid_size,
                                     ffnn_hid_dropout)

    def forward(self, sequence_output, attention_mask, spans):
        batch_size, sequence_length, hidden_size = sequence_output.size()
        
        sta_ids = torch.arange(sequence_length).unsqueeze(1).to(sequence_output.device)
        end_ids = torch.arange(sequence_length).unsqueeze(0).to(sequence_output.device)
        dist_ids = (end_ids - sta_ids + sequence_length) \
            .unsqueeze(0).expand(batch_size, sequence_length, sequence_length)
        extend_attention_mask = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
        
        conv_inputs = torch.cat([
            sequence_output.unsqueeze(2).expand(batch_size, sequence_length, sequence_length, hidden_size), 
            sequence_output.unsqueeze(1).expand(batch_size, sequence_length, sequence_length, hidden_size),
            self.dist_embeddings(dist_ids),
            self.type_embeddings(extend_attention_mask)
        ], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, extend_attention_mask.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.conv(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, extend_attention_mask.eq(0).unsqueeze(-1), 0.0)

        # (batch_size, num_spans, num_labels)
        logits = self.predictor(sequence_output, sequence_output, conv_outputs)
        logits = self.batched_index_select_2d(logits, spans)

        return logits