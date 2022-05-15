import torch
from torch import nn
import numpy as np

class WKPooling(nn.Module):
    def __init__(self, layer_start: int = 4, context_window_size: int = 2):
        super(WKPooling, self).__init__()
        self.layer_start = layer_start
        self.context_window_size = context_window_size

    def forward(self, all_hidden_states, attention_mask):
        # ft_all_layers = all_hidden_states
        # org_device = ft_all_layers.device
        # all_layer_embedding = ft_all_layers.transpose(1, 0)
        # all_layer_embedding = all_layer_embedding[:, self.layer_start:, :, :]  # Start from 4th layers output

        org_device = attention_mask.device
        all_layer_embedding = torch.stack(all_hidden_states, dim=1)
        all_layer_embedding = all_layer_embedding[:, self.layer_start:, :, :]  # Start from 4th layers output

        # torch.qr is slow on GPU (see https://github.com/pytorch/pytorch/issues/22573). So compute it on CPU until issue is fixed
        all_layer_embedding = all_layer_embedding.cpu()

        unmask_num = np.array([sum(mask) for mask in attention_mask.cpu().numpy()]) - 1  # Not considering the last item
        embedding = []

        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, :unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            ##features.update({'sentence_embedding': features['cls_token_embeddings']})

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return output_vector

    def unify_token(self, token_feature):
        ## Unify Token Representation
        window_size = self.context_window_size

        alpha_alignment = torch.zeros(token_feature.size()[0], device=token_feature.device)
        alpha_novelty = torch.zeros(token_feature.size()[0], device=token_feature.device)

        for k in range(token_feature.size()[0]):
            left_window = token_feature[k - window_size:k, :]
            right_window = token_feature[k + 1:k + window_size + 1, :]
            window_matrix = torch.cat([left_window, right_window, token_feature[k, :][None, :]])
            Q, R = torch.qr(window_matrix.T)

            r = R[:, -1]
            alpha_alignment[k] = torch.mean(self.norm_vector(R[:-1, :-1], dim=0), dim=1).matmul(R[:-1, -1]) / torch.norm(r[:-1])
            alpha_alignment[k] = 1 / (alpha_alignment[k] * window_matrix.size()[0] * 2)
            alpha_novelty[k] = torch.abs(r[-1]) / torch.norm(r)

        # Sum Norm
        alpha_alignment = alpha_alignment / torch.sum(alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / torch.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        alpha = alpha / torch.sum(alpha)  # Normalize

        out_embedding = torch.mv(token_feature.t(), alpha)
        return out_embedding

    def norm_vector(self, vec, p=2, dim=0):
        ## Implements the normalize() function from sklearn
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def unify_sentence(self, sentence_feature, one_sentence_embedding):
        ## Unify Sentence By Token Importance
        sent_len = one_sentence_embedding.size()[0]

        var_token = torch.zeros(sent_len, device=one_sentence_embedding.device)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = self.cosine_similarity_torch(token_feature)
            var_token[token_index] = torch.var(sim_map.diagonal(-1))

        var_token = var_token / torch.sum(var_token)
        sentence_embedding = torch.mv(one_sentence_embedding.t(), var_token)

        return sentence_embedding
    
    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

wk = WKPooling()
all_hidden_states = [torch.rand(8, 512, 768) for i in range(12)]
attention_mask = torch.ones(8, 512, dtype=torch.long)
wk(all_hidden_states, attention_mask)

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

if opts.decode_labeled_as_ref:
    train_dataset = load_dataset(data_class, process_class, opts.train_input_file, opts.data_dir, "train",
                                    tokenizer, opts.train_max_seq_length, opts.context_size, opts.max_span_length, 
                                    negative_sampling=0.0,
                                )
    entities = set((chain(*[
        ["".join(entity[3]) for entity in example["entities"]] 
        for example in train_dataset.examples
    ])))
    [jieba.add_word(entity) for entity in entities]
    test_dataloader = trainer.build_test_dataloader(test_dataset)
    for step, (batch, result) in enumerate(zip(test_dataloader, results)):
        refs = []
        for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
            sequence_length = attention_mask.sum()
            input_ids = input_ids[1: sequence_length - 1]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            string = tokenizer.convert_tokens_to_string(tokens)
            words  = jieba.cut(string)
            start = 0
            refs.append([])
            for word in words:
                end = start + len(word)
                if word in entities:
                    refs[-1].append((start, end, word))
                start = end
            print()

        logits = result["logits"]
        spans = result["spans"]
        spans_mask = result["spans_mask"]
        logits = result["logits"]
        decoded = model.decode(logits, spans, spans_mask, 0.0, opts.label2id, opts.id2label, refs=refs)
