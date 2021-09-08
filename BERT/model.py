import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)
        return context, attn_prob



class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.W_Q = nn.Linear(config.d_hidn, config.n_head * config.d_head)
    self.W_K = nn.Linear(config.d_hidn, config.n_head * config.d_head)
    self.W_V = nn.Linear(config.d_hidn, config.n_head * config.d_head)

    self.scaled_dot_attn = ScaledDotProductAttention(config)
    self.linear = nn.Linear(config.n_head * config.d_head, config.d_hidn)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, Q, K, V, attn_mask):

    batch_size = Q.size(0)
    q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
    k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
    v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

    attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

    context, attn_prob = self.scaled_dot_attn.forward(q_s, k_s, v_s, attn_mask)
    context = context.transpose(1,2).contiguous().\
    view(batch_size, -1, self.config.n_head * self.config.d_head)
    output = self.linear(context)
    output = self.dropout(output)
    return output, attn_prob



class PoswiseFeedForwardNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.linear1 = nn.Linear(config.d_hidn, config.d_ff)
    self.linear2 = nn.Linear(config.d_ff, config.d_hidn)
    self.active = F.relu
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, inputs):
    #inputs : (batch_size, n_q_seq, d_hidn)
    output = self.linear1(inputs)
    output = self.active(output)

    output = self.linear2(output)
    output = self.dropout(output)

    return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(config)
        self.layer_norm2 = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(att_outputs + inputs)

        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)

        return ffn_outputs, attn_prob


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask= pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    # return pad_attn_mask
    return pad_attn_mask


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(config.n_enc_vocab, config.d_hidn)
        self.pos_emb = nn.Embedding(config.n_enc_seq + 1, config.d_hidn)  ### 왜 n_enc_seq + 1인지 ??????
        self.seg_emb = nn.Embedding(config.n_seg_type, config.d_hidn)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs, segments):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype). \
                        expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        outputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(segments)
        # (bs, n_enc_seq, d_hidn)

        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        return outputs, attn_probs


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.d_hidn, config.d_hidn)
        self.activation = torch.tanh

    def forward(self, inputs, segments):
        outputs, self_attn_probs = self.encoder(inputs, segments)
        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)

        return outputs, outputs_cls, self_attn_probs
        # outputs : (bs, n_enc_seq, n_enc_vocab)

    def save(self, epoch, loss, path):
        torch.save({"epoch": epoch, "loss": loss, \
                    "state_dict": self.state_dict()}, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]


class BERTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = BERT(config)
        self.projection_cls = nn.Linear(config.d_hidn, 2, bias=False)
        self.projection_lm = nn.Linear(config.d_hidn, config.n_enc_vocab, bias=False)
        self.projection_lm.weight = self.bert.encoder.enc_emb.weight

    def forward(self, inputs, segments):
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)

        logits_cls = self.projection_cls(outputs_cls)
        logits_lm = self.projection_lm(outputs)

        return logits_cls, logits_lm, attn_probs
