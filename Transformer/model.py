import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
from sublayers import *



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.config = config

        self.multi_head_attn = MultiHeadAttention(config)
        self.ffnn = PoswiseFeedForwardNet(config)
        self.ln1 = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        context, attn_prob = self.multi_head_attn(inputs, inputs, inputs, attn_mask)
        residual = context + inputs
        normalized1 = self.ln1(residual)

        output1 = self.ffnn(normalized1)
        residual = output1 + normalized1
        output = self.ln2(residual)

        return output, attn_prob

class DecoderLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.self_attn = MultiHeadAttention(self.config)
    self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)
    self.dec_enc_attn = MultiHeadAttention(self.config)
    self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)
    self.pos_ffn = PoswiseFeedForwardNet(self.config)
    self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)

  def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
    self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
    self_att_outputs = self.layer_norm1(self_att_outputs + dec_inputs)

    dec_enc_outputs, dec_enc_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
    dec_enc_outputs = self.layer_norm2(dec_enc_outputs + self_att_outputs)

    ffnn_outputs = self.pos_ffn(dec_enc_outputs)
    ffnn_outputs = self.layer_norm3(ffnn_outputs + dec_enc_outputs)

    return ffnn_outputs, self_attn_prob, dec_enc_prob


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.enc_emb = nn.Embedding(config.n_enc_vocab, config.d_hidn)
        sinusoid_table = get_sinusoid_table(config.n_enc_seq + 1, config.d_hidn)
        position_encoding = torch.FloatTensor(sinusoid_table)
        self.pos_emb = nn.Embedding.from_pretrained(position_encoding, freeze=True)
        # 위에서 +1 하는 이유는 position이 [1,2,3,4,...256] 까지 1부터 시작하므로

        self.layers = nn.ModuleList([EncoderLayer(config)] * config.n_layer)

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device,
                                 dtype=inputs.dtype).expand(inputs.size(0),
                                                            inputs.size(1)).contiguous() + 1
        mask_position = inputs.eq(self.config.i_pad)
        positions.masked_fill_(mask_position, 0)

        outputs = self.enc_emb(inputs) + self.pos_emb(positions)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []

        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        return outputs, attn_probs

class Decoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
    sinusoid_table = torch.FloatTensor(get_sinusoid_table(self.config.n_dec_seq + 1, self.config.d_hidn))
    self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

    self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

  def forward(self, dec_inputs, enc_inputs, enc_outputs):
    positions = torch.arange(dec_inputs.size(1), device = dec_inputs.device, dtype = dec_inputs.dtype).\
    expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
    mask_position = dec_inputs.eq(self.config.i_pad)
    positions.masked_fill_(mask_position, 0)

    dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(dec_inputs)

    dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)

    dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)

    dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask) ,0)

    dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

    self_attn_probs, dec_enc_attn_probs = [], []

    for layer in self.layers :
      dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, \
                                                             dec_enc_attn_mask)
      self_attn_probs.append(self_attn_prob)
      dec_enc_attn_probs.append(dec_enc_attn_prob)

    return dec_outputs, self_attn_probs, dec_enc_attn_probs


class Transformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.encoder = Encoder(self.config)
    self.decoder = Decoder(self.config)

  def forward(self, enc_inputs, dec_inputs):
    enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)

    dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)

    return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

