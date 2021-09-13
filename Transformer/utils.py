import numpy as np
import torch
import torch.nn as nn

def get_sinusoid_table(n_seq, d_hidn):
    def cal_angle(pos, i):
        return pos / np.power(10000, 2 * (i // 2) / d_hidn)

    def get_posi_angle_vec(pos):
        return [cal_angle(pos, i) for i in range(d_hidn)]

    table = np.array([get_posi_angle_vec(pos) for pos in range(n_seq)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return table


def get_attn_pad_mask(seq_q, seq_k, i_pad):
  batch_size, len_q = seq_q.size()
  batch_size, len_k = seq_k.size()
  pad_attn_mask = seq_k.data.eq(i_pad)
  pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
  return pad_attn_mask


def get_attn_decoder_mask(seq):
  subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
  subsequent_mask = subsequent_mask.triu(diagonal=1)
  return subsequent_mask


class SclaedDotProductAttention(nn.Module):
  def __init__(self, d_head):
    super().__init__()
    self.scale = 1 / (d_head ** 0.5)

  def forward(self, Q, K, V, attn_mask):
    scores = torch.matmul(Q, K.transpose(-1,-2))
    scores.masked_fill_(attn_mask, -1e9)
    attn_prob = nn.Softmax(dim=-1)(scores)
    context = torch.matmul(attn_prob, V)

    return context, attn_prob