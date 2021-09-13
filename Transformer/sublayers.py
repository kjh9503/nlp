import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super(MultiHeadAttention, self).__init__()
    self.config = config

    self.W_Q = nn.Linear(config.d_hidn, config.d_head * config.n_head)
    self.W_K = nn.Linear(config.d_hidn, config.d_head * config.n_head)
    self.W_V = nn.Linear(config.d_hidn, config.d_head * config.n_head)
    self.scaled_dot_attn = SclaedDotProductAttention(config.d_head)
    self.linear = nn.Linear(config.n_head * config.d_head, config.d_hidn)

  def forward(self, Q, K, V, attn_mask):
    batch_size = Q.size(0)

    Q_vectors = self.W_Q(Q).view(
        batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

    K_vectors = self.W_K(K).view(
        batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

    V_vectors = self.W_V(V).view(
        batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

    attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

    context, attn_prob = self.scaled_dot_attn(Q_vectors, K_vectors, V_vectors, attn_mask)
    context = context.transpose(1,2).contiguous().view(
        batch_size, -1, self.config.n_head * self.config.d_head)

    output = self.linear(context)

    return output, attn_prob



class PoswiseFeedForwardNet(nn.Module):
  def __init__(self, config):
    super(PoswiseFeedForwardNet, self).__init__()
    self.config = config

    self.conv1 = nn.Conv1d(in_channels = config.d_hidn,
                           out_channels = config.d_ff, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels = config.d_ff,
                           out_channels = config.d_hidn, kernel_size=1)
    self.active = F.gelu
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, inputs):
    output = self.conv1(inputs.transpose(-1,-2))
    activated = self.active(output)
    output = self.conv2(activated).transpose(-1,-2)
    output = self.dropout(output)

    return output