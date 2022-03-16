import math
import torch
import torch.nn as nn
from typing import Optional, List

class LinearTrasformation(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super(LinearTrasformation, self).__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads
        self.heads = heads

        self.query = LinearTrasformation(d_model, heads, self.d_k, bias)
        self.key = LinearTrasformation(d_model, heads, self.d_k, bias)
        self.value = LinearTrasformation(d_model, heads, self.d_k, bias)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)

        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_score(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd, jbhd->ijbh', query, key)
        
    def masks(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        seq_len, batch_size, _ = q.shape

        if mask is not None:
            mask = self.masks(mask, q.shape, k.shape)
        
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        scores  = self.get_score(query, key)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        x = torch.einsum('ijbh, jbhd->ibhd', attn, value)

        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)
        return self.output(x)
