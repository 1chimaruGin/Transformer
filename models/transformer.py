import math
from turtle import forward
import torch
import torch.nn as nn

from .pos_encoding import positional_encoding
from .multihead_attn import MultiHeadAttention
from .utils import clone_module_list


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        features: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias_1: bool = True,
        bias_2: bool = True,
        bias_gate: bool = True,
    ):
        super(FeedForward, self).__init__()
        self.layer_1 = nn.Linear(d_model, features, bias=bias_1)
        self.layer_2 = nn.Linear(features, d_model, bias=bias_2)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, features, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer_1(x))
        if self.is_gated:
            g = g * self.linear_v(x)
        else:
            x = g

        x = self.dropout(x)
        return self.layer_2(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer("pos_encodings", positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.pos_encodings[: x.shape, [0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super(LearnedPositionalEmbedding, self).__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.pos_encodings = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        pe = self.pos_encodings[: x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention = None,
        feed_forward: FeedForward = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_FFN = nn.LayerNorm([d_model])

        self.is_save_FFN_input = False

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        src: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ):
        z = self.self_attn(x)
        self_attn = self.self_attn(q=z, k=z, v=z, mask=mask)

        x = x + self.dropout(self_attn)

        if src is not None:
            z = self.norm_src_attn(x)
            attn_src = self.src_attn(q=z, k=src, v=src, mask=src_mask)
            x = x + self.dropout(attn_src)

        z = self.norm_FFN(x)
        if self.is_save_FFN_input:
            self.FFN_Input = z.clone()

        FFN = self.feed_forward(z)
        x = x + self.dropout(FFN)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super(Encoder, self).__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super(Decoder, self).__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask=tgt_mask, src=memory, src_mask=src_mask)

        return self.norm(x)

