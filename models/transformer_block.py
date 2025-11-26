from .attention import *
import torch.nn as nn
class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        # x: (..., normalized_shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Decoder-only Transformer block with pre-norm. Supports optional past_kv for caching.
    past_kv: tuple (past_k, past_v) for this layer, or None.
    Returns:
    out, present_kv
    """
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = MyLayerNorm(n_embed)
        self.ln2 = MyLayerNorm(n_embed)

    def forward(self, x, past_kv=None, return_attention=True):
        if return_attention:
            y, present_kv, att = self.sa(self.ln1(x), past_kv=past_kv, return_attn=True)
        else:
            y, present_kv = self.sa(self.ln1(x), past_kv=past_kv, return_attn=False)
            att = None
        x = x + y
        x = x + self.ffwd(self.ln2(x))
        return x, present_kv, att


