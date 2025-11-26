import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention with optional KV caching.
    This version builds the causal mask dynamically from (T, Tk) to avoid
    shape mismatches when past_kv grows beyond registered block_size.
    """
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.block_size = block_size

        # projections
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def _shape(self, x):
        # (B, T, C) -> (B, n_head, T, head_dim)
        B, T, C = x.size()
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()

    def _unshape(self, x):
        # (B, n_head, T, head_dim) -> (B, T, n_head*head_dim)
        B, H, T, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, x, past_kv=None, return_attn=False):
        B, T, C = x.size()

        # project
        q = self._shape(self.q_proj(x))  # (B, n_head, T, head_dim)
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        if past_kv is not None:
            past_k, past_v = past_kv
            # concat past keys/values along time dim
            k = torch.cat([past_k, k], dim=2)   # (B, n_head, Tk, head_dim)
            v = torch.cat([past_v, v], dim=2)

        present_k = k
        present_v = v

        # compute raw attention scores
        # q: (B, n_head, T, D), k: (B, n_head, Tk, D)
        att = q @ k.transpose(-2, -1) * (1.0 / (self.head_dim ** 0.5))  # (B, n_head, T, Tk)

        # Build dynamic causal mask so (T, Tk) shapes always match
        # mask[i,j] = True iff j <= i (i : query pos, j : key pos)
        Tk = k.size(2)
        # create indices on the correct device
        device = att.device
        # small, direct construction:
        # i: (T,1), j: (1,Tk) -> compare j <= i -> (T, Tk)
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(Tk, device=device).unsqueeze(0)
        causal_mask = (j <= i).unsqueeze(0).unsqueeze(0)  # shape (1,1,T,Tk), broadcastable to att

        # apply mask: positions that are False become -inf
        att = att.masked_fill(~causal_mask, float("-inf"))

        # softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # attention output
        out = att @ v  # (B, n_head, T, head_dim)
        out = self._unshape(out)  # (B, T, n_embed)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if return_attn:
            return out, (present_k, present_v), att
        else:
            return out, (present_k, present_v)
