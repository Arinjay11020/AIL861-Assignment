import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer_block import Block, MyLayerNorm

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, positions):
        return self.pe[positions]


def top_k_logits(logits, k):
    if k is None or k == 0:
        return logits
    v, _ = torch.topk(logits, k)
    min_v = v[:, -1].unsqueeze(1)
    return torch.where(logits < min_v, torch.full_like(logits, -1e10), logits)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embed=300, n_layer=3, n_head=6, block_size=64, dropout=0.1, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_enc = SinusoidalPositionalEncoding(n_embed, max_len=block_size)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = MyLayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None, return_attentions: bool = False):
        B, T = idx.shape
        assert T <= self.block_size

        tok = self.tok_emb(idx)
        pos_idx = torch.arange(T, device=idx.device)
        pos = self.pos_enc(pos_idx)[None, :, :]
        x = self.drop(tok + pos)

        present_kvs = []
        attentions = [] if return_attentions else None

        for block in self.blocks:
            if return_attentions:
                x, present_kv, att = block(x, past_kv=None, return_attention=True)
                attentions.append(att)
            else:
                x, present_kv, _ = block(x, past_kv=None, return_attention=False)
            present_kvs.append(present_kv)

        x = self.ln_f(x)
        logits = self.head(x)

        if return_attentions:
            return logits, attentions

        if targets is not None:
            B_, T_, V = logits.shape
            loss = F.cross_entropy(logits.view(B_ * T_, V), targets.view(B_ * T_), ignore_index=getattr(self, "pad_id", -100))
            return logits, loss

        return logits

    def generate(self, idx, max_new_tokens=50, temperature=1.0, do_sample=True, top_k=None):
        device = next(self.parameters()).device
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T_init = idx.shape
        assert B == 1

        past_kvs = [None] * len(self.blocks)
        pos_table_size = self.pos_enc.pe.size(0)
        generated = idx.clone().to(device)

        for t_i in range(generated.shape[1]):
            token_t = generated[:, t_i].unsqueeze(1).to(device)
            tok = self.tok_emb(token_t)
            pos = self.pos_enc(torch.tensor([t_i % pos_table_size], device=device))[None, :, :]
            x = self.drop(tok + pos)

            new_past = []
            for layer_idx, block in enumerate(self.blocks):
                out, present_kv, _ = block(x, past_kv=past_kvs[layer_idx])
                new_past.append(present_kv)
                x = out
            past_kvs = new_past

        for step in range(max_new_tokens):
            next_pos = generated.shape[1]
            last_token = generated[:, -1].unsqueeze(1).to(device)
            tok = self.tok_emb(last_token)
            pos = self.pos_enc(torch.tensor([next_pos % pos_table_size], device=device))[None, :, :]
            x = self.drop(tok + pos)

            new_past = []
            for layer_idx, block in enumerate(self.blocks):
                out, present_kv, _ = block(x, past_kv=past_kvs[layer_idx])
                new_past.append(present_kv)
                x = out
            past_kvs = new_past

            x = self.ln_f(x)
            logits = self.head(x)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                logits = top_k_logits(logits, top_k)

            probs = torch.softmax(logits, dim=-1)
            if do_sample:
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_id], dim=1)

        return generated

    def generate_with_attn(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, do_sample=True, eos_id=None):
        device = next(self.parameters()).device
        self.eval()
        idx = idx.to(device)
        generated = idx.clone()
        per_step_atts = []

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.forward(generated, return_attentions=True)
            logits, atts = out

            last_logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                last_logits = top_k_logits(last_logits, top_k)
            probs = F.softmax(last_logits / max(temperature, 1e-8), dim=-1)

            next_id = torch.multinomial(probs, num_samples=1) if do_sample else torch.argmax(probs, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)

            step_att_list = []
            for a in atts:
                step_att_list.append(None if a is None else a.detach().cpu())
            per_step_atts.append(step_att_list)

            if eos_id is not None and int(next_id.item()) == int(eos_id):
                break

        return generated.cpu().tolist()[0], per_step_atts
