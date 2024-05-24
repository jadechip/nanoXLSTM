import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR

DEBUG_MODE=False

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        output = F.layer_norm(input, input.shape[-1:], self.weight, self.bias, 1e-5)
        if DEBUG_MODE: print(f"LayerNorm - input shape: {input.shape}, output shape: {output.shape}")
        return output

class sLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.U = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.f_bias = nn.Parameter(torch.zeros(config.n_embd).fill_(3.0))
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.n_embd, config.bias)

    def forward(self, x, hidden_states):
        x = self.ln(x)
        if DEBUG_MODE: print(f"sLSTM - x after LayerNorm: {x.shape}")

        batch_size, seq_len, _ = x.size()
        Z = self.W(x) + self.U(hidden_states)
        i, f, c, o = Z.chunk(4, dim=-1)
        if DEBUG_MODE: print(f"sLSTM - i: {i.shape}, f: {f.shape}, c: {c.shape}, o: {o.shape}")

        stab_factor = torch.max(i.max(dim=-1, keepdim=True)[0], f.max(dim=-1, keepdim=True)[0])
        i = torch.exp(i - stab_factor)
        f = torch.exp(f + self.f_bias - stab_factor)
        if DEBUG_MODE: print(f"sLSTM - i after exp: {i.shape}, f after exp: {f.shape}")

        cell_state = f * hidden_states + i * torch.tanh(c)
        normalizer_state = f + i
        cell_state = cell_state / normalizer_state

        o = torch.sigmoid(o)
        hidden_state = o * torch.tanh(cell_state)

        hidden_state = self.o_proj(hidden_state)
        hidden_state = self.dropout(hidden_state)
        if DEBUG_MODE: print(f"sLSTM - hidden_state: {hidden_state.shape}")

        return x, hidden_state

class mLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_q = nn.Linear(config.n_embd, config.n_embd)
        self.W_k = nn.Linear(config.n_embd, config.n_embd)
        self.W_v = nn.Linear(config.n_embd, config.n_embd)
        self.W_f = nn.Linear(config.n_embd, config.n_embd)
        self.W_i = nn.Linear(config.n_embd, config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.f_bias = nn.Parameter(torch.zeros(config.n_embd).fill_(3.0))
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.n_embd, config.bias)

    def forward(self, x):
        x = self.ln(x)
        if DEBUG_MODE: print(f"mLSTM - x after LayerNorm: {x.shape}")

        batch_size, seq_len, _ = x.size()
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        f = torch.sigmoid(self.W_f(x) + self.f_bias)
        i = torch.exp(self.W_i(x))
        if DEBUG_MODE: print(f"mLSTM - q: {q.shape}, k: {k.shape}, v: {v.shape}, f: {f.shape}, i: {i.shape}")

        memory = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1)), dim=-1)
        memory = torch.matmul(memory, v)
        if DEBUG_MODE: print(f"mLSTM - memory after softmax and matmul: {memory.shape}")

        memory = f * memory + i * v
        normalizer = f + i
        memory = memory / normalizer

        out = self.o_proj(memory)
        out = self.dropout(out)
        if DEBUG_MODE: print(f"mLSTM - out: {out.shape}")

        return out

class xLSTMBlock(nn.Module):
    def __init__(self, config, ratio_mLSTM=0.5):
        super().__init__()
        self.num_sLSTM = int(config.n_layer * (1 - ratio_mLSTM))
        self.num_mLSTM = config.n_layer - self.num_sLSTM
        self.sLSTM_blocks = nn.ModuleList([sLSTM(config) for _ in range(self.num_sLSTM)])
        self.mLSTM_blocks = nn.ModuleList([mLSTM(config) for _ in range(self.num_mLSTM)])

    def forward(self, x, hidden_states):
        residual = x
        for i in range(self.num_sLSTM):
            x, hidden_states = self.sLSTM_blocks[i](x, hidden_states)
            if DEBUG_MODE: print(f"xLSTMBlock - After sLSTM {i}: x: {x.shape}, hidden_states: {hidden_states.shape}")
        for i in range(self.num_mLSTM):
            x = self.mLSTM_blocks[i](x)
            if DEBUG_MODE: print(f"xLSTMBlock - After mLSTM {i}: x: {x.shape}")
        x = x + residual
        return x, hidden_states

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        if DEBUG_MODE: print(f"MLP - output: {x.shape}")
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.xLSTM_blocks = nn.ModuleList([xLSTMBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        token_embeddings = self.embedding(idx)
        if DEBUG_MODE: print(f"GPT - token_embeddings: {token_embeddings.shape}")

        # apply dropout
        x = self.drop(token_embeddings)
        if DEBUG_MODE: print(f"GPT - x after dropout: {x.shape}")

        hidden_states = [torch.zeros_like(x)] * self.config.n_layer
        for i, (block, hidden_state) in enumerate(zip(self.xLSTM_blocks, hidden_states)):
            x, hidden_state = block(x, hidden_state)
            x = x + token_embeddings  # Add residual connection
            if DEBUG_MODE: print(f"GPT - After xLSTMBlock {i}: x: {x.shape}, hidden_state: {hidden_state.shape}")

        x = self.ln_f(x)
        if DEBUG_MODE: print(f"GPT - x after final LayerNorm: {x.shape}")

        logits = self.lm_head(x)
        if DEBUG_MODE: print(f"GPT - logits: {logits.shape}")

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if DEBUG_MODE: print(f"GPT - loss: {loss.item()}")
        else:
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if DEBUG_MODE: print(f"GPT - Number of parameters: {n_params}")
        return n_params

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if DEBUG_MODE: print(f"GPT - block_size cropped to: {block_size}")

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        if DEBUG_MODE: print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        if DEBUG_MODE: print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            if DEBUG_MODE: print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys)
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, max_iters):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=max_iters, pct_start=0.05)
        if DEBUG_MODE: print(f"Configured optimizers with weight_decay: {weight_decay}, learning_rate: {learning_rate}, betas: {betas}, device_type: {device_type}, max_iters: {max_iters}")
        return optimizer, scheduler

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        if DEBUG_MODE: print(f"MFU estimated: {mfu:.6f}")
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for i in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if DEBUG_MODE: print(f"Generated token {i+1}/{max_new_tokens}: {idx_next.item()}")

        return idx
