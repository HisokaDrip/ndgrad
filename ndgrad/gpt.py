import numpy as np

from . import functional as F
from .nn import MLP, Embedding, LayerNorm, Linear, Module
from .tensor import no_grad


class CausalSelfAttention(Module):
    def __init__(self, dim, n_heads, rng):
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.q = Linear(dim, dim, rng=rng)
        self.k = Linear(dim, dim, rng=rng)
        self.v = Linear(dim, dim, rng=rng)
        self.proj = Linear(dim, dim, rng=rng)

    def _split(self, x, B, T):
        # (B, T, C) -> (B, heads, T, head_dim)
        return x.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q = self._split(self.q(x), B, T)
        k = self._split(self.k(x), B, T)
        v = self._split(self.v(x), B, T)

        scores = (q @ k.transpose(0, 1, 3, 2)) * (C // self.n_heads) ** -0.5
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        attn = F.softmax(scores.masked_fill(future, -1e9), axis=-1)

        y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.proj(y)
        return (y, attn) if return_attn else y


class Block(Module):
    def __init__(self, dim, n_heads, rng):
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, rng)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim, 4 * dim, rng=rng)

    def forward(self, x):
        # pre-norm residual, trains a lot more forgivingly than post-norm
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(Module):
    def __init__(self, vocab_size, block_size, dim=64, n_heads=4, n_layers=2, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok = Embedding(vocab_size, dim, rng=rng)
        self.pos = Embedding(block_size, dim, rng=rng)
        self.blocks = [Block(dim, n_heads, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab_size, rng=rng)

    def forward(self, idx):
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.block_size, f"sequence of {T} > block size {self.block_size}"
        x = self.tok(idx) + self.pos(np.arange(T))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def loss(self, idx, targets):
        logits = self(idx)
        B, T, V = logits.shape
        return F.cross_entropy(logits.reshape(B * T, V), np.asarray(targets).reshape(-1))

    def generate(self, idx, n_new, temperature=1.0, top_k=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        idx = np.array(idx, dtype=np.int64)
        with no_grad():
            for _ in range(n_new):
                logits = self(idx[:, -self.block_size:]).data[:, -1, :] / temperature
                if top_k is not None:
                    cutoff = np.sort(logits, axis=-1)[:, -top_k][:, None]
                    logits = np.where(logits < cutoff, -np.inf, logits)
                p = np.exp(logits - logits.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                nxt = np.array([rng.choice(self.vocab_size, p=row) for row in p])
                idx = np.concatenate([idx, nxt[:, None]], axis=1)
        return idx

    def state_dict(self):
        return {f"p{i}": p.data for i, p in enumerate(self.parameters())}

    def load_state_dict(self, state):
        for i, p in enumerate(self.parameters()):
            p.data[...] = state[f"p{i}"]
