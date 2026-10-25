import numpy as np

from . import functional as F
from .tensor import Tensor


class Module:
    def parameters(self):
        params = []
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_params(self):
        return sum(p.data.size for p in self.parameters())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _param(arr):
    return Tensor(arr, requires_grad=True)


class Linear(Module):
    def __init__(self, n_in, n_out, bias=True, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.w = _param(rng.normal(0.0, n_in**-0.5, size=(n_in, n_out)))
        self.b = _param(np.zeros(n_out)) if bias else None

    def forward(self, x):
        y = x @ self.w
        return y + self.b if self.b is not None else y


class Embedding(Module):
    def __init__(self, n, dim, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.w = _param(rng.normal(0.0, 0.02, size=(n, dim)))

    def forward(self, idx):
        return self.w.take(idx)


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.gamma = _param(np.ones(dim))
        self.beta = _param(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        return centered / (var + self.eps) ** 0.5 * self.gamma + self.beta


class MLP(Module):
    def __init__(self, dim, hidden, rng=None):
        self.fc = Linear(dim, hidden, rng=rng)
        self.out = Linear(hidden, dim, rng=rng)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))
