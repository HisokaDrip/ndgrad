import numpy as np

from .tensor import Tensor


def softmax(x, axis=-1):
    # subtracting the max is a constant shift, so the gradient doesn't care,
    # but exp() very much does
    shifted = x - x.data.max(axis=axis, keepdims=True)
    e = shifted.exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    shifted = x - x.data.max(axis=axis, keepdims=True)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits, targets):
    """logits: (N, C) Tensor, targets: (N,) ints. Mean NLL."""
    targets = np.asarray(targets)
    n = logits.shape[0]
    onehot = np.zeros(logits.shape)
    onehot[np.arange(n), targets] = 1.0
    return -(log_softmax(logits, axis=-1) * onehot).sum() / float(n)


def gelu(x):
    # tanh approximation, same one GPT-2 used
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + (c * (x + 0.044715 * x**3)).tanh())
