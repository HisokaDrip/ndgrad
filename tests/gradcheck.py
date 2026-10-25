import numpy as np

from ndgrad import Tensor


def numeric_grad(f, x, eps=1e-6):
    """Central differences on every element of x. Slow, but it's the ground truth."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        hi = f(x)
        x[i] = old - eps
        lo = f(x)
        x[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def check(fn, *arrays, atol=1e-5):
    """fn takes Tensors and returns a Tensor; we reduce with a fixed random
    projection so every output element contributes a different weight."""
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    rng = np.random.default_rng(1)
    probe = None

    def scalar(*xs):
        nonlocal probe
        out = fn(*xs)
        if probe is None:
            probe = rng.normal(size=out.shape)
        return (out * probe).sum()

    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    scalar(*tensors).backward()

    for k, a in enumerate(arrays):
        def f(x, k=k):
            xs = [Tensor(arr) for arr in arrays]
            xs[k] = Tensor(x)
            return scalar(*xs).data

        expected = numeric_grad(f, a.copy())
        np.testing.assert_allclose(tensors[k].grad, expected, atol=atol, rtol=1e-4)
