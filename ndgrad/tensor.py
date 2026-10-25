import contextlib

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Skip graph building, e.g. while sampling from a trained model."""
    global _grad_enabled
    prev, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = prev


def _unbroadcast(grad, shape):
    # numpy happily stretches dims in the forward pass, so on the way back
    # we have to sum the stretched dims away to get a grad of the right shape
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    # --- plumbing -----------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        op = f", op={self._op!r}" if self._op else ""
        return f"Tensor(shape={self.shape}{op})"

    @staticmethod
    def _lift(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def _child(self, data, parents, op):
        track = _grad_enabled and any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=track, _parents=parents if track else (), _op=op)

    def _accum(self, g):
        if self.requires_grad:
            self.grad = g if self.grad is None else self.grad + g

    # --- elementwise --------------------------------------------------------

    def __add__(self, other):
        other = Tensor._lift(other)
        out = self._child(self.data + other.data, (self, other), "add")

        def _backward():
            self._accum(_unbroadcast(out.grad, self.shape))
            other._accum(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor._lift(other)
        out = self._child(self.data * other.data, (self, other), "mul")

        def _backward():
            self._accum(_unbroadcast(out.grad * other.data, self.shape))
            other._accum(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only scalar powers for now"
        out = self._child(self.data**p, (self,), "pow")

        def _backward():
            self._accum(out.grad * p * self.data ** (p - 1))

        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor._lift(other))

    def __rsub__(self, other):
        return Tensor._lift(other) + (-self)

    def __truediv__(self, other):
        return self * Tensor._lift(other) ** -1

    def __rtruediv__(self, other):
        return Tensor._lift(other) * self**-1

    def exp(self):
        e = np.exp(self.data)
        out = self._child(e, (self,), "exp")
        out._backward = lambda: self._accum(out.grad * e)
        return out

    def log(self):
        out = self._child(np.log(self.data), (self,), "log")
        out._backward = lambda: self._accum(out.grad / self.data)
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = self._child(t, (self,), "tanh")
        out._backward = lambda: self._accum(out.grad * (1 - t * t))
        return out

    def relu(self):
        out = self._child(np.maximum(self.data, 0), (self,), "relu")
        out._backward = lambda: self._accum(out.grad * (self.data > 0))
        return out

    # --- linear algebra / reductions ---------------------------------------

    def __matmul__(self, other):
        other = Tensor._lift(other)
        assert self.ndim >= 2 and other.ndim >= 2, "matmul wants >=2d operands"
        out = self._child(self.data @ other.data, (self, other), "matmul")

        def _backward():
            g = out.grad
            self._accum(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            other._accum(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = self._child(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    # --- shape stuff --------------------------------------------------------

    def reshape(self, *shape):
        out = self._child(self.data.reshape(*shape), (self,), "reshape")
        out._backward = lambda: self._accum(out.grad.reshape(self.shape))
        return out

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.ndim)))
        out = self._child(self.data.transpose(axes), (self,), "transpose")
        inverse = np.argsort(axes)
        out._backward = lambda: self._accum(out.grad.transpose(inverse))
        return out

    def take(self, idx):
        """Row lookup along axis 0, i.e. an embedding table."""
        idx = np.asarray(idx)
        out = self._child(self.data[idx], (self,), "take")

        def _backward():
            g = np.zeros_like(self.data)
            # add.at instead of g[idx] += ... so repeated tokens all count
            np.add.at(g, idx, out.grad)
            self._accum(g)

        out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        out = self._child(np.where(mask, value, self.data), (self,), "masked_fill")
        out._backward = lambda: self._accum(np.where(mask, 0.0, out.grad))
        return out

    # --- the actual point of all this --------------------------------------

    def backward(self, grad=None):
        # iterative topo sort, a 6-layer transformer graph blows past
        # python's recursion limit with the naive recursive version
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if id(p) not in seen)

        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        for node in reversed(order):
            node._backward()
