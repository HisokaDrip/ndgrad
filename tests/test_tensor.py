import numpy as np
import pytest

from gradcheck import check
from ndgrad import Tensor, no_grad

rng = np.random.default_rng(0)


def r(*shape):
    return rng.normal(size=shape)


def test_add_mul_with_broadcasting():
    check(lambda a, b: a * b + a, r(3, 4), r(4))
    check(lambda a, b: a + b, r(2, 1, 5), r(3, 1))


def test_sub_div_pow():
    check(lambda a, b: (a - b) / (b * b + 1.0), r(3, 3), r(3, 3))
    check(lambda a: a**3 - 2.0 / (a**2 + 1.0), r(5))
    check(lambda a: 1.0 - a, r(4))


def test_unary():
    check(lambda a: a.tanh(), r(6))
    check(lambda a: a.exp(), r(2, 3))
    check(lambda a: (a * a + 0.5).log(), r(2, 3))
    # keep away from 0 so finite differences don't straddle the kink
    x = r(10)
    x[np.abs(x) < 0.1] = 0.5
    check(lambda a: a.relu(), x)


def test_matmul_batched_and_broadcast():
    check(lambda a, b: a @ b, r(3, 4), r(4, 2))
    check(lambda a, b: a @ b, r(2, 3, 4), r(4, 5))
    check(lambda a, b: a @ b, r(2, 2, 3, 4), r(2, 2, 4, 3))


def test_reductions():
    check(lambda a: a.sum(), r(3, 4))
    check(lambda a: a.sum(axis=1), r(3, 4, 2))
    check(lambda a: a.mean(axis=-1, keepdims=True), r(3, 4))
    check(lambda a: a.mean(axis=(0, 2)), r(3, 4, 2))


def test_shapes():
    check(lambda a: a.reshape(4, 6), r(2, 3, 4))
    check(lambda a: a.transpose(0, 2, 1, 3), r(2, 3, 4, 5))
    check(lambda a: a.transpose(), r(3, 5))


def test_take_handles_repeated_indices():
    idx = np.array([[0, 2, 2], [1, 0, 2]])
    check(lambda w: w.take(idx), r(4, 3))


def test_masked_fill():
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    check(lambda a: a.masked_fill(mask, -5.0) * 2.0, r(2, 4, 4))


def test_reused_node_accumulates():
    x = Tensor([2.0, -1.0], requires_grad=True)
    y = x * x + x * 3.0  # x shows up in three places
    y.sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 3)


def test_no_grad_builds_nothing():
    w = Tensor(r(3, 3), requires_grad=True)
    with no_grad():
        out = (w @ w).tanh().sum()
    assert not out.requires_grad
    assert out._parents == ()


def test_non_scalar_backward_needs_explicit_grad_shape():
    x = Tensor(r(3), requires_grad=True)
    (x * 2.0).backward(np.ones(3))
    np.testing.assert_allclose(x.grad, 2.0)


def test_pow_rejects_tensor_exponent():
    with pytest.raises(AssertionError):
        Tensor(1.0) ** Tensor(2.0)
