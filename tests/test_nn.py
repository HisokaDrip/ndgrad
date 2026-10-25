import numpy as np

from gradcheck import check
from ndgrad import Tensor
from ndgrad import functional as F
from ndgrad.nn import LayerNorm, Linear
from ndgrad.optim import Adam, clip_grad_norm

rng = np.random.default_rng(42)


def test_softmax_and_log_softmax_grads():
    check(lambda x: F.softmax(x, axis=-1), rng.normal(size=(3, 5)))
    check(lambda x: F.log_softmax(x, axis=0), rng.normal(size=(4, 2)))


def test_softmax_is_stable_for_huge_logits():
    out = F.softmax(Tensor([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(out.data))
    np.testing.assert_allclose(out.data.sum(), 1.0)


def test_cross_entropy_matches_manual():
    logits = rng.normal(size=(6, 4))
    targets = rng.integers(0, 4, size=6)
    loss = F.cross_entropy(Tensor(logits), targets).data
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    np.testing.assert_allclose(loss, -np.log(p[np.arange(6), targets]).mean())
    check(lambda x: F.cross_entropy(x, targets), logits)


def test_gelu_grad():
    check(F.gelu, rng.normal(size=(8,)))


def test_layernorm_grad_and_output_stats():
    ln = LayerNorm(6)
    x = rng.normal(3.0, 5.0, size=(4, 6))
    y = ln(Tensor(x)).data
    np.testing.assert_allclose(y.mean(-1), 0.0, atol=1e-9)
    np.testing.assert_allclose(y.std(-1), 1.0, atol=1e-3)
    check(lambda t: ln(t), x)


def test_adam_fits_linear_regression():
    X = rng.normal(size=(200, 3))
    true_w = np.array([[2.0], [-1.0], [0.5]])
    y = X @ true_w + 0.3

    model = Linear(3, 1, rng=np.random.default_rng(0))
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        pred = model(Tensor(X))
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    np.testing.assert_allclose(model.w.data, true_w, atol=1e-2)
    np.testing.assert_allclose(model.b.data, [0.3], atol=1e-2)


def test_clip_grad_norm():
    a = Tensor(np.zeros(2), requires_grad=True)
    a.grad = np.array([3.0, 4.0])
    norm = clip_grad_norm([a], 1.0)
    assert norm == 5.0
    np.testing.assert_allclose(np.linalg.norm(a.grad), 1.0)
