import numpy as np

from gradcheck import numeric_grad
from ndgrad.gpt import GPT, CausalSelfAttention
from ndgrad.optim import Adam
from ndgrad import Tensor


def tiny(seed=0):
    return GPT(vocab_size=11, block_size=8, dim=16, n_heads=2, n_layers=2, seed=seed)


def test_logits_shape():
    model = tiny()
    out = model(np.zeros((3, 8), dtype=int))
    assert out.shape == (3, 8, 11)


def test_future_tokens_cannot_leak_into_the_past():
    model = tiny()
    rng = np.random.default_rng(0)
    a = rng.integers(0, 11, size=(1, 8))
    b = a.copy()
    b[0, 5:] = (b[0, 5:] + 3) % 11  # scramble only the future
    la, lb = model(a).data, model(b).data
    np.testing.assert_allclose(la[0, :5], lb[0, :5], atol=1e-12)
    assert not np.allclose(la[0, 5:], lb[0, 5:])


def test_attention_rows_are_causal_distributions():
    attn = CausalSelfAttention(8, 2, np.random.default_rng(1))
    _, weights = attn(Tensor(np.random.default_rng(2).normal(size=(2, 5, 8))), return_attn=True)
    w = weights.data
    np.testing.assert_allclose(w.sum(-1), 1.0)
    assert np.all(w[..., np.triu_indices(5, k=1)[0], np.triu_indices(5, k=1)[1]] < 1e-12)


def test_whole_model_gradcheck_on_a_few_params():
    model = GPT(vocab_size=5, block_size=4, dim=8, n_heads=2, n_layers=1, seed=3)
    idx = np.array([[1, 2, 3, 4]])
    tgt = np.array([[2, 3, 4, 0]])
    model.loss(idx, tgt).backward()

    # full numeric check on every weight would take a while, so spot check
    # one tensor from each part of the network
    for p in [model.tok.w, model.blocks[0].attn.k.w, model.blocks[0].ln2.gamma, model.head.b]:
        analytic = p.grad.copy()

        def f(x, p=p):
            saved = p.data.copy()
            p.data[...] = x
            val = model.loss(idx, tgt).data
            p.data[...] = saved
            return val

        np.testing.assert_allclose(analytic, numeric_grad(f, p.data.copy()), atol=1e-6)


def test_memorises_a_short_sequence():
    model = tiny(seed=1)
    seq = np.array([[1, 4, 1, 5, 9, 2, 6, 5, 3]])
    x, y = seq[:, :-1], seq[:, 1:]
    opt = Adam(model.parameters(), lr=1e-2)
    first = model.loss(x, y).data
    for _ in range(150):
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.data < 0.05 < first

    out = model.generate(seq[:, :1], 8, temperature=0.01, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, seq)


def test_state_dict_roundtrip():
    a, b = tiny(seed=0), tiny(seed=99)
    b.load_state_dict(a.state_dict())
    idx = np.arange(8)[None] % 11
    np.testing.assert_allclose(a(idx).data, b(idx).data)
