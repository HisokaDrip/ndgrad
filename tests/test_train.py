import numpy as np

from train import get_batch, zen_of_python


def test_zen_is_decoded():
    zen = zen_of_python()
    assert zen.startswith("The Zen of Python")
    assert "Beautiful is better than ugly." in zen


def test_batches_are_shifted_by_one():
    data = np.arange(100)
    x, y = get_batch(data, 10, 4, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 10)
    np.testing.assert_array_equal(y, x + 1)
