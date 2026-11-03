# ndgrad

A small reverse-mode autograd engine on top of numpy, and a GPT built with it.

I wanted to actually understand what `loss.backward()` does inside PyTorch, so I wrote the whole stack myself: a `Tensor` that records the ops applied to it, backward rules for each op (broadcasting included, which was the annoying part), a few layers, Adam, and then a decoder-only transformer with causal multi-head attention. No torch, no jax, just numpy.

Every op is checked against finite differences in the tests, and the full model gets a spot gradcheck too, so I trust the gradients.

## what's in here

- `ndgrad/tensor.py` has the engine: add/mul/pow/matmul (batched, broadcasting), sum/mean, exp/log/tanh/relu, reshape/transpose, embedding lookup, masked_fill, and `no_grad()` for inference
- `ndgrad/functional.py` holds softmax, log_softmax, cross entropy and gelu
- `ndgrad/nn.py` has Linear, Embedding, LayerNorm and MLP
- `ndgrad/optim.py` is Adam with decoupled weight decay, plus grad norm clipping
- `ndgrad/gpt.py` is the transformer: pre-norm blocks, learned positional embeddings, top-k sampling
- `train.py` is a char-level training loop

## running it

```
pip install -r requirements.txt
python train.py
pytest
```

By default it trains on the Zen of Python, which ships inside CPython (`import this`, stored rot13'd for some reason). It's only 856 characters, so the model memorises it. That's the point of a sanity check: if a from-scratch transformer can't overfit a tiny corpus, something is broken. Point it at a bigger text with `--text file.txt` and bump `--dim` / `--layers` if you want it to actually generalise.

Output from a run on my machine (CPU, about 70s):

```
856 chars, vocab 45
107,949 parameters

step     1  loss 4.3595  (0.1s)
step   100  loss 1.5090  (6.1s)
step   200  loss 0.3832  (11.1s)
step   500  loss 0.1666  (24.9s)
step  1000  loss 0.1455  (47.4s)
step  1500  loss 0.1213  (69.6s)

--- sample ---
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy 
```

## using the engine on its own

```python
from ndgrad import Tensor

w = Tensor([[0.5, -1.0]], requires_grad=True)
x = Tensor([[2.0], [3.0]])
loss = (w @ x).tanh().sum()
loss.backward()
print(w.grad)
```

## how it works

- **recording the graph.** every op on a `Tensor` returns a new tensor that keeps
  references to its inputs (`_parents`) plus a small `_backward` closure. that closure
  knows the local derivative of that one op, e.g. for `a * b` it pushes `grad * b` into `a`
  and `grad * a` into `b`. nothing gets differentiated symbolically. the graph is just
  whatever ops actually ran.
- **backward.** `loss.backward()` does a topological sort of the graph (iteratively, so
  deep graphs don't hit the recursion limit), seeds the output grad with ones, and calls
  each node's `_backward` in reverse order. grads are summed, so a tensor
  used in several places gets the sum of all its paths.
- **broadcasting.** if numpy stretched an input in the forward pass (say a bias of shape
  `(C,)` added to `(B, T, C)`), the incoming grad has the big shape. `_unbroadcast` sums
  over the stretched axes so every grad ends up the same shape as its tensor.
- **layers on top.** softmax, log-softmax, cross entropy, gelu, LayerNorm etc. are all
  built from the basic ops, so they get their gradients for free. only the primitives in
  `tensor.py` have hand-written backward rules.
- **the GPT.** token + learned position embeddings go through pre-norm blocks
  (`x + attn(ln(x))`, then `x + mlp(ln(x))`). attention splits q/k/v into heads, scales
  `q @ k^T` by `1/sqrt(head_dim)`, masks out future positions with `masked_fill` before the
  softmax, and mixes `v`. a final LayerNorm and linear layer give next-char logits, trained with cross
  entropy and Adam. generation feeds the last `block_size` chars back in one token at a time.

## notes

- Everything is float64. Slower, but gradchecks with eps=1e-6 need the precision, and the models here are tiny anyway.
- The backward pass uses an iterative topo sort, since deep graphs can blow past Python's recursion limit with the recursive one.
- The embedding backward uses `np.add.at` rather than `g[idx] += ...`, because fancy-index `+=` silently drops repeated indices. That bug cost me a while.

## what I'd do next

- fuse softmax + cross entropy into one op with a hand-written backward (faster and less memory)
- KV cache for generation, right now every new token reruns the whole context
- a float32 mode and some profiling, most of the time goes to Python overhead on small matmuls
- train on something bigger, like tiny shakespeare, and compare loss curves against a PyTorch version of the same model

## license

MIT
