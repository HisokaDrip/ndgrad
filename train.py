"""Train a character-level GPT on the Zen of Python (or any text file) using
nothing but ndgrad + numpy.

    python train.py
    python train.py --text some_book.txt --steps 3000 --dim 96 --layers 3
"""
import argparse
import codecs
import contextlib
import io
import time

import numpy as np

from ndgrad.gpt import GPT
from ndgrad.optim import Adam, clip_grad_norm


def zen_of_python():
    # `import this` prints the zen as a side effect; the module itself only
    # stores it rot13'd, which is a fun little easter egg
    with contextlib.redirect_stdout(io.StringIO()):
        import this
    return codecs.decode(this.s, "rot13")


def get_batch(data, block_size, batch_size, rng):
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s : s + block_size] for s in starts])
    y = np.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="train on this file instead of the zen of python")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=300, help="chars to generate at the end")
    ap.add_argument("--save", help="write weights to this .npz when done")
    args = ap.parse_args()

    text = open(args.text, encoding="utf-8").read() if args.text else zen_of_python()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text])
    print(f"{len(text)} chars, vocab {len(chars)}")

    model = GPT(len(chars), args.block, args.dim, args.heads, args.layers, seed=args.seed)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"{model.num_params():,} parameters\n")
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        # short linear warmup then cosine decay, tiny models still like it
        warm = min(1.0, step / 100)
        opt.lr = args.lr * warm * 0.5 * (1 + np.cos(np.pi * step / args.steps))

        x, y = get_batch(data, args.block, args.batch, rng)
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(opt.params, 1.0)
        opt.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step:5d}  loss {loss.data:.4f}  ({time.time() - t0:.1f}s)")

    if args.save:
        np.savez(args.save, **model.state_dict())
        print(f"saved weights to {args.save}")

    prompt = np.array([[stoi[text[0]]]])
    out = model.generate(prompt, args.sample, temperature=0.8, top_k=10, rng=rng)
    print("\n--- sample ---")
    print("".join(chars[i] for i in out[0]))


if __name__ == "__main__":
    main()
