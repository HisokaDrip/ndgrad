import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        # bias correction folded into the step size, saves two array ops per param
        lr_t = self.lr * np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            if self.wd:
                # decoupled (AdamW style), not L2 added to the grad
                p.data -= self.lr * self.wd * p.data
            p.data -= lr_t * m / (np.sqrt(v) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def clip_grad_norm(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total = np.sqrt(sum(float((g * g).sum()) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total
