"""P1 starter — implement the three layers. Only numpy is allowed."""
import numpy as np


class Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He/Xavier-style init is a good choice; keep attribute names W / b.
        self.W = rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.dW = None
        self.db = None

    def forward(self, x):
        # TODO: cache what backward needs, return x @ W + b
        raise NotImplementedError

    def backward(self, grad_out):
        # TODO: set self.dW, self.db and return dL/dx
        raise NotImplementedError


class ReLU:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        # TODO: return mean cross-entropy loss (float). Tip: subtract row max
        # before exp for numerical stability.
        raise NotImplementedError

    def backward(self):
        # TODO: return dL/dlogits, shape (N, C), already divided by N.
        # (You proved in Lecture 2 that this is (p - y_onehot) / N.)
        raise NotImplementedError
