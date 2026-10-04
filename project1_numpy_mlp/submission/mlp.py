"""P1 starter — assemble layers into an MLP."""
import numpy as np

from layers import Linear, ReLU, SoftmaxCrossEntropy


def build_model(seed: int = 0) -> "MLP":
    # Architecture is up to you, e.g. 784 -> 256 -> 10.
    return MLP([784, 256, 10], seed=seed)


class MLP:
    def __init__(self, dims, seed: int = 0):
        self.dims = dims
        # TODO: create Linear/ReLU layers from dims

        self.linear1 = Linear(dims[0], dims[1], seed=seed)
        self.relu = ReLU()
        self.linear2 = Linear(dims[1], dims[2], seed=seed + 1)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        # TODO: return logits (N, 10)
        hidden = self.linear1.forward(X)
        hidden = self.relu.forward(hidden)
        return self.linear2.forward(hidden)
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save(self, path):
        #懒得写，dirtywork就交给codex了
        np.savez(
            path,
            linear1_W=self.linear1.W,
            linear1_b=self.linear1.b,
            linear2_W=self.linear2.W,
            linear2_b=self.linear2.b,
        )

    def load(self, path):
        #懒得写，dirtywork就交给codex了
        with np.load(path) as data:
            self.linear1.W = data["linear1_W"].copy()
            self.linear1.b = data["linear1_b"].copy()
            self.linear2.W = data["linear2_W"].copy()
            self.linear2.b = data["linear2_b"].copy()
