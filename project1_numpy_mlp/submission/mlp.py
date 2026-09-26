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
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        # TODO: return logits (N, 10)
        raise NotImplementedError

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save(self, path):
        # TODO: np.savez(path, **{...all W and b...})
        raise NotImplementedError

    def load(self, path):
        # TODO: restore weights saved by save()
        raise NotImplementedError
