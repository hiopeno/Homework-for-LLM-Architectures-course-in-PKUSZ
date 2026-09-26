"""Public smoke tests for P1. These are intentionally not the hidden grader."""

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_module(root, name):
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission")
    args = parser.parse_args()
    root = Path(args.submission).resolve()
    sys.path.insert(0, str(root))

    layers = load_module(root, "layers")
    mlp = load_module(root, "mlp")

    linear = layers.Linear(3, 2, seed=0)
    linear.W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    linear.b = np.array([1.0, -1.0])
    x = np.array([[1.0, 0.0, 2.0]])
    assert np.allclose(linear.forward(x), [[12.0, 13.0]])
    dx = linear.backward(np.ones((1, 2)))
    assert dx.shape == x.shape
    assert linear.dW.shape == linear.W.shape
    assert linear.db.shape == linear.b.shape

    relu = layers.ReLU()
    assert np.array_equal(
        relu.forward(np.array([[-1.0, 0.0, 2.0]])),
        np.array([[0.0, 0.0, 2.0]]),
    )

    loss_fn = layers.SoftmaxCrossEntropy()
    loss = loss_fn.forward(np.zeros((2, 10)), np.array([0, 9]))
    assert np.isclose(loss, np.log(10.0))
    assert loss_fn.backward().shape == (2, 10)

    model = mlp.build_model(seed=0)
    sample = np.zeros((2, 784), dtype=np.float32)
    assert np.asarray(model.predict(sample)).shape == (2,)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.npz"
        before = np.asarray(model.predict(sample))
        model.save(path)
        restored = mlp.build_model(seed=123)
        restored.load(path)
        after = np.asarray(restored.predict(sample))
        assert np.array_equal(before, after)

    print("Public tests passed.")


if __name__ == "__main__":
    main()
