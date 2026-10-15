"""Public structural checks for Project 2."""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


def load_model_module(root):
    spec = importlib.util.spec_from_file_location("model", root / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    return module


def has_batch_norm(model):
    return any(isinstance(module, nn.modules.batchnorm._BatchNorm)
               for module in model.modules())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission")
    args = parser.parse_args()
    root = Path(args.submission).resolve()
    module = load_model_module(root)
    baseline = module.build_model("baseline")
    no_bn = module.build_model("no_bn")

    params = sum(parameter.numel() for parameter in baseline.parameters())
    assert params <= 5_000_000, f"too many parameters: {params:,}"
    assert has_batch_norm(baseline), "baseline must contain BatchNorm"
    assert not has_batch_norm(no_bn), "no_bn must not contain BatchNorm"
    baseline.eval()
    no_bn.eval()
    with torch.no_grad():
        assert baseline(torch.rand(2, 3, 32, 32)).shape == (2, 10)
        assert no_bn(torch.rand(2, 3, 32, 32)).shape == (2, 10)
    print(f"Public tests passed. baseline params={params:,}")


if __name__ == "__main__":
    main()
