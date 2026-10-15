"""Download the official CIFAR-10 train and test sets with torchvision."""

import argparse

import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="cifar_data")
    args = parser.parse_args()
    train = torchvision.datasets.CIFAR10(args.root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(args.root, train=False, download=True)
    print(f"CIFAR-10 ready: train={len(train)}, test={len(test)}, root={args.root}")


if __name__ == "__main__":
    main()
