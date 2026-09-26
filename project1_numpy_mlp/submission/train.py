"""P1 starter — training loop skeleton.

Run:  python train.py --data mnist.npz --out weights.npz --log train_log.csv
"""
import argparse
import csv

import numpy as np

from mlp import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mnist.npz")
    ap.add_argument("--out", default="weights.npz")
    ap.add_argument("--log", default="train_log.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    d = np.load(args.data)
    x_train, y_train = d["x_train"], d["y_train"]
    x_val, y_val = d["x_test"], d["y_test"]  # use as dev set

    model = build_model(seed=0)
    rows = []
    for epoch in range(args.epochs):
        # TODO: shuffle; loop over mini-batches:
        #   logits = model.forward(xb)
        #   loss   = model.loss_fn.forward(logits, yb)
        #   dlogits = model.loss_fn.backward()
        #   backprop through layers; SGD update W -= lr * dW, b -= lr * db
        train_loss = 0.0  # TODO: epoch mean loss
        val_acc = float((model.predict(x_val) == y_val).mean())
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch}  loss {train_loss:.4f}  val_acc {val_acc:.4f}")

    model.save(args.out)
    with open(args.log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
