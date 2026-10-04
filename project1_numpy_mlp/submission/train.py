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

    lr=args.lr
    model = build_model(seed=0)
    rows = []
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        # TODO: shuffle; loop over mini-batches:
        indices = rng.permutation(len(x_train))  # 打乱样本下标
        epoch_loss_sum = 0.0

        for start in range(0, len(x_train), args.batch):
            batch_indices = indices[start:start + args.batch]
            xb = x_train[batch_indices]
            yb = y_train[batch_indices]

            # 前向传播与损失
            logits = model.forward(xb)
            loss   = model.loss_fn.forward(logits, yb)

            # 反向传播
            dlogits = model.loss_fn.backward()
            d_hidden = model.linear2.backward(dlogits)
            d_hidden = model.relu.backward(d_hidden)
            model.linear1.backward(d_hidden)

            # SGD update  W -= lr * dW, b -= lr * db
            model.linear1.W-=lr*model.linear1.dW
            model.linear1.b-=lr*model.linear1.db

            model.linear2.W-=lr*model.linear2.dW
            model.linear2.b-=lr*model.linear2.db
         
            # loss 是这个 batch 的平均损失，乘回样本数后累计
            epoch_loss_sum += loss * len(xb)
        train_loss = epoch_loss_sum / len(x_train)# TODO: epoch mean loss
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
