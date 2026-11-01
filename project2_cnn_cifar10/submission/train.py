"""P2 training skeleton for the four required runs.

Example:
    python train.py --run baseline --data-root cifar_data --out-dir .
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from model import build_model

RUNS = ("baseline", "no_bn", "no_aug", "lr_10x")
RESULT_FIELDS = ("run_name", "test_acc", "params", "train_minutes")


def make_transforms(run_name):
    """Return train/test transforms; no_aug must disable all augmentation."""
    test_transform = transforms.ToTensor()
    if run_name == "no_aug":
        train_transform = transforms.ToTensor()
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return train_transform, test_transform


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
    return correct / total


def write_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_loss", "val_acc"))
        writer.writeheader()
        writer.writerows(rows)


def update_results(path, new_row):
    """Insert or replace one run in results.csv."""
    by_run = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as file:
            by_run = {row["run_name"]: row for row in csv.DictReader(file)}
    by_run[new_row["run_name"]] = new_row
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for run_name in RUNS:
            if run_name in by_run:
                writer.writerow(by_run[run_name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--data-root", default="cifar_data")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.epochs < 10:
        parser.error("--epochs must be at least 10 so each log has at least 10 rows")
    if args.batch_size <= 0 or args.workers < 0 or args.lr <= 0:
        parser.error("--batch-size and --lr must be positive; --workers cannot be negative")

    torch.manual_seed(0)
    device = torch.device(args.device)
    train_transform, test_transform = make_transforms(args.run)
    train_set = torchvision.datasets.CIFAR10(
        args.data_root, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        args.data_root, train=False, download=True, transform=test_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    model_variant = "no_bn" if args.run == "no_bn" else "baseline"
    model = build_model(model_variant).to(device)
    base_lr = args.lr * (10.0 if args.run == "lr_10x" else 1.0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=base_lr * 0.01,
    )

    rows = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)

        scheduler.step()
        val_acc = evaluate(model, test_loader, device)
        rows.append({
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val_acc": val_acc,
        })
        out_dir = Path(args.out_dir)
        write_log(out_dir / "logs" / f"{args.run}.csv", rows)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{args.run} epoch={epoch:03d} "
            f"loss={rows[-1]['train_loss']:.4f} val_acc={val_acc:.4f} "
            f"lr={current_lr:.6g}",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run == "baseline":
        torch.save(model.state_dict(), out_dir / "weights.pt")
    minutes = (time.perf_counter() - started) / 60.0
    params = sum(parameter.numel() for parameter in model.parameters())
    update_results(out_dir / "results.csv", {
        "run_name": args.run,
        "test_acc": f"{rows[-1]['val_acc']:.6f}",
        "params": str(params),
        "train_minutes": f"{minutes:.2f}",
    })
    print(f"finished {args.run}: acc={rows[-1]['val_acc']:.4f}, {minutes:.2f} min")


if __name__ == "__main__":
    main()
