import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MMFormer
from utils.data import (
    HSIDataset,
    load_dataset,
    prepare_data,
    split_by_test_ratio,
    split_fixed_train_per_class,
)
from utils.metrics import compute_metrics, format_metric_report, format_summary_report, summarize_history
from utils.visualization import save_prediction_and_gt_maps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MMFormer for small-sample Mars hyperspectral image classification."
    )
    parser.add_argument("--dataset", type=str, default="HC", choices=["HC", "NF", "UP"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")

    parser.add_argument("--use_pca", dest="use_pca", action="store_true", default=True)
    parser.add_argument("--no_pca", dest="use_pca", action="store_false")
    parser.add_argument("--pca_components", type=int, default=30)
    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--train_samples_per_class", type=int, default=10)
    parser.add_argument("--test_ratio", type=float, default=None)

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--spatial_macro_kernel_sizes", type=int, nargs="+", default=[5, 7, 9])
    parser.add_argument("--spectral_macro_kernel_ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--spatial_micro_kernel_size", type=int, default=3)
    parser.add_argument("--spectral_micro_kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--save_maps", action="store_true")
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(input_channels, num_classes, args, device):
    return MMFormer(
        in_channels=input_channels,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        spatial_macro_kernel_sizes=args.spatial_macro_kernel_sizes,
        spectral_macro_kernel_ratios=args.spectral_macro_kernel_ratios,
        spatial_micro_kernel_size=args.spatial_micro_kernel_size,
        spectral_micro_kernel_size=args.spectral_micro_kernel_size,
        dropout=args.dropout,
    ).to(device)


def build_loaders(x_train, x_test, y_train, y_test, args):
    train_loader = DataLoader(
        HSIDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        HSIDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_run(model, loader, optimizer, criterion, device, epochs):
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress = tqdm(loader, desc=f"Train epoch {epoch + 1}/{epochs}")
        for batch_data, batch_labels in progress:
            x = batch_data.to(device)
            labels = batch_labels.to(device)
            batch_size = x.shape[0]

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

    return time.time() - start_time


def evaluate(model, loader, num_classes, device):
    start_time = time.time()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc="Testing"):
            logits = model(batch_data.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch_labels.numpy()
            for true_label, pred_label in zip(labels, preds):
                confusion_matrix[int(true_label), int(pred_label)] += 1

    return confusion_matrix, time.time() - start_time


def split_dataset(cubes, labels, args, seed):
    if args.test_ratio is not None:
        return split_by_test_ratio(cubes, labels, test_ratio=args.test_ratio, random_state=seed)
    return split_fixed_train_per_class(
        cubes,
        labels,
        train_samples_per_class=args.train_samples_per_class,
        random_state=seed,
    )


def save_run_report(save_dir, run_idx, metrics, confusion_matrix, total_time):
    run_dir = save_dir / f"run_{run_idx + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = format_metric_report(metrics, total_time=total_time)
    (run_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    np.savetxt(run_dir / "confusion_matrix.txt", confusion_matrix, fmt="%d")


def save_checkpoint(save_dir, model_state, optimizer_state, args, best_oa, best_run_idx):
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "oa": best_oa,
        "run_idx": best_run_idx,
        "args": vars(args),
    }
    checkpoint_path = save_dir / f"best_model_OA_{best_oa:.4f}.pth"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def main():
    args = parse_args()
    set_random_seed(args.base_seed)
    device = resolve_device(args.device)

    data, labels, config = load_dataset(args.dataset, args.data_dir)
    cubes, cube_labels, input_channels = prepare_data(
        data,
        labels,
        patch_size=args.patch_size,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
    )

    save_root = Path(args.output_dir) / config.name / "MMFormer"
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    all_loader = DataLoader(
        HSIDataset(cubes, cube_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    history = []
    best_oa = -1.0
    best_run_idx = -1
    best_model_state = None
    best_optimizer_state = None

    print(f"Dataset: {config.name}")
    print(f"Samples: {len(cube_labels)} | Classes: {config.num_classes} | Input channels: {input_channels}")
    print(f"Device: {device}")
    print(f"Results: {save_root}")

    for run_idx in range(args.runs):
        seed = args.base_seed + run_idx
        set_random_seed(seed)
        print(f"\nRun {run_idx + 1}/{args.runs} | Seed: {seed}")

        x_train, x_test, y_train, y_test = split_dataset(cubes, cube_labels, args, seed)
        print(f"Train samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}")

        train_loader, test_loader = build_loaders(x_train, x_test, y_train, y_test, args)
        model = build_model(input_channels, config.num_classes, args, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        train_time = train_one_run(model, train_loader, optimizer, criterion, device, args.epochs)
        confusion_matrix, test_time = evaluate(model, test_loader, config.num_classes, device)
        total_time = train_time + test_time
        metrics = compute_metrics(confusion_matrix)
        metrics["time"] = total_time
        history.append(metrics)

        save_run_report(save_root, run_idx, metrics, confusion_matrix, total_time)
        print(format_metric_report(metrics, total_time=total_time))

        if metrics["OA"] > best_oa:
            best_oa = metrics["OA"]
            best_run_idx = run_idx
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            print(f"New best model: OA={best_oa:.4f}")

    summary = summarize_history(history)
    summary_text = format_summary_report(summary)
    (save_root / "overall_metrics.txt").write_text(summary_text + "\n", encoding="utf-8")
    print("\nFinal summary")
    print(summary_text)

    if best_model_state is not None:
        best_model_dir = save_root / "best_model"
        checkpoint_path = save_checkpoint(
            best_model_dir,
            best_model_state,
            best_optimizer_state,
            args,
            best_oa,
            best_run_idx,
        )
        print(f"Best checkpoint: {checkpoint_path}")

        if args.save_maps:
            best_model = build_model(input_channels, config.num_classes, args, device)
            best_model.load_state_dict(best_model_state)
            pred_path, gt_path = save_prediction_and_gt_maps(
                best_model,
                all_loader,
                device,
                labels,
                config.name,
                best_model_dir,
                best_oa,
            )
            print(f"Prediction map: {pred_path}")
            print(f"Ground-truth map: {gt_path}")


if __name__ == "__main__":
    main()
