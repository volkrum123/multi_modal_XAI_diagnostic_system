import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from tqdm import tqdm

from src.config import load_config
from src.dataset import NIHPneumoniaDataset
from src.model import MultiModalPneumoniaModel
from src.utils import seed_everything, get_device


def evaluate(model, loader, device):
    model.eval()

    ys = []
    ps = []

    with torch.no_grad():
        for images, metadata, targets, _ in loader:
            images = images.to(device)
            metadata = metadata.to(device)

            logits = model(images, metadata)
            probs = torch.sigmoid(logits).cpu()

            ys.extend(targets.numpy())
            ps.extend(probs.numpy())

    # Convert probabilities into binary predictions
    preds = [int(p >= 0.5) for p in ps]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        ys,
        preds,
        labels=[0, 1],
    ).ravel()

    # Classification metrics
    accuracy = accuracy_score(ys, preds)

    precision = precision_score(
        ys,
        preds,
        zero_division=0,
    )

    recall = recall_score(
        ys,
        preds,
        zero_division=0,
    )

    f1 = f1_score(
        ys,
        preds,
        zero_division=0,
    )

    # Specificity = TN / (TN + FP)
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # ROC-AUC
    roc_auc = (
        roc_auc_score(ys, ps)
        if len(set(ys)) > 1
        else float("nan")
    )

    # PR-AUC / Average Precision
    pr_auc = (
        average_precision_score(ys, ps)
        if len(set(ys)) > 1
        else float("nan")
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    cfg = load_config()

    seed_everything(cfg["seed"])

    device = get_device()

    print("Device:", device)
    print("Experiment: E2 - Multi-modal DenseNet-121 + metadata")

    manifest = cfg["data"]["manifest"]
    image_size = cfg["data"]["image_size"]

    train_ds = NIHPneumoniaDataset(
        manifest,
        "train",
        image_size,
        augment=True,
    )

    val_ds = NIHPneumoniaDataset(
        manifest,
        "val",
        image_size,
        augment=False,
    )

    if args.max_samples:
        train_ds.df = train_ds.df.head(args.max_samples).copy()

    print("Training samples:", len(train_ds))
    print("Validation samples:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    model = MultiModalPneumoniaModel(
        pretrained=cfg["model"]["pretrained"],
        metadata_dim=cfg["model"]["metadata_dim"],
        image_embedding_dim=cfg["model"]["image_embedding_dim"],
        metadata_embedding_dim=cfg["model"]["metadata_embedding_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Freeze DenseNet backbone initially.
    model.freeze_backbone()

    # Calculate class weighting.
    y = train_ds.df["target"]

    positives = max(int(y.sum()), 1)
    negatives = max(len(y) - positives, 1)

    pos_weight = torch.tensor(
        [negatives / positives],
        device=device,
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = args.epochs or cfg["training"]["epochs"]

    best_auc = -1.0

    checkpoint = Path(
        cfg["output"]["checkpoint"]
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Store metrics for every epoch.
    history = []

    for epoch in range(epochs):

        # Unfreeze DenseNet after the configured number
        # of frozen epochs.
        if epoch == cfg["training"]["freeze_backbone_epochs"]:
            model.unfreeze_backbone()

        model.train()

        running_loss = 0.0

        for images, metadata, targets, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        ):
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(
                images,
                metadata,
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        epoch_record = {
            "experiment": "E2",
            "model": "Multi-modal DenseNet-121",
            "epoch": epoch + 1,
            "training_samples": len(train_ds),
            "validation_samples": len(val_ds),
            "train_loss": train_loss,

            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],

            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"specificity={metrics['specificity']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "experiment": "E2",
                    "best_val_auc": best_auc,
                },
                checkpoint,
            )

            print("Saved:", checkpoint)

    # Save experiment history.
    metrics_path = Path(
        "outputs/metrics/e2_multimodal_metrics.csv"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(history).to_csv(
        metrics_path,
        index=False,
    )

    print()
    print("Experiment complete.")
    print("Best validation AUC:", best_auc)
    print("Metrics saved:", metrics_path)


if __name__ == "__main__":
    main()