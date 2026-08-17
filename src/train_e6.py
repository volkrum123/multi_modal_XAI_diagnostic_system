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

            logits = model(
                images,
                metadata,
            )

            probs = torch.sigmoid(
                logits
            ).cpu()

            ys.extend(
                targets.numpy()
            )

            ps.extend(
                probs.numpy()
            )

    preds = [
        int(p >= 0.5)
        for p in ps
    ]

    tn, fp, fn, tp = confusion_matrix(
        ys,
        preds,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        ys,
        preds,
    )

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    roc_auc = (
        roc_auc_score(
            ys,
            ps,
        )
        if len(set(ys)) > 1
        else float("nan")
    )

    pr_auc = (
        average_precision_score(
            ys,
            ps,
        )
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


def train_experiment(
    preprocessing,
    cfg,
    args,
    device,
):

    experiment_name = (
        "E6-A"
        if preprocessing == "baseline"
        else "E6-B"
    )

    print()
    print("=" * 70)
    print(
        f"{experiment_name} - "
        f"Preprocessing: {preprocessing}"
    )
    print("=" * 70)

    manifest = cfg["data"]["manifest"]
    image_size = cfg["data"]["image_size"]

    train_ds = NIHPneumoniaDataset(
        manifest,
        "train",
        image_size,
        augment=True,
        preprocessing=preprocessing,
    )

    val_ds = NIHPneumoniaDataset(
        manifest,
        "val",
        image_size,
        augment=False,
        preprocessing=preprocessing,
    )

    if args.max_samples:

        train_ds.df = (
            train_ds.df
            .head(args.max_samples)
            .copy()
        )

    print(
        "Preprocessing:",
        preprocessing,
    )

    print(
        "Training samples:",
        len(train_ds),
    )

    print(
        "Validation samples:",
        len(val_ds),
    )

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

    model.freeze_backbone()

    y = train_ds.df["target"]

    positives = max(
        int(y.sum()),
        1,
    )

    negatives = max(
        len(y) - positives,
        1,
    )

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

    epochs = (
        args.epochs
        or cfg["training"]["epochs"]
    )

    best_auc = -1.0

    history = []

    checkpoint = Path(
        "outputs/checkpoints/"
        f"{experiment_name.lower().replace('-', '_')}_"
        f"multimodal.pt"
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(epochs):

        if (
            epoch
            == cfg["training"]["freeze_backbone_epochs"]
        ):
            model.unfreeze_backbone()

        model.train()

        running_loss = 0.0

        for images, metadata, targets, _ in tqdm(
            train_loader,
            desc=(
                f"{experiment_name} "
                f"Epoch {epoch + 1}/{epochs}"
            ),
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

        train_loss = (
            running_loss
            / len(train_loader)
        )

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        epoch_record = {
            "experiment": experiment_name,
            "preprocessing": preprocessing,
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

        history.append(
            epoch_record
        )

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
                    "experiment": experiment_name,
                    "preprocessing": preprocessing,
                    "best_val_auc": best_auc,
                },
                checkpoint,
            )

            print(
                "Saved:",
                checkpoint,
            )

    metrics_path = Path(
        "outputs/metrics/"
        f"{experiment_name.lower().replace('-', '_')}_"
        f"metrics.csv"
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
    print(
        f"{experiment_name} complete."
    )

    print(
        "Best validation AUC:",
        best_auc,
    )

    print(
        "Metrics saved:",
        metrics_path,
    )

    return history


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    cfg = load_config()

    device = get_device()

    print("Device:", device)
    print(
        "Experiment: E6 - "
        "Preprocessing comparison"
    )

    all_history = []

    # --------------------------------------------------
    # E6-A: Existing baseline preprocessing
    # --------------------------------------------------

    seed_everything(
        cfg["seed"]
    )

    baseline_history = train_experiment(
        "baseline",
        cfg,
        args,
        device,
    )

    all_history.extend(
        baseline_history
    )

    # --------------------------------------------------
    # E6-B: CLAHE preprocessing
    # --------------------------------------------------

    seed_everything(
        cfg["seed"]
    )

    clahe_history = train_experiment(
        "clahe",
        cfg,
        args,
        device,
    )

    all_history.extend(
        clahe_history
    )

    # --------------------------------------------------
    # Combined results
    # --------------------------------------------------

    combined_path = Path(
        "outputs/metrics/"
        "e6_preprocessing_metrics.csv"
    )

    pd.DataFrame(
        all_history
    ).to_csv(
        combined_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("E6 COMPLETE")
    print("=" * 70)
    print(
        "Combined metrics:",
        combined_path,
    )


if __name__ == "__main__":
    main()