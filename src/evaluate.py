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
    classification_report,
)

from src.config import load_config
from src.dataset import NIHPneumoniaDataset
from src.model import MultiModalPneumoniaModel
from src.utils import get_device


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. If omitted, uses config threshold.",
    )

    args = parser.parse_args()

    cfg = load_config()

    device = get_device()

    print("Device:", device)
    print("Experiment: E2 - Multi-modal DenseNet-121 + metadata")
    print("Split:", args.split)

    checkpoint = Path(
        cfg["output"]["checkpoint"]
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Train a model first: {checkpoint}"
        )

    bundle = torch.load(
        checkpoint,
        map_location=device,
    )

    model = MultiModalPneumoniaModel(
        pretrained=False,
        metadata_dim=cfg["model"]["metadata_dim"],
        image_embedding_dim=cfg["model"]["image_embedding_dim"],
        metadata_embedding_dim=cfg["model"]["metadata_embedding_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    model.load_state_dict(
        bundle["model_state"]
    )

    model.eval()

    ds = NIHPneumoniaDataset(
        cfg["data"]["manifest"],
        args.split,
        cfg["data"]["image_size"],
        augment=False,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    print("Samples:", len(ds))

    y_true = []
    y_prob = []
    paths = []

    with torch.no_grad():

        for images, metadata, targets, image_paths in loader:

            images = images.to(device)
            metadata = metadata.to(device)

            logits = model(
                images,
                metadata,
            )

            probs = torch.sigmoid(
                logits
            ).cpu().numpy()

            y_prob.extend(
                probs.tolist()
            )

            y_true.extend(
                targets.numpy().tolist()
            )

            paths.extend(
                image_paths
            )

    threshold = (
        args.threshold
        if args.threshold is not None
        else cfg["training"]["threshold"]
    )

    y_pred = [
        int(probability >= threshold)
        for probability in y_prob
    ]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # Specificity
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # Classification metrics
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ROC-AUC
    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    # PR-AUC / Average Precision
    pr_auc = average_precision_score(
        y_true,
        y_prob,
    )

    metrics = {
        "experiment": "E2",
        "model": "Multi-modal DenseNet-121 + metadata",
        "split": args.split,
        "n": len(y_true),
        "threshold": threshold,

        "accuracy": accuracy,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    print()
    print("Evaluation metrics")
    print("-------------------")

    print(
        pd.Series(metrics)
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    # Save metrics
    out = Path(
        cfg["output"]["metrics"]
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        out,
        index=False,
    )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()