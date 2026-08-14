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
from src.image_only_model import ImageOnlyPneumoniaModel
from src.utils import get_device


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate the E1 image-only DenseNet-121 model."
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. If omitted, uses config.yaml threshold.",
    )

    args = parser.parse_args()

    cfg = load_config()
    device = get_device()

    print("Device:", device)
    print("Experiment: E1 - Image-only DenseNet-121")
    print("Split:", args.split)

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = Path(
        "outputs/checkpoints/e1_image_only.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Train E1 first: {checkpoint}"
        )

    bundle = torch.load(
        checkpoint,
        map_location=device,
    )

    model = ImageOnlyPneumoniaModel(
        pretrained=False,
        image_embedding_dim=cfg["model"]["image_embedding_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    model.load_state_dict(
        bundle["model_state"]
    )

    model.eval()

    print("Model loaded:", checkpoint)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Threshold
    # ---------------------------------------------------------

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = cfg["training"]["threshold"]

    print("Threshold:", threshold)

    # ---------------------------------------------------------
    # Generate predictions
    # ---------------------------------------------------------

    y_true = []
    y_prob = []
    paths = []

    with torch.no_grad():

        for images, _, targets, image_paths in loader:

            images = images.to(device)

            logits = model(images)

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

    # ---------------------------------------------------------
    # Convert probabilities to predictions
    # ---------------------------------------------------------

    y_pred = [
        int(p >= threshold)
        for p in y_prob
    ]

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = {
        "experiment": "E1",
        "model": "Image-only DenseNet-121",
        "split": args.split,
        "n": len(y_true),
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "sensitivity": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print()
    print("Evaluation metrics")
    print("-------------------")

    print(
        pd.Series(metrics)
    )

    print(
        "\nConfusion matrix:\n",
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ),
    )

    print(
        "\nClassification report:\n",
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        ),
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    if args.split == "test":
        output_filename = (
            "e1_image_only_test_metrics.csv"
        )
    else:
        output_filename = (
            "e1_image_only_val_metrics.csv"
        )

    out = Path(
        "outputs/metrics"
    ) / output_filename

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

    print(
        "\nSaved:",
        out,
    )


if __name__ == "__main__":
    main()