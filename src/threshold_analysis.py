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
    confusion_matrix,
    roc_auc_score,
)

from src.config import load_config
from src.dataset import NIHPneumoniaDataset
from src.model import MultiModalPneumoniaModel
from src.utils import get_device


def load_model(cfg, device):
    checkpoint = Path(cfg["output"]["checkpoint"])

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
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

    model.load_state_dict(bundle["model_state"])
    model.eval()

    return model


def get_predictions(model, loader, device):
    y_true = []
    y_prob = []

    with torch.no_grad():
        for images, metadata, targets, _ in loader:

            images = images.to(device)
            metadata = metadata.to(device)

            logits = model(images, metadata)

            probabilities = torch.sigmoid(logits)

            y_true.extend(
                targets.cpu().numpy().tolist()
            )

            y_prob.extend(
                probabilities.cpu().numpy().tolist()
            )

    return y_true, y_prob


def calculate_metrics(y_true, y_prob, threshold):

    y_pred = [
        int(probability >= threshold)
        for probability in y_prob
    ]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
    )

    args = parser.parse_args()

    cfg = load_config()

    device = get_device()

    print("Device:", device)
    print("Experiment: E2 threshold analysis")
    print("Split:", args.split)

    model = load_model(
        cfg,
        device,
    )

    dataset = NIHPneumoniaDataset(
        cfg["data"]["manifest"],
        args.split,
        cfg["data"]["image_size"],
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    print("Samples:", len(dataset))

    y_true, y_prob = get_predictions(
        model,
        loader,
        device,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    print()
    print(f"ROC-AUC: {roc_auc:.4f}")

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            y_prob,
            threshold,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    print()
    print(results_df.to_string(index=False))

    output_path = Path(
        f"outputs/metrics/e2_{args.split}_threshold_analysis.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Saved:", output_path)

    # Find threshold with highest F1.
    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print()
    print("Best threshold by F1")
    print("--------------------")
    print(
        f"Threshold:   {best_f1['threshold']:.2f}"
    )
    print(
        f"Accuracy:    {best_f1['accuracy']:.4f}"
    )
    print(
        f"Precision:   {best_f1['precision']:.4f}"
    )
    print(
        f"Sensitivity: {best_f1['sensitivity']:.4f}"
    )
    print(
        f"Specificity: {best_f1['specificity']:.4f}"
    )
    print(
        f"F1:          {best_f1['f1']:.4f}"
    )


if __name__ == "__main__":
    main()