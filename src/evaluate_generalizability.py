from pathlib import Path

import numpy as np
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

from src.config import load_config
from src.dataset import NIHPneumoniaDataset
from src.model import MultiModalPneumoniaModel
from src.utils import get_device


CHECKPOINT = Path("outputs/checkpoints/e2_multimodal.pt")
MANIFEST = Path("data/processed/nih_pneumonia_manifest.csv")
OUTPUT = Path("outputs/metrics/e5_generalizability.csv")

THRESHOLD = 0.75


def calculate_metrics(y_true, probabilities, threshold=THRESHOLD):

    predictions = (np.array(probabilities) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = (
        roc_auc_score(y_true, probabilities)
        if len(set(y_true)) > 1
        else float("nan")
    )

    pr_auc = (
        average_precision_score(y_true, probabilities)
        if len(set(y_true)) > 1
        else float("nan")
    )

    return {
        "n": len(y_true),
        "threshold": threshold,
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


def evaluate_loader(model, loader, device):

    model.eval()

    y_true = []
    probabilities = []

    with torch.no_grad():

        for images, metadata, targets, _ in loader:

            images = images.to(device)
            metadata = metadata.to(device)

            logits = model(
                images,
                metadata,
            )

            probs = torch.sigmoid(logits)

            y_true.extend(
                targets.cpu().numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.array(y_true), np.array(probabilities)


def evaluate_subgroup(
    df,
    model,
    device,
    image_size,
    batch_size,
    num_workers,
    subgroup_name,
    subgroup_column,
    subgroup_value,
):

    subgroup_df = df[
        df[subgroup_column] == subgroup_value
    ].copy()

    if len(subgroup_df) == 0:
        return None

    # Create a temporary manifest containing only this subgroup.
    temp_manifest = Path(
        "data/processed/_e5_temp_manifest.csv"
    )

    subgroup_df.to_csv(
        temp_manifest,
        index=False,
    )

    dataset = NIHPneumoniaDataset(
        temp_manifest,
        "test",
        image_size,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    y_true, probabilities = evaluate_loader(
        model,
        loader,
        device,
    )

    metrics = calculate_metrics(
    y_true,
    probabilities,
    )

    metrics["positive_n"] = int(np.sum(y_true == 1))
    metrics["negative_n"] = int(np.sum(y_true == 0))
    metrics["positive_rate"] = (
        metrics["positive_n"] / metrics["n"]
        if metrics["n"] > 0
        else 0.0
    )

    metrics["experiment"] = "E5"
    metrics["analysis"] = "subgroup"
    metrics["subgroup"] = subgroup_name

    return metrics


def main():

    cfg = load_config()

    device = get_device()

    print("Device:", device)
    print("Experiment: E5 - Generalizability")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
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
        checkpoint["model_state"]
    )

    model.eval()

    print(
        "Loaded E2 checkpoint."
    )

    print(
        "Best validation AUC:",
        checkpoint.get("best_val_auc"),
    )

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    df = pd.read_csv(MANIFEST)

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        "Test patients:",
        test_df["Patient ID"].nunique(),
    )

    print(
        "Test images:",
        len(test_df),
    )

    # ---------------------------------------------------------
    # Overall test-set evaluation
    # ---------------------------------------------------------

    test_dataset = NIHPneumoniaDataset(
        MANIFEST,
        "test",
        cfg["data"]["image_size"],
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    y_true, probabilities = evaluate_loader(
        model,
        test_loader,
        device,
    )

    overall = calculate_metrics(
        y_true,
        probabilities,
    )

    overall["positive_n"] = int(np.sum(y_true == 1))
    overall["negative_n"] = int(np.sum(y_true == 0))
    overall["positive_rate"] = (
        overall["positive_n"] / overall["n"]
        if overall["n"] > 0
        else 0.0
    )
    overall["experiment"] = "E5"
    overall["analysis"] = "overall"
    overall["subgroup"] = "all_test_patients"

    results = [
        overall
    ]

    # ---------------------------------------------------------
    # Age subgroup analysis
    # ---------------------------------------------------------

    test_df["age_group"] = pd.cut(
        test_df["age"],
        bins=[-1, 39, 59, 120],
        labels=[
            "<40",
            "40-59",
            "60+",
        ],
    )

    for group in [
        "<40",
        "40-59",
        "60+",
    ]:

        result = evaluate_subgroup(
            test_df,
            model,
            device,
            cfg["data"]["image_size"],
            cfg["training"]["batch_size"],
            cfg["data"]["num_workers"],
            f"age_{group}",
            "age_group",
            group,
        )

        if result:
            results.append(result)

    # ---------------------------------------------------------
    # Sex subgroup analysis
    # ---------------------------------------------------------

    for value, name in [
        (0, "female"),
        (1, "male"),
    ]:

        result = evaluate_subgroup(
            test_df,
            model,
            device,
            cfg["data"]["image_size"],
            cfg["training"]["batch_size"],
            cfg["data"]["num_workers"],
            f"sex_{name}",
            "sex",
            value,
        )

        if result:
            results.append(result)

    # ---------------------------------------------------------
    # View-position subgroup analysis
    # ---------------------------------------------------------

    for value, name in [
        (0, "PA"),
        (1, "AP"),
    ]:

        result = evaluate_subgroup(
            test_df,
            model,
            device,
            cfg["data"]["image_size"],
            cfg["training"]["batch_size"],
            cfg["data"]["num_workers"],
            f"view_{name}",
            "view",
            value,
        )

        if result:
            results.append(result)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("E5 Generalizability Results")
    print("=" * 70)

    print(
        results_df[
            [
                "analysis",
                "subgroup",
                "n",
                "positive_n",
                "negative_n",
                "positive_rate",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Results saved:",
        OUTPUT,
    )

    # Clean temporary manifest.
    temp_manifest = Path(
        "data/processed/_e5_temp_manifest.csv"
    )

    if temp_manifest.exists():
        temp_manifest.unlink()


if __name__ == "__main__":
    main()