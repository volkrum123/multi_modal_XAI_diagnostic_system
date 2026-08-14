from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

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


def decode_metadata(metadata):
    """
    Convert the dataset's encoded metadata back to
    human-readable values.

    Dataset encoding:
        age  = age / 100
        sex  = 1 for M, 0 for F
        view = 1 for AP, 0 for PA
    """

    age = float(metadata[0].item()) * 100.0

    sex = (
        "M"
        if float(metadata[1].item()) >= 0.5
        else "F"
    )

    view = (
        "AP"
        if float(metadata[2].item()) >= 0.5
        else "PA"
    )

    return age, sex, view


def main():
    cfg = load_config()
    device = get_device()

    print("Device:", device)

    model = load_model(cfg, device)

    print(
        "Model loaded:",
        cfg["output"]["checkpoint"],
    )

    dataset = NIHPneumoniaDataset(
        cfg["data"]["manifest"],
        "test",
        cfg["data"]["image_size"],
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    threshold = cfg["training"]["threshold"]

    results = []

    print("Evaluating test set...")

    with torch.no_grad():
        for images, metadata, targets, paths in loader:

            images = images.to(device)
            metadata = metadata.to(device)

            logits = model(
                images,
                metadata,
            )

            probabilities = torch.sigmoid(logits)

            for i in range(len(targets)):

                target = int(
                    targets[i].item()
                )

                probability = float(
                    probabilities[i].item()
                )

                prediction = int(
                    probability >= threshold
                )

                if target == 1 and prediction == 1:
                    category = "TP"

                elif target == 0 and prediction == 0:
                    category = "TN"

                elif target == 0 and prediction == 1:
                    category = "FP"

                else:
                    category = "FN"

                age, sex, view = decode_metadata(
                    metadata[i].detach().cpu()
                )

                results.append({
                    "path": paths[i],
                    "target": target,
                    "probability": probability,
                    "prediction": prediction,
                    "category": category,
                    "age": age,
                    "sex": sex,
                    "view": view,
                })

    df = pd.DataFrame(results)

    print()
    print("Test-set results")
    print("----------------")
    print(
        df["category"]
        .value_counts()
        .sort_index()
    )

    selected = []

    # --------------------------------------------------
    # TRUE POSITIVE
    # Highest-confidence correctly detected pneumonia.
    # --------------------------------------------------

    tp = df[
        df["category"] == "TP"
    ]

    if not tp.empty:
        selected.append(
            tp.loc[
                tp["probability"].idxmax()
            ]
        )

    # --------------------------------------------------
    # TRUE NEGATIVE
    # Lowest probability correctly classified negative.
    # --------------------------------------------------

    tn = df[
        df["category"] == "TN"
    ]

    if not tn.empty:
        selected.append(
            tn.loc[
                tn["probability"].idxmin()
            ]
        )

    # --------------------------------------------------
    # FALSE POSITIVE
    # Highest-confidence incorrect pneumonia prediction.
    # --------------------------------------------------

    fp = df[
        df["category"] == "FP"
    ]

    if not fp.empty:
        selected.append(
            fp.loc[
                fp["probability"].idxmax()
            ]
        )

    # --------------------------------------------------
    # FALSE NEGATIVE
    # Lowest-confidence missed pneumonia case.
    # --------------------------------------------------

    fn = df[
        df["category"] == "FN"
    ]

    if not fn.empty:
        selected.append(
            fn.loc[
                fn["probability"].idxmin()
            ]
        )

    selected_df = pd.DataFrame(
        selected
    )

    output_dir = Path(
        "outputs/explanations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        "xai_cases.csv"
    )

    selected_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Selected XAI cases")
    print("------------------")

    for _, row in selected_df.iterrows():

        print()
        print(
            "Category:",
            row["category"]
        )

        print(
            "Probability:",
            f"{row['probability']:.4f}"
        )

        print(
            "Target:",
            row["target"]
        )

        print(
            "Prediction:",
            row["prediction"]
        )

        print(
            "Age:",
            f"{row['age']:.0f}"
        )

        print(
            "Sex:",
            row["sex"]
        )

        print(
            "View:",
            row["view"]
        )

        print(
            "Image:",
            row["path"]
        )

    print()
    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()