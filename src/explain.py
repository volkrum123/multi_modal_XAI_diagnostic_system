from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from captum.attr import LayerGradCam, LayerAttribution

from src.config import load_config
from src.model import MultiModalPneumoniaModel
from src.utils import get_device


CASES_PATH = Path(
    "outputs/explanations/xai_cases.csv"
)

OUTPUT_DIR = Path(
    "outputs/explanations"
)


def load_model(cfg, device):
    checkpoint = Path(
        cfg["output"]["checkpoint"]
    )

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

    model.load_state_dict(
        bundle["model_state"]
    )

    model.eval()

    return model


def resolve_image_path(image_path):
    """
    Resolve an image path while supporting the
    common NIH directory structure.
    """

    image_path = Path(image_path)

    if image_path.exists():
        return image_path

    candidates = list(
        image_path.parent.glob(
            f"*/{image_path.name}"
        )
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )


def prepare_image(
    image_path,
    image_size,
):
    image_path = resolve_image_path(
        image_path
    )

    original_image = (
        Image.open(image_path)
        .convert("RGB")
    )

    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    image = transform(
        original_image
    ).unsqueeze(0)

    return (
        original_image,
        image,
    )


def create_metadata(
    age,
    sex,
    view,
    device,
):
    metadata = torch.tensor(
        [[
            float(age) / 100.0,

            1.0
            if str(sex).upper() == "M"
            else 0.0,

            1.0
            if str(view).upper() == "AP"
            else 0.0,
        ]],
        dtype=torch.float32,
        device=device,
    )

    return metadata


def generate_gradcam(
    model,
    image,
    metadata,
):
    """
    Generate Grad-CAM from DenseNet-121's
    final dense block.
    """

    target_layer = (
        model
        .image_encoder
        .features
        .denseblock4
    )

    def forward_func(
        image,
        metadata,
    ):
        logits = model(
            image,
            metadata,
        )

        return logits.unsqueeze(1)

    gradcam = LayerGradCam(
        forward_func,
        target_layer,
    )

    attribution = gradcam.attribute(
        image,
        target=0,
        additional_forward_args=(
            metadata,
        ),
    )

    attribution = (
        LayerAttribution.interpolate(
            attribution,
            image.shape[-2:],
        )
    )

    heatmap = (
        attribution[0]
        .mean(dim=0)
    )

    heatmap = (
        heatmap
        .detach()
        .cpu()
        .numpy()
    )

    heatmap = np.maximum(
        heatmap,
        0,
    )

    if heatmap.max() > 0:
        heatmap = (
            heatmap /
            heatmap.max()
        )

    return heatmap


def save_gradcam_visualization(
    original_image,
    heatmap,
    output_path,
    probability,
    label,
    category,
    actual_label,
    age,
    sex,
    view,
):
    original_image = (
        original_image.resize(
            (
                heatmap.shape[1],
                heatmap.shape[0],
            )
        )
    )

    image_array = (
        np.asarray(
            original_image
        ).astype(
            np.float32
        ) / 255.0
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    # --------------------------------------------------
    # Original image
    # --------------------------------------------------

    axes[0].imshow(
        image_array
    )

    axes[0].set_title(
        f"{category}\n"
        f"Actual: {actual_label}"
    )

    axes[0].axis("off")

    # --------------------------------------------------
    # Grad-CAM heatmap
    # --------------------------------------------------

    axes[1].imshow(
        heatmap,
        cmap="jet",
    )

    axes[1].set_title(
        "Grad-CAM"
    )

    axes[1].axis("off")

    # --------------------------------------------------
    # Overlay
    # --------------------------------------------------

    axes[2].imshow(
        image_array
    )

    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
    )

    axes[2].set_title(
        f"Predicted: {label}\n"
        f"Probability: {probability:.2%}\n"
        f"Age: {age:.0f} | "
        f"Sex: {sex} | "
        f"View: {view}"
    )

    axes[2].axis("off")

    fig.suptitle(
        "E3: Grad-CAM Explanation",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_case(
    model,
    device,
    cfg,
    row,
):
    image_path = Path(
        row["path"]
    )

    category = str(
        row["category"]
    )

    target = int(
        row["target"]
    )

    probability = float(
        row["probability"]
    )

    prediction = int(
        row["prediction"]
    )

    age = float(
        row["age"]
    )

    sex = str(
        row["sex"]
    )

    view = str(
        row["view"]
    )

    original_image, image = (
        prepare_image(
            image_path,
            cfg["data"]["image_size"],
        )
    )

    image = image.to(device)

    metadata = create_metadata(
        age,
        sex,
        view,
        device,
    )

    with torch.no_grad():
        model_probability = (
            torch.sigmoid(
                model(
                    image,
                    metadata,
                )
            ).item()
        )

    # Use the model probability generated here
    # rather than relying only on the CSV value.
    probability = model_probability

    label = (
        "Pneumonia"
        if prediction == 1
        else "No pneumonia"
    )

    actual_label = (
        "Pneumonia"
        if target == 1
        else "No pneumonia"
    )

    heatmap = generate_gradcam(
        model,
        image,
        metadata,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_id = (
        image_path.stem
    )

    heatmap_path = (
        OUTPUT_DIR /
        f"{image_id}_gradcam.npy"
    )

    overlay_path = (
        OUTPUT_DIR /
        f"{image_id}_gradcam_overlay.png"
    )

    np.save(
        heatmap_path,
        heatmap,
    )

    save_gradcam_visualization(
        original_image=original_image,
        heatmap=heatmap,
        output_path=overlay_path,
        probability=probability,
        label=label,
        category=category,
        actual_label=actual_label,
        age=age,
        sex=sex,
        view=view,
    )

    return {
        "category": category,
        "image_id": image_id,
        "heatmap_path": str(
            heatmap_path
        ),
        "mean_activation": float(
            heatmap.mean()
        ),
        "min_activation": float(
            heatmap.min()
        ),
        "max_activation": float(
            heatmap.max()
        ),
        "probability": probability,
        "target": target,
        "prediction": prediction,
        "age": age,
        "sex": sex,
        "view": view,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM explanations "
            "for the selected E3 XAI cases."
        )
    )

    parser.add_argument(
        "--cases",
        default=str(CASES_PATH),
        help=(
            "Path to xai_cases.csv"
        ),
    )

    args = parser.parse_args()

    cases_path = Path(
        args.cases
    )

    if not cases_path.exists():
        raise FileNotFoundError(
            f"XAI cases file not found: "
            f"{cases_path}"
        )

    cfg = load_config()
    device = get_device()

    print(
        "Device:",
        device,
    )

    model = load_model(
        cfg,
        device,
    )

    print(
        "Model loaded:",
        cfg["output"]["checkpoint"],
    )

    cases = pd.read_csv(
        cases_path
    )

    expected_categories = [
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    missing_categories = [
        category
        for category in expected_categories
        if category
        not in cases["category"].values
    ]

    if missing_categories:
        raise ValueError(
            "Missing XAI categories: "
            + ", ".join(
                missing_categories
            )
        )

    cases = (
        cases
        .set_index("category")
        .loc[expected_categories]
        .reset_index()
    )

    results = []

    print()
    print(
        "Generating Grad-CAM explanations..."
    )

    for _, row in cases.iterrows():

        print()
        print(
            f"Case: {row['category']}"
        )

        print(
            f"Image: "
            f"{Path(row['path']).name}"
        )

        print(
            f"Metadata: "
            f"Age={float(row['age']):.0f}, "
            f"Sex={row['sex']}, "
            f"View={row['view']}"
        )

        result = generate_case(
            model,
            device,
            cfg,
            row,
        )

        results.append(
            result
        )

        print(
            f"Probability: "
            f"{result['probability']:.4f}"
        )

        print(
            f"Heatmap mean: "
            f"{result['mean_activation']:.4f}"
        )

        print(
            "Saved:",
            result["heatmap_path"],
        )

    # --------------------------------------------------
    # Save summary metrics
    # --------------------------------------------------

    metrics_df = pd.DataFrame(
        results
    )

    metrics_path = (
        Path("outputs/metrics")
        / "e3_xai_case_metrics.csv"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print()
    print(
        "=" * 60
    )
    print(
        "E3 XAI GENERATION COMPLETE"
    )
    print(
        "=" * 60
    )

    print()
    print(
        "Generated cases:",
        len(results),
    )

    print(
        "Metrics saved:",
        metrics_path,
    )


if __name__ == "__main__":
    main()