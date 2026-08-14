from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


CASES_PATH = Path(
    "outputs/explanations/xai_cases.csv"
)

OUTPUT_PATH = Path(
    "outputs/figures/e3_gradcam_four_cases.png"
)


def resolve_image_path(path):
    """
    Resolve an NIH image path.

    The manifest may point directly to:
        data/raw/images/image.png

    while the actual NIH dataset may store images inside
    numbered subdirectories:
        data/raw/images/<folder>/image.png
    """

    path = Path(path)

    # First try the path exactly as provided.
    if path.exists():
        return path

    # Otherwise search one directory level below the
    # expected image directory.
    candidates = list(
        path.parent.glob(
            f"*/{path.name}"
        )
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Image not found: {path}"
    )


def load_image(path):
    image = Image.open(path).convert("RGB")

    return np.asarray(image)


def load_heatmap(path):
    heatmap = np.load(path)

    # Remove negative attribution values.
    heatmap = np.maximum(
        heatmap,
        0,
    )

    # Normalize heatmap to [0, 1].
    if heatmap.max() > 0:
        heatmap = (
            heatmap
            / heatmap.max()
        )

    return heatmap


def resize_heatmap(
    heatmap,
    width,
    height,
):
    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        (width, height)
    )

    return (
        np.asarray(
            heatmap_image
        )
        .astype(np.float32)
        / 255.0
    )


def main():

    # --------------------------------------------------
    # Check that the XAI cases file exists.
    # --------------------------------------------------

    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"Missing XAI cases file: {CASES_PATH}"
        )

    cases = pd.read_csv(
        CASES_PATH
    )

    # --------------------------------------------------
    # Validate that the expected classification
    # categories are present.
    # --------------------------------------------------

    expected_categories = [
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    missing_categories = [
        category
        for category in expected_categories
        if category not in cases["category"].values
    ]

    if missing_categories:
        raise ValueError(
            "Missing XAI categories: "
            + ", ".join(missing_categories)
        )

    # --------------------------------------------------
    # Arrange the cases in a consistent order:
    #
    # TP -> TN -> FP -> FN
    # --------------------------------------------------

    cases = (
        cases
        .set_index("category")
        .loc[expected_categories]
        .reset_index()
    )

    # --------------------------------------------------
    # Create output directory.
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Create the four-case Grad-CAM figure.
    #
    # Top row:
    #   Original X-rays
    #
    # Bottom row:
    #   Grad-CAM overlays
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    # --------------------------------------------------
    # Process each classification outcome.
    # --------------------------------------------------

    for column, (_, row) in enumerate(
        cases.iterrows()
    ):

        # ----------------------------------------------
        # Resolve the original image path.
        #
        # This supports both:
        #
        # data/raw/images/image.png
        #
        # and:
        #
        # data/raw/images/<folder>/image.png
        # ----------------------------------------------

        image_path = resolve_image_path(
            row["path"]
        )

        # ----------------------------------------------
        # The Grad-CAM files use the image filename:
        #
        # 00020408_014.png
        #
        # becomes:
        #
        # 00020408_014_gradcam.npy
        # ----------------------------------------------

        heatmap_path = (
            Path("outputs/explanations")
            / f"{image_path.stem}_gradcam.npy"
        )

        if not heatmap_path.exists():
            raise FileNotFoundError(
                f"Heatmap not found: {heatmap_path}"
            )

        # ----------------------------------------------
        # Load image and Grad-CAM heatmap.
        # ----------------------------------------------

        image = load_image(
            image_path
        )

        heatmap = load_heatmap(
            heatmap_path
        )

        # ----------------------------------------------
        # Resize heatmap to the original image
        # dimensions.
        # ----------------------------------------------

        height, width = image.shape[:2]

        heatmap = resize_heatmap(
            heatmap,
            width,
            height,
        )

        # ----------------------------------------------
        # Retrieve classification information.
        # ----------------------------------------------

        category = row["category"]

        target = int(
            row["target"]
        )

        probability = float(
            row["probability"]
        )

        prediction = int(
            row["prediction"]
        )

        # ----------------------------------------------
        # Convert numerical labels into readable labels.
        # ----------------------------------------------

        actual_label = (
            "Pneumonia"
            if target == 1
            else "No pneumonia"
        )

        predicted_label = (
            "Pneumonia"
            if prediction == 1
            else "No pneumonia"
        )

        # ----------------------------------------------
        # Top row: original X-ray.
        # ----------------------------------------------

        axes[0, column].imshow(
            image,
            cmap="gray",
        )

        axes[0, column].set_title(
            f"{category}\n"
            f"Actual: {actual_label}",
            fontsize=12,
            fontweight="bold",
        )

        axes[0, column].axis(
            "off"
        )

        # ----------------------------------------------
        # Bottom row: Grad-CAM overlay.
        # ----------------------------------------------

        axes[1, column].imshow(
            image,
            cmap="gray",
        )

        axes[1, column].imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
        )

        axes[1, column].set_title(
            f"Predicted: {predicted_label}\n"
            f"Probability: {probability:.2%}",
            fontsize=11,
        )

        axes[1, column].axis(
            "off"
        )

    # --------------------------------------------------
    # Overall figure title.
    # --------------------------------------------------

    fig.suptitle(
        "E3: Grad-CAM Explanations Across Classification Outcomes",
        fontsize=16,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Improve spacing.
    # --------------------------------------------------

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    # --------------------------------------------------
    # Save figure.
    # --------------------------------------------------

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()