from pathlib import Path

import numpy as np
import pandas as pd


CASES = {
    "TP": "00020408_014",
    "TN": "00009641_000",
    "FP": "00027427_004",
    "FN": "00003535_025",
}

OUTPUT_DIR = Path("outputs/explanations")
METRICS_DIR = Path("outputs/metrics")


def analyze_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)

    # Basic activation statistics
    mean_activation = float(heatmap.mean())
    max_activation = float(heatmap.max())
    min_activation = float(heatmap.min())

    # Percentage of pixels with meaningful activation
    active_25 = float((heatmap >= 0.25).mean() * 100)
    active_50 = float((heatmap >= 0.50).mean() * 100)
    active_75 = float((heatmap >= 0.75).mean() * 100)

    # Location of strongest activation
    max_position = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape,
    )

    max_y, max_x = max_position

    height, width = heatmap.shape

    # Normalize coordinates to 0-1
    centroid_x = float(max_x / width)
    centroid_y = float(max_y / height)

    # Central region: middle 50% of image
    y1 = height // 4
    y2 = 3 * height // 4
    x1 = width // 4
    x2 = 3 * width // 4

    central_region = heatmap[y1:y2, x1:x2]

    central_mean = float(central_region.mean())

    # Ratio of total activation occurring in central region
    total_activation = float(heatmap.sum())

    if total_activation > 0:
        central_activation_ratio = float(
            central_region.sum() / total_activation
        )
    else:
        central_activation_ratio = 0.0

    return {
        "mean_activation": mean_activation,
        "min_activation": min_activation,
        "max_activation": max_activation,
        "active_pixels_25_percent": active_25,
        "active_pixels_50_percent": active_50,
        "active_pixels_75_percent": active_75,
        "max_activation_x": centroid_x,
        "max_activation_y": centroid_y,
        "central_mean_activation": central_mean,
        "central_activation_ratio": central_activation_ratio,
    }


def main():
    rows = []

    for category, image_id in CASES.items():

        heatmap_path = OUTPUT_DIR / f"{image_id}_gradcam.npy"

        if not heatmap_path.exists():
            print(f"Missing heatmap: {heatmap_path}")
            continue

        heatmap = np.load(heatmap_path)

        metrics = analyze_heatmap(heatmap)

        row = {
            "category": category,
            "image_id": image_id,
            "heatmap_path": str(heatmap_path),
            **metrics,
        }

        rows.append(row)

    if not rows:
        raise RuntimeError("No Grad-CAM heatmaps found.")

    df = pd.DataFrame(rows)

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = METRICS_DIR / "e3_xai_metrics.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("E3 XAI Analysis")
    print("----------------")
    print(df.to_string(index=False))

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()