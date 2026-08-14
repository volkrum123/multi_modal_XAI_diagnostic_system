from pathlib import Path
import pandas as pd


INPUT_FILE = Path(
    "outputs/metrics/overall_results.csv"
)

OUTPUT_FILE = Path(
    "outputs/metrics/e1_e2_comparison.csv"
)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {INPUT_FILE}"
        )

    results = pd.read_csv(INPUT_FILE)

    results = results[
        results["experiment"].isin(["E1", "E2"])
    ].copy()

    if len(results) != 2:
        raise ValueError(
            "Expected exactly two rows: E1 and E2."
        )

    e1 = results[
        results["experiment"] == "E1"
    ].iloc[0]

    e2 = results[
        results["experiment"] == "E2"
    ].iloc[0]

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall / Sensitivity", "recall"),
        ("Specificity", "specificity"),
        ("F1-score", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]

    rows = []

    for name, column in metrics:

        e1_value = float(e1[column])
        e2_value = float(e2[column])

        difference = e2_value - e1_value

        rows.append({
            "metric": name,
            "E1_image_only": e1_value,
            "E2_multimodal": e2_value,
            "absolute_change": difference,
            "change_percentage_points": difference * 100,
        })

    comparison = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("E1 vs E2 Comparison")
    print("===================")

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()