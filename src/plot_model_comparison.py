from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path(
    "outputs/metrics/overall_results.csv"
)

OUTPUT_FILE = Path(
    "outputs/figures/e1_e2_model_comparison.png"
)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {INPUT_FILE}"
        )

    results = pd.read_csv(INPUT_FILE)

    # Keep only E1 and E2.
    results = results[
        results["experiment"].isin(["E1", "E2"])
    ].copy()

    if len(results) != 2:
        raise ValueError(
            "Expected exactly two rows for E1 and E2."
        )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    e1 = results[
        results["experiment"] == "E1"
    ].iloc[0]

    e2 = results[
        results["experiment"] == "E2"
    ].iloc[0]

    e1_values = [
        float(e1[m])
        for m in metrics
    ]

    e2_values = [
        float(e2[m])
        for m in metrics
    ]

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    x = range(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        [i - width / 2 for i in x],
        e1_values,
        width,
        label="E1: Image-only DenseNet-121",
    )

    ax.bar(
        [i + width / 2 for i in x],
        e2_values,
        width,
        label="E2: Multi-modal DenseNet-121 + metadata",
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "E1 vs E2 Model Performance on the Test Set"
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    # Add values above bars.
    for i, value in enumerate(e1_values):

        ax.text(
            i - width / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for i, value in enumerate(e2_values):

        ax.text(
            i + width / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.01,
        "Test set: n = 9,326; classification threshold = 0.75",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("E1/E2 comparison figure generated.")
    print("-------------------------------------")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()