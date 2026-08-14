from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("outputs/metrics")
SUMMARY_FILE = OUTPUT_DIR / "overall_results.csv"


def load_metric_file(filename):
    path = OUTPUT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required metrics file not found: {path}"
        )

    return pd.read_csv(path)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # E1 - Image-only model
    # ---------------------------------------------------------
    #
    # E1 metrics contain validation results for each epoch.
    # The final epoch is used for the baseline summary.
    #
    e1 = load_metric_file(
        "e1_image_only_test_metrics.csv"
    )

    e1_row = e1.iloc[0]

    e1_results = {
        "experiment": "E1",
        "description": "Image-only DenseNet-121",
        "split": e1_row["split"],
        "threshold": e1_row["threshold"],
        "accuracy": e1_row["accuracy"],
        "precision": e1_row["precision"],
        "recall": e1_row["recall"],
        "specificity": e1_row["specificity"],
        "f1": e1_row["f1"],
        "roc_auc": e1_row["roc_auc"],
        "pr_auc": e1_row["pr_auc"],
    }

    # ---------------------------------------------------------
    # E2 - Multi-modal model
    # ---------------------------------------------------------
    #
    # The threshold of 0.75 was selected using the validation
    # threshold analysis and is then evaluated on the held-out
    # test set.
    #
    e2_test = load_metric_file(
        "metrics.csv"
    )

    e2_row = e2_test.iloc[-1]

    e2_results = {
        "experiment": "E2",
        "description": "Multi-modal DenseNet-121 + metadata",
        "split": e2_row.get("split", "test"),
        "threshold": e2_row.get("threshold", 0.75),
        "accuracy": e2_row["accuracy"],
        "precision": e2_row["precision"],
        "recall": e2_row["recall"],
        "specificity": e2_row["specificity"],
        "f1": e2_row["f1"],
        "roc_auc": e2_row["roc_auc"],
        "pr_auc": e2_row["pr_auc"],
    }

    # ---------------------------------------------------------
    # E3 - XAI
    # ---------------------------------------------------------
    e3 = load_metric_file(
        "e3_xai_metrics.csv"
    )

    e3_results = {
        "experiment": "E3",
        "description": "Grad-CAM XAI analysis",
        "split": "test",
        "threshold": 0.75,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1": None,
        "roc_auc": None,
        "pr_auc": None,
        "mean_activation": e3[
            "mean_activation"
        ].mean(),
        "central_mean_activation": e3[
            "central_mean_activation"
        ].mean(),
        "central_activation_ratio": e3[
            "central_activation_ratio"
        ].mean(),
    }

    # ---------------------------------------------------------
    # E4 - Human-in-the-Loop
    # ---------------------------------------------------------
    e4 = load_metric_file(
        "e4_hitl_metrics.csv"
    )

    e4_row = e4.iloc[0]

    e4_results = {
        "experiment": "E4",
        "description": "Human-in-the-Loop review",
        "split": "test",
        "threshold": 0.75,
        "accuracy": e4_row[
            "model_accuracy"
        ],
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1": None,
        "roc_auc": None,
        "pr_auc": None,
        "reviewer_agreement_rate": e4_row[
            "reviewer_agreement_rate"
        ],
        "reviewer_accuracy": e4_row[
            "reviewer_accuracy"
        ],
        "model_errors": e4_row[
            "model_errors"
        ],
        "model_errors_overridden": e4_row[
            "model_errors_overridden"
        ],
        "model_errors_corrected": e4_row[
            "model_errors_corrected"
        ],
    }

    # ---------------------------------------------------------
    # Save overall results
    # ---------------------------------------------------------

    rows = [
        e1_results,
        e2_results,
        e3_results,
        e4_results,
    ]

    summary = pd.DataFrame(rows)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("Overall Experiment Results")
    print("==========================")
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()