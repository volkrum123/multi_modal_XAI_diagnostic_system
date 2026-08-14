from pathlib import Path
import pandas as pd


FEEDBACK_FILE = Path("outputs/feedback/feedback.csv")
CASES_FILE = Path("outputs/explanations/xai_cases.csv")
OUTPUT_FILE = Path("outputs/metrics/e4_hitl_metrics.csv")


def main():
    if not FEEDBACK_FILE.exists():
        raise FileNotFoundError(
            f"Feedback file not found: {FEEDBACK_FILE}"
        )

    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"XAI cases file not found: {CASES_FILE}"
        )

    feedback = pd.read_csv(FEEDBACK_FILE)
    cases = pd.read_csv(CASES_FILE)

    # Only analyse the current four XAI cases.
    feedback = feedback[
        feedback["image_path"].isin(cases["path"])
    ].copy()

    if feedback.empty:
        raise ValueError("No matching HITL feedback found.")

    # Match the hidden ground-truth category.
    merged = feedback.merge(
        cases[
            ["path", "target", "prediction", "category"]
        ],
        left_on="image_path",
        right_on="path",
        how="left",
    )

    # Whether the reviewer agreed with the model.
    merged["reviewer_agreed"] = (
        merged["reviewer_decision"] == "agree"
    )

    # Whether the model was correct.
    merged["model_correct"] = (
        merged["target"] == merged["prediction"]
    )

    # Convert corrected human label to numeric.
    merged["corrected_numeric"] = (
        merged["corrected_label"]
        .map({
            "Pneumonia": 1,
            "No pneumonia": 0,
        })
    )

    # Whether the human correction matches ground truth.
    merged["reviewer_correct"] = (
        merged["corrected_numeric"] == merged["target"]
    )

    # Whether HITL corrected a model error.
    merged["error_corrected"] = (
        (~merged["model_correct"])
        & merged["reviewer_correct"]
    )

    # Whether the human changed an incorrect model prediction.
    merged["model_error_overridden"] = (
        (~merged["model_correct"])
        & (~merged["reviewer_agreed"])
    )

    total = len(merged)

    agreement_rate = (
        merged["reviewer_agreed"].mean()
    )

    model_accuracy = (
        merged["model_correct"].mean()
    )

    reviewer_accuracy = (
        merged["reviewer_correct"].mean()
    )

    corrections = int(
        merged["model_error_overridden"].sum()
    )

    corrected_errors = int(
        merged["error_corrected"].sum()
    )

    metrics = pd.DataFrame([
        {
            "total_cases": total,
            "model_accuracy": model_accuracy,
            "reviewer_agreement_rate": agreement_rate,
            "reviewer_accuracy": reviewer_accuracy,
            "model_errors": int((~merged["model_correct"]).sum()),
            "model_errors_overridden": corrections,
            "model_errors_corrected": corrected_errors,
        }
    ])

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("E4 HITL Analysis")
    print("----------------")
    print(
        f"Cases reviewed:              {total}"
    )
    print(
        f"Model accuracy:              {model_accuracy:.4f}"
    )
    print(
        f"Reviewer agreement rate:     {agreement_rate:.4f}"
    )
    print(
        f"Reviewer accuracy:           {reviewer_accuracy:.4f}"
    )
    print(
        f"Model errors:                {int((~merged['model_correct']).sum())}"
    )
    print(
        f"Model errors overridden:     {corrections}"
    )
    print(
        f"Model errors corrected:      {corrected_errors}"
    )

    print()
    print("Case-level results")
    print("------------------")

    print(
        merged[
            [
                "category",
                "model_correct",
                "reviewer_agreed",
                "reviewer_correct",
                "error_corrected",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()