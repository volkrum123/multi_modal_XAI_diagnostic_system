from pathlib import Path
import pandas as pd

from src.feedback import save_feedback


CASES_FILE = Path("outputs/explanations/xai_cases.csv")
XAI_DIR = Path("outputs/explanations")


def get_label(value):
    return "Pneumonia" if int(value) == 1 else "No pneumonia"


def main():
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"XAI cases file not found: {CASES_FILE}"
        )

    cases = pd.read_csv(CASES_FILE)

    print()
    print("=" * 60)
    print("E4 HUMAN-IN-THE-LOOP REVIEW")
    print("=" * 60)
    print()
    print(f"Cases to review: {len(cases)}")
    print()

    for index, row in cases.iterrows():

        image_path = Path(row["path"])
        image_id = image_path.stem

        probability = float(row["probability"])
        prediction = int(row["prediction"])

        overlay_path = (
            XAI_DIR / f"{image_id}_gradcam_overlay.png"
        )

        print()
        print("=" * 60)
        print(f"CASE {index + 1}/{len(cases)}")
        print("=" * 60)

        print(f"Image:              {image_path}")
        print(f"Model probability:  {probability:.4f}")
        print(f"Model prediction:   {get_label(prediction)}")
        print(f"Grad-CAM overlay:   {overlay_path}")

        print()
        print("Reviewer decision")
        print("-----------------")
        print("1 = Agree with model prediction")
        print("2 = Disagree with model prediction")

        while True:
            decision = input("Decision [1/2]: ").strip()

            if decision in ("1", "2"):
                break

            print("Please enter 1 or 2.")

        reviewer_decision = (
            "agree"
            if decision == "1"
            else "disagree"
        )

        print()
        print("Corrected label")
        print("---------------")
        print("0 = No pneumonia")
        print("1 = Pneumonia")

        while True:
            corrected = input(
                "Corrected label [0/1]: "
            ).strip()

            if corrected in ("0", "1"):
                break

            print("Please enter 0 or 1.")

        corrected_label = get_label(int(corrected))

        print()
        comment = input(
            "Reviewer comment (optional): "
        ).strip()

        save_feedback(
            image_path=str(image_path),
            model_probability=probability,
            model_label=get_label(prediction),
            reviewer_decision=reviewer_decision,
            corrected_label=corrected_label,
            comment=comment,
        )

        print()
        print("Feedback saved.")

    print()
    print("=" * 60)
    print("HITL REVIEW COMPLETE")
    print("=" * 60)
    print()
    print("Feedback saved to:")
    print("outputs/feedback/feedback.csv")


if __name__ == "__main__":
    main()