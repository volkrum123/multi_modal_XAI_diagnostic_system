from pathlib import Path
import csv
from datetime import datetime

FIELDS = [
    "timestamp",
    "image_path",
    "model_probability",
    "model_label",
    "reviewer_decision",
    "corrected_label",
    "comment",
]

def save_feedback(
    image_path,
    model_probability,
    model_label,
    reviewer_decision,
    corrected_label,
    comment="",
    output="outputs/feedback/feedback.csv",
):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "image_path": image_path,
            "model_probability": model_probability,
            "model_label": model_label,
            "reviewer_decision": reviewer_decision,
            "corrected_label": corrected_label,
            "comment": comment,
        })
