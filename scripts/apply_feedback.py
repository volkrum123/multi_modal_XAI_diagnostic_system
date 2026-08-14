from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
feedback_path = ROOT / "outputs/feedback/feedback.csv"
manifest_path = ROOT / "data/processed/nih_pneumonia_manifest.csv"
out_path = ROOT / "data/processed/nih_pneumonia_manifest_hitl.csv"

if not feedback_path.exists():
    raise FileNotFoundError("No HITL feedback file exists yet.")

manifest = pd.read_csv(manifest_path)
feedback = pd.read_csv(feedback_path)

feedback = feedback.dropna(subset=["corrected_label"]).copy()
feedback["corrected_label"] = feedback["corrected_label"].astype(int)

updated = manifest.merge(
    feedback[["image_path", "corrected_label"]],
    on="image_path",
    how="left",
)
updated["target"] = updated["corrected_label"].fillna(updated["target"]).astype(int)
updated = updated.drop(columns=["corrected_label"])

updated.to_csv(out_path, index=False)
print(f"Saved HITL-adjusted manifest: {out_path}")
