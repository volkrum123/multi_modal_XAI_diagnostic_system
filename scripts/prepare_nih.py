from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

META = RAW / "Data_Entry_2017_v2020.csv"
TRAIN = RAW / "train_val_list.txt"
TEST = RAW / "test_list.txt"

if not META.exists():
    raise FileNotFoundError(
        f"Missing {META}. Download the NIH ChestX-ray14 metadata CSV "
        "and place it in data/raw/."
    )

df = pd.read_csv(META)

required = ["Image Index", "Finding Labels", "Patient ID", "Patient Age",
            "Patient Sex", "View Position"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Keep only clean binary targets:
# Pneumonia = positive
# No Finding = negative
labels = df["Finding Labels"].fillna("")
positive = labels.str.contains(r"\bPneumonia\b", regex=True, na=False)
negative = labels.eq("No Finding")

df = df.loc[positive | negative].copy()
df["target"] = positive.loc[df.index].astype(int)

# Normalize age; NIH contains an occasional outlier/invalid age.
df["age"] = pd.to_numeric(df["Patient Age"], errors="coerce").clip(0, 120)
df["sex"] = df["Patient Sex"].map({"M": 1, "F": 0})
df["view"] = df["View Position"].map({"PA": 0, "AP": 1})

df = df.dropna(subset=["age", "sex", "view"]).copy()

# Standardize image path to the raw image root.
df["image_path"] = df["Image Index"].apply(lambda x: str(RAW / "images" / x))

# Patient-level deterministic split.
patients = df["Patient ID"].astype(str).drop_duplicates().to_numpy()
rng = np.random.default_rng(42)
rng.shuffle(patients)

n = len(patients)
train_patients = set(patients[:int(0.70*n)])
val_patients = set(patients[int(0.70*n):int(0.85*n)])
test_patients = set(patients[int(0.85*n):])

df["split"] = np.where(
    df["Patient ID"].astype(str).isin(train_patients), "train",
    np.where(df["Patient ID"].astype(str).isin(val_patients), "val", "test")
)

manifest = df[
    ["image_path", "Patient ID", "target", "age", "sex", "view", "split",
     "Finding Labels"]
].copy()

manifest.to_csv(OUT / "nih_pneumonia_manifest.csv", index=False)

print(f"Saved {len(manifest):,} records to {OUT / 'nih_pneumonia_manifest.csv'}")
print(manifest.groupby(["split", "target"]).size())
