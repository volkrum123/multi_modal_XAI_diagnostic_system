import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

from src.dataset import NIHPneumoniaDataset
from src.model import MultiModalPneumoniaModel
from src.train_e7 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config_e10.yaml"
ARMS = ("pre", "control", "feedback")
SOURCE_FILES = (
    "config_e10.yaml",
    "src/dataset.py",
    "src/model.py",
    "src/e10_protocol.py",
    "src/train_e10.py",
    "src/evaluate_e10.py",
    "src/utils.py",
)
FEEDBACK_COLUMNS = [
    "experiment",
    "schema_version",
    "feedback_source",
    "model_arm",
    "model_checkpoint_sha256",
    "image_id",
    "image_path",
    "patient_id",
    "original_split",
    "model_probability",
    "model_prediction",
    "reviewer_decision",
    "corrected_label",
    "reference_label",
]


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_e10_config(path=DEFAULT_CONFIG):
    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config, config_path


def validate_config(config):
    if config.get("experiment", {}).get("id") != "E10":
        raise ValueError("E10 configuration must declare experiment.id as E10.")
    if config.get("model", {}).get("backbone") != "densenet121":
        raise ValueError("E10 is restricted to DenseNet-121.")
    if config.get("model", {}).get("preprocessing") != "baseline":
        raise ValueError("E10 must isolate feedback under baseline preprocessing.")
    if Path(config.get("data", {}).get("image_root", "")).is_absolute():
        raise ValueError("E10 data.image_root must be repository-relative.")
    partition = config.get("partition", {})
    if partition.get("source_split") != "train":
        raise ValueError("E10 feedback must originate from the training split.")
    if float(partition.get("feedback_fraction", -1)) != 0.10:
        raise ValueError("E10 feedback_fraction is locked at 0.10.")
    if not partition.get("stratify_by_patient_positive"):
        raise ValueError("E10 requires patient-level positive-label stratification.")
    if int(config.get("pre_training", {}).get("epochs", -1)) != 5:
        raise ValueError("E10 pre-feedback training is locked at five epochs.")
    if int(config.get("adaptation_training", {}).get("epochs", -1)) != 3:
        raise ValueError("E10 continuation arms are locked at three epochs.")
    expected_prefixes = {
        "image_encoder.features.denseblock4",
        "image_encoder.features.norm5",
        "image_projection",
        "metadata_encoder",
        "classifier",
    }
    if set(config.get("adaptation_training", {}).get("trainable_prefixes", [])) != expected_prefixes:
        raise ValueError("E10 adaptation trainable prefixes differ from the approved scope.")
    selection = config.get("threshold_selection", {})
    if selection.get("split") != "val" or selection.get("metric") != "f1":
        raise ValueError("E10 thresholds must be selected on validation F1.")
    if selection.get("tie_break") != "lowest_threshold":
        raise ValueError("E10 threshold ties must select the lowest threshold.")
    grid = list(map(float, selection.get("grid", [])))
    if not grid or grid != sorted(set(grid)) or any(x <= 0 or x >= 1 for x in grid):
        raise ValueError("E10 threshold grid must be unique, ascending, and inside (0, 1).")
    bootstrap = config.get("bootstrap", {})
    if int(bootstrap.get("resamples", -1)) != 1000 or int(bootstrap.get("seed", -1)) != 42:
        raise ValueError("E10 bootstrap is locked at 1,000 resamples and seed 42.")
    outputs = config.get("output", {})
    required = {
        "partition_manifest", "partition_metadata", "pre_manifest", "control_manifest",
        "feedback_manifest", "pre_checkpoint", "pre_training_metrics", "pre_run_metadata",
        "control_checkpoint", "control_training_metrics", "control_run_metadata",
        "feedback_checkpoint", "feedback_training_metrics", "feedback_run_metadata",
        "pre_val_predictions", "pre_val_metrics", "pre_threshold_analysis", "pre_feedback_lock",
        "feedback_pool_predictions", "simulated_feedback", "feedback_lock",
        "control_val_predictions", "control_val_metrics", "control_threshold_analysis",
        "feedback_val_predictions", "feedback_val_metrics", "feedback_threshold_analysis",
        "validation_lock", "pre_test_predictions", "pre_test_metrics",
        "control_test_predictions", "control_test_metrics", "feedback_test_predictions",
        "feedback_test_metrics", "test_comparison", "bootstrap_intervals",
    }
    missing = sorted(required.difference(outputs))
    if missing:
        raise ValueError("E10 configuration is missing outputs: " + ", ".join(missing))
    paths = [Path(value).as_posix() for value in outputs.values()]
    if len(paths) != len(set(paths)):
        raise ValueError("E10 output paths must be unique.")
    if any(not value.startswith("outputs/") or "e10_" not in value for value in paths):
        raise ValueError("Every E10 output must be isolated below outputs/ under an e10_* name.")


def require_hash(path, expected_hash, label):
    resolved = resolve_repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    actual = sha256_file(resolved)
    expected = str(expected_hash).upper()
    if actual != expected:
        raise ValueError(f"{label} hash mismatch. Expected {expected}, found {actual}.")
    return resolved, actual


def validate_static_inputs(config):
    pairs = [
        (config["data"]["manifest"], config["data"]["manifest_sha256"], "E10 manifest"),
        (config["data"]["dataset_source"], config["data"]["dataset_source_sha256"], "dataset source"),
        (config["model"]["source"], config["model"]["source_sha256"], "model source"),
        (config["reference"]["e2_checkpoint"], config["reference"]["e2_checkpoint_sha256"], "E2 checkpoint"),
        (config["reference"]["e8_validation_predictions"], config["reference"]["e8_validation_predictions_sha256"], "E8 validation predictions"),
        (config["reference"]["e4_feedback"], config["reference"]["e4_feedback_sha256"], "E4 feedback"),
        (config["reference"]["e3_cases"], config["reference"]["e3_cases_sha256"], "E3 cases"),
    ]
    return {label: require_hash(path, digest, label) for path, digest, label in pairs}


def validate_manifest(config):
    path, digest = require_hash(config["data"]["manifest"], config["data"]["manifest_sha256"], "E10 manifest")
    frame = pd.read_csv(path)
    required = {"image_path", "Patient ID", "target", "age", "sex", "view", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("E10 manifest is missing columns: " + ", ".join(missing))
    frame = frame.copy()
    frame["image_id"] = frame["image_path"].map(lambda value: Path(str(value)).name)
    if frame["image_id"].duplicated().any():
        raise ValueError("E10 requires globally unique image filenames.")
    if int(frame.groupby("Patient ID")["split"].nunique().max()) != 1:
        raise ValueError("E10 manifest has patients in multiple original splits.")
    counts = {str(k): int(v) for k, v in frame.groupby("split").size().items()}
    positives = {str(k): int(g["target"].sum()) for k, g in frame.groupby("split")}
    expected_counts = {k: int(v) for k, v in config["data"]["expected_split_counts"].items()}
    expected_positive = {k: int(v) for k, v in config["data"]["expected_positive_counts"].items()}
    if counts != expected_counts or positives != expected_positive:
        raise ValueError(f"E10 manifest counts differ: counts={counts}, positives={positives}.")
    if set(frame["target"].astype(int).unique()) != {0, 1}:
        raise ValueError("E10 manifest targets must be binary.")
    return frame, path, digest


def validate_e4_is_test_only(config, manifest):
    feedback = pd.read_csv(resolve_repo_path(config["reference"]["e4_feedback"]))
    feedback_ids = feedback["image_path"].map(lambda value: Path(str(value)).name)
    matched = manifest[manifest["image_id"].isin(feedback_ids)]
    if len(feedback) != 4 or len(matched) != 4 or set(matched["split"]) != {"test"}:
        raise ValueError("Retained E4 feedback is not the approved four-case test-only evidence.")


def _sample_exact(frame, counts, seed):
    selected = []
    for target, count in sorted(counts.items()):
        candidates = frame[frame["target"].astype(int) == int(target)].sort_values("image_id")
        if len(candidates) < count:
            raise ValueError(f"Insufficient target {target} replay candidates.")
        selected.append(candidates.sample(n=count, random_state=seed + int(target)))
    return pd.concat(selected).sort_values("image_id").reset_index(drop=True)


def build_partitions(config, manifest):
    train = manifest[manifest["split"] == "train"].copy()
    patients = train.groupby("Patient ID").agg(has_positive=("target", "max")).reset_index()
    patients = patients.sort_values("Patient ID").reset_index(drop=True)
    base_ids, feedback_ids = train_test_split(
        patients["Patient ID"],
        test_size=float(config["partition"]["feedback_fraction"]),
        random_state=int(config["seed"]),
        stratify=patients["has_positive"],
    )
    base = train[train["Patient ID"].isin(set(base_ids))].copy()
    feedback = train[train["Patient ID"].isin(set(feedback_ids))].copy()
    target_counts = {int(k): int(v) for k, v in feedback.groupby("target").size().items()}
    common = _sample_exact(base, target_counts, int(config["seed"]))
    remaining = base[~base["image_id"].isin(common["image_id"])].copy()
    extra = _sample_exact(remaining, target_counts, int(config["seed"]) + 100)

    expected = config["partition"]["expected"]
    observed = {
        "base_train_patients": int(base["Patient ID"].nunique()),
        "base_train_images": len(base),
        "base_train_positives": int(base["target"].sum()),
        "feedback_patients": int(feedback["Patient ID"].nunique()),
        "feedback_images": len(feedback),
        "feedback_positives": int(feedback["target"].sum()),
        "replay_images_per_set": len(common),
        "replay_positives_per_set": int(common["target"].sum()),
    }
    if observed != {key: int(value) for key, value in expected.items()}:
        raise ValueError(f"E10 partition differs from the approved counts: {observed}.")
    sets = [set(base_ids), set(feedback_ids)]
    if not sets[0].isdisjoint(sets[1]) or not set(common["image_id"]).isdisjoint(set(extra["image_id"])):
        raise ValueError("E10 patient or replay partitions overlap.")

    partition = manifest.copy()
    partition["e10_role"] = partition["split"]
    partition.loc[partition["Patient ID"].isin(set(base_ids)), "e10_role"] = "base_train"
    partition.loc[partition["Patient ID"].isin(set(feedback_ids)), "e10_role"] = "feedback_pool"

    val_test = manifest[manifest["split"].isin(["val", "test"])].copy()
    pre_train = base.copy()
    control_train = pd.concat([common, extra], ignore_index=True)
    feedback_train = pd.concat([feedback, common], ignore_index=True)
    for frame in (pre_train, control_train, feedback_train):
        frame["split"] = "train"
    pre_manifest = pd.concat([pre_train, val_test], ignore_index=True)
    control_manifest = pd.concat([control_train, val_test], ignore_index=True)
    feedback_manifest = pd.concat([feedback_train, val_test], ignore_index=True)
    for frame in (partition, pre_manifest, control_manifest, feedback_manifest):
        if frame["image_id"].duplicated().any():
            raise ValueError("An E10 generated manifest contains duplicate image IDs.")
    return partition, pre_manifest, control_manifest, feedback_manifest, observed


def write_csv_atomic(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def require_outputs_absent(paths):
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing:
        listing = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError("E10 outputs already exist; retained evidence will not be overwritten:\n" + listing)


def source_hashes():
    hashes = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required E10 source is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def model_kwargs(config):
    model = config["model"]
    return {
        "metadata_dim": int(model["metadata_dim"]),
        "image_embedding_dim": int(model["image_embedding_dim"]),
        "metadata_embedding_dim": int(model["metadata_embedding_dim"]),
        "dropout": float(model["dropout"]),
    }


def build_model(config, pretrained):
    return MultiModalPneumoniaModel(pretrained=pretrained, **model_kwargs(config))


def load_checkpoint_model(config, checkpoint_path, expected_arm, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("experiment") != "E10" or checkpoint.get("arm") != expected_arm:
        raise ValueError(f"Checkpoint is not the E10 {expected_arm} arm.")
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def configure_adaptation_layers(model, prefixes):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad = True
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not any(name.startswith(prefix) for prefix in prefixes) for name in trainable):
        raise RuntimeError("E10 adaptation trainable-parameter contract failed.")
    return trainable


def make_dataset(config, manifest_path, split, augment):
    return NIHPneumoniaDataset(
        manifest=str(manifest_path),
        split=split,
        image_size=int(config["data"]["image_size"]),
        augment=augment,
        preprocessing="baseline",
        image_root=str(resolve_repo_path(config["data"]["image_root"])),
    )


def output_path(config, key):
    return resolve_repo_path(config["output"][key])


def validate_partition_outputs(config):
    metadata_path = output_path(config, "partition_metadata")
    if not metadata_path.is_file():
        raise FileNotFoundError("E10 partitions are missing. Run evaluate_e10 --stage prepare first.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("stage") != "partition_lock":
        raise ValueError("E10 partition metadata has an unexpected stage.")
    if metadata.get("source_manifest_sha256") != str(config["data"]["manifest_sha256"]).upper():
        raise ValueError("E10 partition metadata references a different source manifest.")
    if metadata.get("config_sha256") != sha256_file(config["_config_path"]):
        raise ValueError("E10 partition metadata references a different config revision.")
    if metadata.get("source_sha256") != source_hashes():
        raise ValueError("E10 executable sources changed after the partition lock was created.")
    expected_counts = {key: int(value) for key, value in config["partition"]["expected"].items()}
    if metadata.get("counts") != expected_counts:
        raise ValueError("E10 partition metadata counts differ from the approved protocol.")
    for key in ("partition_manifest", "pre_manifest", "control_manifest", "feedback_manifest"):
        path = output_path(config, key)
        if not path.is_file() or sha256_file(path) != metadata["output_sha256"][key]:
            raise ValueError(f"E10 partition evidence mismatch: {key}.")
    return metadata
