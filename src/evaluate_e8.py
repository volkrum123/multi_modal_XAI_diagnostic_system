import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import NIHPneumoniaDataset
from src.model import MultiModalPneumoniaModel
from src.train_e7 import calculate_metrics, sha256_file, utc_now
from src.utils import get_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config_e8.yaml"

COMPONENT_COLUMNS = [
    "experiment",
    "model",
    "split",
    "image_id",
    "target",
    "probability",
    "prediction",
    "threshold",
]

ENSEMBLE_COLUMNS = [
    "experiment",
    "model",
    "split",
    "image_id",
    "target",
    "densenet_probability",
    "resnet_probability",
    "probability",
    "prediction",
    "threshold",
]

METRIC_COLUMNS = [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
]


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_e8_config(path=DEFAULT_CONFIG):
    config_path = resolve_repo_path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    validate_config(config)
    return config, config_path


def validate_config(config):
    if config.get("experiment", {}).get("id") != "E8":
        raise ValueError("E8 configuration must declare experiment.id as E8.")

    if config.get("data", {}).get("preprocessing") != "baseline":
        raise ValueError("E8 is restricted to baseline preprocessing; CLAHE is E9 scope.")

    image_root = config.get("data", {}).get("image_root")

    if not image_root or Path(image_root).is_absolute():
        raise ValueError("E8 data.image_root must be repository-relative.")

    fusion = config.get("fusion", {})

    if fusion.get("method") != "equal_weight_probability_average":
        raise ValueError("E8 fusion method must be equal_weight_probability_average.")

    densenet_weight = float(fusion.get("densenet_weight", -1))
    resnet_weight = float(fusion.get("resnet_weight", -1))

    if densenet_weight != 0.5 or resnet_weight != 0.5:
        raise ValueError("E8 component weights are pre-specified at 0.50 and 0.50.")

    threshold = config.get("threshold_selection", {})

    if threshold.get("split") != "val":
        raise ValueError("E8 threshold selection must use validation only.")

    if threshold.get("metric") != "f1":
        raise ValueError("E8 threshold selection metric must be f1.")

    if threshold.get("tie_break") != "lowest_threshold":
        raise ValueError("E8 threshold ties must select the lowest threshold.")

    grid = list(map(float, threshold.get("grid", [])))

    if not grid or grid != sorted(set(grid)):
        raise ValueError("E8 threshold grid must be non-empty, unique, and ascending.")

    if any(value <= 0 or value >= 1 for value in grid):
        raise ValueError("E8 threshold grid values must be strictly between 0 and 1.")

    if float(config.get("densenet", {}).get("selected_threshold", -1)) != 0.75:
        raise ValueError("E8 must preserve the validation-selected E2 threshold of 0.75.")

    if float(config.get("resnet", {}).get("selected_threshold", -1)) != 0.85:
        raise ValueError("E8 must preserve the validation-selected E7 threshold of 0.85.")

    required_outputs = {
        "densenet_val_predictions",
        "densenet_val_metrics",
        "ensemble_val_predictions",
        "ensemble_val_metrics",
        "ensemble_val_threshold_analysis",
        "locked_threshold",
        "densenet_test_predictions",
        "densenet_test_metrics",
        "ensemble_test_predictions",
        "ensemble_test_metrics",
        "model_comparison",
    }
    outputs = config.get("output", {})
    missing_outputs = sorted(required_outputs.difference(outputs))

    if missing_outputs:
        raise ValueError("E8 configuration is missing outputs: " + ", ".join(missing_outputs))

    output_values = [Path(value).as_posix() for value in outputs.values()]

    if len(output_values) != len(set(output_values)):
        raise ValueError("E8 output paths must be unique.")

    if any(not path.startswith("outputs/") or "e8_" not in path for path in output_values):
        raise ValueError("Every E8 output must be an isolated outputs/.../e8_* path.")

    input_paths = {
        Path(config["densenet"]["checkpoint"]).as_posix(),
        Path(config["densenet"]["validation_threshold_analysis"]).as_posix(),
        Path(config["resnet"]["locked_threshold"]).as_posix(),
        Path(config["resnet"]["val_predictions"]).as_posix(),
        Path(config["resnet"]["test_predictions"]).as_posix(),
    }

    collisions = input_paths.intersection(output_values)

    if collisions:
        raise ValueError("E8 outputs collide with retained inputs: " + ", ".join(collisions))


def require_hash(path, expected_hash, label):
    path = resolve_repo_path(path)

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    actual_hash = sha256_file(path)
    expected_hash = str(expected_hash).upper()

    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} hash mismatch. Expected {expected_hash}, found {actual_hash}."
        )

    return path, actual_hash


def validate_manifest(config):
    manifest_path, manifest_hash = require_hash(
        config["data"]["manifest"],
        config["data"]["manifest_sha256"],
        "E8 manifest",
    )
    frame = pd.read_csv(manifest_path)
    required_columns = {
        "image_path",
        "Patient ID",
        "target",
        "age",
        "sex",
        "view",
        "split",
    }
    missing = sorted(required_columns.difference(frame.columns))

    if missing:
        raise ValueError("E8 manifest is missing columns: " + ", ".join(missing))

    image_ids = frame["image_path"].map(lambda value: Path(str(value)).name)

    if image_ids.duplicated().any():
        raise ValueError("E8 requires globally unique image filenames.")

    if int((frame.groupby("Patient ID")["split"].nunique() > 1).sum()) != 0:
        raise ValueError("E8 manifest contains patients assigned to multiple splits.")

    expected_counts = {
        key: int(value)
        for key, value in config["data"]["expected_split_counts"].items()
    }
    actual_counts = {
        str(key): int(value)
        for key, value in frame.groupby("split").size().items()
    }

    if actual_counts != expected_counts:
        raise ValueError(
            f"E8 split counts differ. Expected {expected_counts}, found {actual_counts}."
        )

    expected_positive = {
        key: int(value)
        for key, value in config["data"]["expected_positive_counts"].items()
    }
    actual_positive = {
        str(key): int(group["target"].sum())
        for key, group in frame.groupby("split")
    }

    if actual_positive != expected_positive:
        raise ValueError(
            "E8 positive counts differ. "
            f"Expected {expected_positive}, found {actual_positive}."
        )

    return frame, manifest_path, manifest_hash


def validate_static_inputs(config, stage):
    paths = {}
    hashes = {}

    specifications = [
        (
            "densenet_checkpoint",
            config["densenet"]["checkpoint"],
            config["densenet"]["checkpoint_sha256"],
            "E2 checkpoint",
        ),
        (
            "densenet_model_source",
            config["densenet"]["model_source"],
            config["densenet"]["model_source_sha256"],
            "E2 model source",
        ),
        (
            "dataset_source",
            config["integrity"]["dataset_source"],
            config["integrity"]["dataset_source_sha256"],
            "dataset source",
        ),
        (
            "e2_validation_analysis",
            config["densenet"]["validation_threshold_analysis"],
            config["densenet"]["validation_threshold_analysis_sha256"],
            "E2 validation threshold analysis",
        ),
        (
            "e7_threshold_lock",
            config["resnet"]["locked_threshold"],
            config["resnet"]["locked_threshold_sha256"],
            "E7 threshold lock",
        ),
        (
            "e7_predictions",
            config["resnet"][f"{stage}_predictions"],
            config["resnet"][f"{stage}_predictions_sha256"],
            f"E7 {stage} predictions",
        ),
    ]

    for key, value, expected_hash, label in specifications:
        path, actual_hash = require_hash(value, expected_hash, label)
        paths[key] = path
        hashes[key] = actual_hash

    with paths["e7_threshold_lock"].open("r", encoding="utf-8") as handle:
        e7_lock = json.load(handle)

    expected_lock_values = {
        "experiment": "E7",
        "selection_split": "val",
        "selection_metric": "f1",
        "tie_break": "lowest_threshold",
        "threshold": float(config["resnet"]["selected_threshold"]),
        "config_sha256": str(config["resnet"]["config_sha256"]).upper(),
        "manifest_sha256": str(config["data"]["manifest_sha256"]).upper(),
        "checkpoint_sha256": str(config["resnet"]["checkpoint_sha256"]).upper(),
        "validation_predictions_sha256": str(
            config["resnet"]["val_predictions_sha256"]
        ).upper(),
    }

    for key, expected in expected_lock_values.items():
        if e7_lock.get(key) != expected:
            raise ValueError(
                f"E7 threshold lock {key} mismatch. "
                f"Expected {expected}, found {e7_lock.get(key)}."
            )

    return paths, hashes, e7_lock


def load_densenet_model(config, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint.get("experiment") != "E2":
        raise ValueError("Configured DenseNet checkpoint is not an E2 checkpoint.")

    state = checkpoint.get("model_state")

    if not isinstance(state, dict) or not state:
        raise ValueError("E2 checkpoint does not contain a model state.")

    nonfinite = sum(
        int((~torch.isfinite(tensor)).sum().item())
        for tensor in state.values()
        if torch.is_tensor(tensor)
    )

    if nonfinite:
        raise ValueError(f"E2 checkpoint contains {nonfinite} non-finite values.")

    model_config = config["densenet"]
    model = MultiModalPneumoniaModel(
        pretrained=False,
        metadata_dim=int(model_config["metadata_dim"]),
        image_embedding_dim=int(model_config["image_embedding_dim"]),
        metadata_embedding_dim=int(model_config["metadata_embedding_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    expected_auc = float(model_config["best_validation_roc_auc"])
    actual_auc = float(checkpoint.get("best_val_auc", float("nan")))
    tolerance = float(config["integrity"]["numeric_tolerance"])

    if not np.isclose(actual_auc, expected_auc, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"E2 checkpoint best validation ROC-AUC differs: {actual_auc}."
        )

    return model, checkpoint


def collect_densenet_probabilities(model, loader, device, split):
    image_ids = []
    targets = []
    probabilities = []

    with torch.no_grad():
        for images, metadata, batch_targets, image_paths in tqdm(
            loader,
            desc=f"E8 DenseNet {split}",
        ):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            logits = model(images, metadata)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()

            image_ids.extend(Path(path).name for path in image_paths)
            targets.extend(batch_targets.numpy().astype(int).tolist())
            probabilities.extend(batch_probabilities.astype(float).tolist())

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"E2 {split} inference produced duplicate image IDs.")

    if not np.isfinite(np.asarray(probabilities, dtype=np.float64)).all():
        raise ValueError(f"E2 {split} inference produced non-finite probabilities.")

    return image_ids, targets, probabilities


def build_component_frame(
    experiment,
    model,
    split,
    image_ids,
    targets,
    probabilities,
    threshold,
):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "experiment": experiment,
            "model": model,
            "split": split,
            "image_id": image_ids,
            "target": np.asarray(targets, dtype=np.int64),
            "probability": probabilities,
            "prediction": (probabilities >= float(threshold)).astype(np.int64),
            "threshold": float(threshold),
        }
    )

    validate_component_frame(frame, experiment, split, threshold)
    return frame


def validate_component_frame(frame, experiment, split, threshold):
    if frame.columns.tolist() != COMPONENT_COLUMNS:
        raise ValueError("Component prediction columns do not match the E8 contract.")

    if frame["image_id"].duplicated().any():
        raise ValueError(f"{experiment} {split} predictions contain duplicate image IDs.")

    if set(frame["experiment"].astype(str).unique()) != {experiment}:
        raise ValueError(f"Component predictions do not consistently identify {experiment}.")

    if set(frame["split"].astype(str).unique()) != {split}:
        raise ValueError(f"{experiment} predictions do not consistently identify {split}.")

    probabilities = pd.to_numeric(frame["probability"], errors="coerce").to_numpy()

    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{experiment} {split} probabilities are invalid.")

    targets = pd.to_numeric(frame["target"], errors="coerce").to_numpy()

    if not np.isfinite(targets).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError(f"{experiment} {split} targets are invalid.")

    thresholds = pd.to_numeric(frame["threshold"], errors="coerce").to_numpy()

    if not np.allclose(thresholds, float(threshold), rtol=0.0, atol=0.0):
        raise ValueError(f"{experiment} {split} threshold differs from the lock.")

    expected_predictions = (probabilities >= float(threshold)).astype(int)
    actual_predictions = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy()

    if not np.array_equal(actual_predictions.astype(int), expected_predictions):
        raise ValueError(f"{experiment} {split} prediction labels do not match probabilities.")


def load_resnet_predictions(path, split, threshold, expected_count):
    frame = pd.read_csv(path)
    validate_component_frame(frame, "E7", split, threshold)

    if len(frame) != int(expected_count):
        raise ValueError(
            f"E7 {split} prediction count differs. "
            f"Expected {expected_count}, found {len(frame)}."
        )

    return frame


def align_components(densenet_frame, resnet_frame, split):
    densenet_ids = set(densenet_frame["image_id"])
    resnet_ids = set(resnet_frame["image_id"])

    if densenet_ids != resnet_ids:
        missing_resnet = sorted(densenet_ids - resnet_ids)
        missing_densenet = sorted(resnet_ids - densenet_ids)
        raise ValueError(
            f"E8 {split} component image IDs do not match. "
            f"Missing from E7: {missing_resnet[:5]}; "
            f"missing from E2: {missing_densenet[:5]}."
        )

    left = densenet_frame[["image_id", "target", "probability"]].rename(
        columns={
            "target": "densenet_target",
            "probability": "densenet_probability",
        }
    )
    right = resnet_frame[["image_id", "target", "probability"]].rename(
        columns={
            "target": "resnet_target",
            "probability": "resnet_probability",
        }
    )
    aligned = left.merge(right, on="image_id", how="inner", validate="one_to_one")

    if not np.array_equal(
        aligned["densenet_target"].to_numpy(dtype=int),
        aligned["resnet_target"].to_numpy(dtype=int),
    ):
        raise ValueError(f"E8 {split} component targets do not match by image ID.")

    return aligned


def build_ensemble_frame(aligned, split, threshold, densenet_weight, resnet_weight):
    densenet_probabilities = aligned["densenet_probability"].to_numpy(dtype=float)
    resnet_probabilities = aligned["resnet_probability"].to_numpy(dtype=float)
    probabilities = (
        float(densenet_weight) * densenet_probabilities
        + float(resnet_weight) * resnet_probabilities
    )
    frame = pd.DataFrame(
        {
            "experiment": "E8",
            "model": "DenseNet-121 + ResNet-50 equal-weight ensemble",
            "split": split,
            "image_id": aligned["image_id"].astype(str),
            "target": aligned["densenet_target"].to_numpy(dtype=int),
            "densenet_probability": densenet_probabilities,
            "resnet_probability": resnet_probabilities,
            "probability": probabilities,
            "prediction": (probabilities >= float(threshold)).astype(int),
            "threshold": float(threshold),
        }
    )

    if frame.columns.tolist() != ENSEMBLE_COLUMNS:
        raise RuntimeError("Ensemble prediction columns do not match the E8 contract.")

    expected = (densenet_probabilities + resnet_probabilities) / 2.0

    if not np.allclose(probabilities, expected, rtol=0.0, atol=1e-15):
        raise ValueError("E8 probabilities are not the equal-weight arithmetic mean.")

    return frame


def analyze_thresholds(targets, probabilities, thresholds):
    rows = [
        calculate_metrics(targets, probabilities, float(threshold))
        for threshold in thresholds
    ]
    frame = pd.DataFrame(rows)
    maximum_f1 = frame["f1"].max()
    selected_row = frame.loc[frame["f1"] == maximum_f1].iloc[0]
    selected = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in selected_row.to_dict().items()
    }
    return frame, selected


def validate_e2_validation_reproduction(config, analysis):
    historical = pd.read_csv(
        resolve_repo_path(config["densenet"]["validation_threshold_analysis"])
    )
    tolerance = float(config["integrity"]["numeric_tolerance"])

    if historical["threshold"].tolist() != analysis["threshold"].tolist():
        raise ValueError("Regenerated E2 validation threshold grid differs from history.")

    integer_columns = ["tn", "fp", "fn", "tp"]
    float_columns = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
    ]

    for column in integer_columns:
        if not np.array_equal(
            historical[column].to_numpy(dtype=int),
            analysis[column].to_numpy(dtype=int),
        ):
            raise ValueError(f"Regenerated E2 validation {column} differs from history.")

    for column in float_columns:
        if not np.allclose(
            historical[column].to_numpy(dtype=float),
            analysis[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(f"Regenerated E2 validation {column} differs from history.")

    selected_threshold = float(
        analysis.loc[analysis["f1"] == analysis["f1"].max(), "threshold"].iloc[0]
    )
    expected_threshold = float(config["densenet"]["selected_threshold"])

    if selected_threshold != expected_threshold:
        raise ValueError(
            f"Regenerated E2 validation selects {selected_threshold}, "
            f"not the retained {expected_threshold}."
        )


def build_metrics_record(experiment, model, split, targets, probabilities, threshold):
    targets_array = np.asarray(targets, dtype=int)
    return {
        "experiment": experiment,
        "model": model,
        "split": split,
        "n": int(len(targets_array)),
        "positive_n": int(np.sum(targets_array == 1)),
        "negative_n": int(np.sum(targets_array == 0)),
        **calculate_metrics(targets_array, probabilities, threshold),
    }


def validate_e2_test_reproduction(config, metrics):
    expected = config["densenet"]["expected_test_metrics"]
    tolerance = float(config["integrity"]["numeric_tolerance"])

    for key in ["tn", "fp", "fn", "tp"]:
        if int(metrics[key]) != int(expected[key]):
            raise ValueError(
                f"Regenerated E2 test {key} differs. "
                f"Expected {expected[key]}, found {metrics[key]}."
            )

    for key in [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        if not np.isclose(
            float(metrics[key]),
            float(expected[key]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                f"Regenerated E2 test {key} differs. "
                f"Expected {expected[key]}, found {metrics[key]}."
            )


def require_available_outputs(paths):
    existing = [path for path in paths if path.exists()]

    if existing:
        joined = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "E8 outputs already exist. Do not overwrite retained evidence without "
            "a documented protocol amendment:\n" + joined
        )


def write_csv_atomic(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    temporary.replace(path)


def validate_lock_record(record, config, config_hash, val_paths):
    expected = {
        "experiment": "E8",
        "selection_split": "val",
        "selection_metric": "f1",
        "tie_break": "lowest_threshold",
        "fusion_method": "equal_weight_probability_average",
        "densenet_weight": 0.5,
        "resnet_weight": 0.5,
        "config_sha256": config_hash,
        "manifest_sha256": str(config["data"]["manifest_sha256"]).upper(),
        "densenet_checkpoint_sha256": str(
            config["densenet"]["checkpoint_sha256"]
        ).upper(),
        "resnet_checkpoint_sha256": str(
            config["resnet"]["checkpoint_sha256"]
        ).upper(),
        "resnet_validation_predictions_sha256": str(
            config["resnet"]["val_predictions_sha256"]
        ).upper(),
    }

    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise ValueError(
                f"E8 threshold lock {key} mismatch. "
                f"Expected {expected_value}, found {record.get(key)}."
            )

    threshold = float(record.get("threshold", float("nan")))

    if not 0 < threshold < 1:
        raise ValueError(f"E8 locked threshold is invalid: {threshold}.")

    hash_fields = {
        "densenet_validation_predictions_sha256": val_paths[
            "densenet_val_predictions"
        ],
        "densenet_validation_metrics_sha256": val_paths[
            "densenet_val_metrics"
        ],
        "ensemble_validation_predictions_sha256": val_paths[
            "ensemble_val_predictions"
        ],
        "ensemble_validation_metrics_sha256": val_paths["ensemble_val_metrics"],
        "ensemble_threshold_analysis_sha256": val_paths[
            "ensemble_val_threshold_analysis"
        ],
    }

    for key, path in hash_fields.items():
        if not path.is_file():
            raise FileNotFoundError(f"E8 validation evidence missing: {path}")

        actual_hash = sha256_file(path)

        if record.get(key) != actual_hash:
            raise ValueError(
                f"E8 threshold lock {key} does not match {path}."
            )

    return threshold


def load_e8_lock(config, config_hash, val_paths):
    lock_path = resolve_repo_path(config["output"]["locked_threshold"])

    if not lock_path.is_file():
        raise FileNotFoundError(
            "E8 validation threshold lock is missing. Complete and review the "
            f"validation stage before test evaluation: {lock_path}"
        )

    with lock_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    threshold = validate_lock_record(record, config, config_hash, val_paths)
    return threshold, record


def make_dataloader(config, split):
    dataset = NIHPneumoniaDataset(
        str(resolve_repo_path(config["data"]["manifest"])),
        split,
        int(config["data"]["image_size"]),
        augment=False,
        preprocessing=config["data"]["preprocessing"],
        image_root=str(resolve_repo_path(config["data"]["image_root"])),
    )
    expected_count = int(config["data"]["expected_split_counts"][split])

    if len(dataset) != expected_count:
        raise ValueError(
            f"E8 {split} dataset expected {expected_count} rows, found {len(dataset)}."
        )

    device = get_device()
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    return dataset, loader, device


def run_stage(config, config_path, stage):
    _, _, manifest_hash = validate_manifest(config)
    static_paths, static_hashes, _ = validate_static_inputs(config, stage)
    config_hash = sha256_file(config_path)
    output = {
        key: resolve_repo_path(value)
        for key, value in config["output"].items()
    }

    if stage == "val":
        stage_output_keys = [
            "densenet_val_predictions",
            "densenet_val_metrics",
            "ensemble_val_predictions",
            "ensemble_val_metrics",
            "ensemble_val_threshold_analysis",
            "locked_threshold",
        ]
        locked_threshold = None
    else:
        stage_output_keys = [
            "densenet_test_predictions",
            "densenet_test_metrics",
            "ensemble_test_predictions",
            "ensemble_test_metrics",
            "model_comparison",
        ]
        val_paths = {
            key: output[key]
            for key in [
                "densenet_val_predictions",
                "densenet_val_metrics",
                "ensemble_val_predictions",
                "ensemble_val_metrics",
                "ensemble_val_threshold_analysis",
            ]
        }
        locked_threshold, _ = load_e8_lock(config, config_hash, val_paths)

    require_available_outputs([output[key] for key in stage_output_keys])

    dataset, loader, device = make_dataloader(config, stage)
    model, _ = load_densenet_model(
        config,
        static_paths["densenet_checkpoint"],
        device,
    )

    print("Device:", device)
    print("Experiment: E8 - DenseNet-121 + ResNet-50 ensemble")
    print("Stage:", stage)
    print("Samples:", len(dataset))
    print("Config SHA-256:", config_hash)
    print("Manifest SHA-256:", manifest_hash)
    print("E2 checkpoint SHA-256:", static_hashes["densenet_checkpoint"])
    print("E7 predictions SHA-256:", static_hashes["e7_predictions"])

    image_ids, targets, probabilities = collect_densenet_probabilities(
        model,
        loader,
        device,
        stage,
    )
    densenet_threshold = float(config["densenet"]["selected_threshold"])
    densenet_frame = build_component_frame(
        "E2",
        config["densenet"]["model_name"],
        stage,
        image_ids,
        targets,
        probabilities,
        densenet_threshold,
    )
    densenet_metrics = build_metrics_record(
        "E2",
        config["densenet"]["model_name"],
        stage,
        targets,
        probabilities,
        densenet_threshold,
    )

    resnet_threshold = float(config["resnet"]["selected_threshold"])
    resnet_frame = load_resnet_predictions(
        static_paths["e7_predictions"],
        stage,
        resnet_threshold,
        config["data"]["expected_split_counts"][stage],
    )
    aligned = align_components(densenet_frame, resnet_frame, stage)

    if stage == "val":
        e2_analysis, _ = analyze_thresholds(
            targets,
            probabilities,
            config["threshold_selection"]["grid"],
        )
        write_csv_atomic(output["densenet_val_predictions"], densenet_frame)
        write_csv_atomic(
            output["densenet_val_metrics"], pd.DataFrame([densenet_metrics])
        )
        validate_e2_validation_reproduction(config, e2_analysis)

        provisional = build_ensemble_frame(
            aligned,
            stage,
            0.5,
            config["fusion"]["densenet_weight"],
            config["fusion"]["resnet_weight"],
        )
        ensemble_analysis, selected = analyze_thresholds(
            provisional["target"],
            provisional["probability"],
            config["threshold_selection"]["grid"],
        )
        ensemble_threshold = float(selected["threshold"])
        ensemble_frame = build_ensemble_frame(
            aligned,
            stage,
            ensemble_threshold,
            config["fusion"]["densenet_weight"],
            config["fusion"]["resnet_weight"],
        )
        ensemble_metrics = build_metrics_record(
            "E8",
            "DenseNet-121 + ResNet-50 equal-weight ensemble",
            stage,
            ensemble_frame["target"],
            ensemble_frame["probability"],
            ensemble_threshold,
        )

        write_csv_atomic(output["ensemble_val_predictions"], ensemble_frame)
        write_csv_atomic(
            output["ensemble_val_metrics"], pd.DataFrame([ensemble_metrics])
        )
        write_csv_atomic(
            output["ensemble_val_threshold_analysis"], ensemble_analysis
        )

        lock_record = {
            "experiment": "E8",
            "model": "DenseNet-121 + ResNet-50 equal-weight ensemble",
            "selected_at_utc": utc_now(),
            "selection_split": "val",
            "selection_metric": "f1",
            "tie_break": "lowest_threshold",
            "fusion_method": config["fusion"]["method"],
            "densenet_weight": float(config["fusion"]["densenet_weight"]),
            "resnet_weight": float(config["fusion"]["resnet_weight"]),
            "threshold": ensemble_threshold,
            "selected_metrics": selected,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "densenet_checkpoint_sha256": static_hashes["densenet_checkpoint"],
            "resnet_checkpoint_sha256": str(
                config["resnet"]["checkpoint_sha256"]
            ).upper(),
            "resnet_validation_predictions_sha256": static_hashes[
                "e7_predictions"
            ],
            "densenet_validation_predictions_sha256": sha256_file(
                output["densenet_val_predictions"]
            ),
            "densenet_validation_metrics_sha256": sha256_file(
                output["densenet_val_metrics"]
            ),
            "ensemble_validation_predictions_sha256": sha256_file(
                output["ensemble_val_predictions"]
            ),
            "ensemble_validation_metrics_sha256": sha256_file(
                output["ensemble_val_metrics"]
            ),
            "ensemble_threshold_analysis_sha256": sha256_file(
                output["ensemble_val_threshold_analysis"]
            ),
        }
        write_json_atomic(output["locked_threshold"], lock_record)

        print()
        print("E2 validation reproduction passed.")
        print("E8 validation threshold locked:", ensemble_threshold)
        print("E8 threshold lock:", output["locked_threshold"])
        print()
        print(pd.Series(ensemble_metrics))

    else:
        write_csv_atomic(output["densenet_test_predictions"], densenet_frame)
        write_csv_atomic(
            output["densenet_test_metrics"], pd.DataFrame([densenet_metrics])
        )
        validate_e2_test_reproduction(config, densenet_metrics)
        ensemble_frame = build_ensemble_frame(
            aligned,
            stage,
            locked_threshold,
            config["fusion"]["densenet_weight"],
            config["fusion"]["resnet_weight"],
        )
        ensemble_metrics = build_metrics_record(
            "E8",
            "DenseNet-121 + ResNet-50 equal-weight ensemble",
            stage,
            ensemble_frame["target"],
            ensemble_frame["probability"],
            locked_threshold,
        )
        resnet_metrics = build_metrics_record(
            "E7",
            config["resnet"]["model_name"],
            stage,
            resnet_frame["target"],
            resnet_frame["probability"],
            resnet_threshold,
        )
        comparison = pd.DataFrame(
            [densenet_metrics, resnet_metrics, ensemble_metrics]
        )

        write_csv_atomic(output["ensemble_test_predictions"], ensemble_frame)
        write_csv_atomic(
            output["ensemble_test_metrics"], pd.DataFrame([ensemble_metrics])
        )
        write_csv_atomic(output["model_comparison"], comparison)

        print()
        print("E2 historical test reproduction passed.")
        print("Locked E8 validation threshold used:", locked_threshold)
        print("Model comparison:", output["model_comparison"])
        print()
        print(comparison.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate E8. Validation regenerates E2 probabilities and locks the "
            "equal-weight ensemble threshold; test consumes that lock once."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the approved E8 YAML configuration.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["val", "test"],
    )
    args = parser.parse_args()

    config, config_path = load_e8_config(args.config)
    run_stage(config, config_path, args.stage)


if __name__ == "__main__":
    main()

