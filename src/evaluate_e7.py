import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import NIHPneumoniaDataset
from src.resnet_model import MultiModalResNet50
from src.train_e7 import (
    DEFAULT_CONFIG,
    calculate_metrics,
    load_e7_config,
    resolve_repo_path,
    sha256_file,
    utc_now,
    validate_manifest,
    validate_image_inventory,
    write_json,
)
from src.utils import get_device


PREDICTION_COLUMNS = [
    "experiment",
    "model",
    "split",
    "image_id",
    "target",
    "probability",
    "prediction",
    "threshold",
]


def build_model(config, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint.get("experiment") != "E7":
        raise ValueError("Checkpoint is not an E7 checkpoint.")

    model_config = config["model"]
    model = MultiModalResNet50(
        pretrained=False,
        metadata_dim=int(model_config["metadata_dim"]),
        image_embedding_dim=int(model_config["image_embedding_dim"]),
        metadata_embedding_dim=int(model_config["metadata_embedding_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint


def collect_probabilities(model, loader, device, split):
    targets = []
    probabilities = []
    image_ids = []

    model.eval()

    with torch.no_grad():
        for images, metadata, batch_targets, image_paths in loader:
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            logits = model(images, metadata)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()

            targets.extend(batch_targets.numpy().astype(int).tolist())
            probabilities.extend(batch_probabilities.astype(float).tolist())
            image_ids.extend(Path(path).name for path in image_paths)

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"E7 {split} predictions contain duplicate image identifiers.")

    return image_ids, targets, probabilities


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


def build_prediction_frame(
    image_ids,
    targets,
    probabilities,
    split,
    threshold,
):
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities_array >= float(threshold)).astype(int)

    frame = pd.DataFrame(
        {
            "experiment": "E7",
            "model": "Multi-modal ResNet-50 + metadata",
            "split": split,
            "image_id": image_ids,
            "target": np.asarray(targets, dtype=int),
            "probability": probabilities_array,
            "prediction": predictions,
            "threshold": float(threshold),
        }
    )

    if frame.columns.tolist() != PREDICTION_COLUMNS:
        raise RuntimeError("E7 prediction schema is not in the approved column order.")

    if frame["image_id"].duplicated().any():
        raise ValueError("E7 prediction frame contains duplicate image identifiers.")

    return frame


def build_metrics_record(
    targets,
    probabilities,
    split,
    threshold,
    config_hash,
    manifest_hash,
    checkpoint_hash,
):
    metrics = calculate_metrics(targets, probabilities, threshold)

    return {
        "experiment": "E7",
        "model": "Multi-modal ResNet-50 + metadata",
        "split": split,
        "n": int(len(targets)),
        "positive_n": int(np.sum(np.asarray(targets) == 1)),
        "negative_n": int(np.sum(np.asarray(targets) == 0)),
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        **metrics,
    }


def require_available_outputs(paths, overwrite):
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        joined = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "E7 evaluation outputs already exist. Do not overwrite retained evidence "
            "without a documented protocol amendment:\n" + joined
        )


def load_locked_threshold(
    path,
    config_hash,
    manifest_hash,
    checkpoint_hash,
):
    if not path.exists():
        raise FileNotFoundError(
            "Validation-selected threshold is missing. Run E7 validation evaluation "
            f"before test evaluation: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        locked = json.load(handle)

    expected = {
        "experiment": "E7",
        "selection_split": "val",
        "selection_metric": "f1",
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
    }

    mismatches = {
        key: (expected_value, locked.get(key))
        for key, expected_value in expected.items()
        if locked.get(key) != expected_value
    }

    if mismatches:
        details = "; ".join(
            f"{key}: expected {values[0]}, found {values[1]}"
            for key, values in mismatches.items()
        )
        raise ValueError("Locked E7 threshold does not match current artefacts: " + details)

    threshold = float(locked["threshold"])

    if not 0 < threshold < 1:
        raise ValueError(f"Locked E7 threshold is invalid: {threshold}")

    return threshold, locked


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate E7. Validation selects and locks the threshold; "
            "test evaluation consumes that lock."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the approved E7 YAML configuration.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["val", "test"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing E7 evaluation evidence only under a documented amendment.",
    )
    args = parser.parse_args()

    config, config_path = load_e7_config(args.config)
    manifest_frame, manifest_summary = validate_manifest(config)
    validate_image_inventory(config, manifest_frame)

    checkpoint_path = resolve_repo_path(config["output"]["checkpoint"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"E7 checkpoint not found. Complete manual training first: {checkpoint_path}"
        )

    config_hash = sha256_file(config_path)
    manifest_hash = manifest_summary["manifest_sha256"]
    checkpoint_hash = sha256_file(checkpoint_path)
    device = get_device()

    if args.split == "val":
        prediction_path = resolve_repo_path(config["output"]["val_predictions"])
        metrics_path = resolve_repo_path(config["output"]["val_metrics"])
        analysis_path = resolve_repo_path(config["output"]["threshold_analysis"])
        lock_path = resolve_repo_path(config["output"]["locked_threshold"])
        require_available_outputs(
            [prediction_path, metrics_path, analysis_path, lock_path],
            args.overwrite,
        )
        locked_threshold = None
    else:
        prediction_path = resolve_repo_path(config["output"]["test_predictions"])
        metrics_path = resolve_repo_path(config["output"]["test_metrics"])
        lock_path = resolve_repo_path(config["output"]["locked_threshold"])
        require_available_outputs(
            [prediction_path, metrics_path],
            args.overwrite,
        )
        locked_threshold, _ = load_locked_threshold(
            lock_path,
            config_hash,
            manifest_hash,
            checkpoint_hash,
        )

    model, checkpoint = build_model(config, checkpoint_path, device)

    checkpoint_expectations = {
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
    }

    for key, expected_value in checkpoint_expectations.items():
        if checkpoint.get(key) != expected_value:
            raise ValueError(
                f"E7 checkpoint {key} does not match current protocol. "
                f"Expected {expected_value}, found {checkpoint.get(key)}."
            )

    dataset = NIHPneumoniaDataset(
        str(resolve_repo_path(config["data"]["manifest"])),
        args.split,
        int(config["data"]["image_size"]),
        augment=False,
        preprocessing=config["data"]["preprocessing"],
        image_root=str(resolve_repo_path(config["data"]["image_root"])),
    )

    expected_count = int(config["data"]["expected_split_counts"][args.split])

    if len(dataset) != expected_count:
        raise ValueError(
            f"E7 {args.split} dataset expected {expected_count} rows, found {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    print("Device:", device)
    print("Experiment: E7 - Multi-modal ResNet-50 + metadata")
    print("Split:", args.split)
    print("Samples:", len(dataset))
    print("Checkpoint SHA-256:", checkpoint_hash)

    image_ids, targets, probabilities = collect_probabilities(
        model,
        loader,
        device,
        args.split,
    )

    if args.split == "val":
        analysis, selected = analyze_thresholds(
            targets,
            probabilities,
            config["threshold_selection"]["grid"],
        )
        threshold = float(selected["threshold"])

        prediction_frame = build_prediction_frame(
            image_ids,
            targets,
            probabilities,
            "val",
            threshold,
        )
        metrics_record = build_metrics_record(
            targets,
            probabilities,
            "val",
            threshold,
            config_hash,
            manifest_hash,
            checkpoint_hash,
        )

        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)

        prediction_frame.to_csv(prediction_path, index=False)
        pd.DataFrame([metrics_record]).to_csv(metrics_path, index=False)
        analysis.to_csv(analysis_path, index=False)

        lock_record = {
            "experiment": "E7",
            "model": "Multi-modal ResNet-50 + metadata",
            "selected_at_utc": utc_now(),
            "selection_split": "val",
            "selection_metric": "f1",
            "tie_break": "lowest_threshold",
            "threshold": threshold,
            "selected_metrics": selected,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
            "validation_predictions_sha256": sha256_file(prediction_path),
            "validation_metrics_sha256": sha256_file(metrics_path),
            "threshold_analysis_sha256": sha256_file(analysis_path),
        }
        write_json(lock_path, lock_record)

        print()
        print("Validation threshold locked:", threshold)
        print("Validation predictions:", prediction_path)
        print("Validation metrics:", metrics_path)
        print("Threshold analysis:", analysis_path)
        print("Threshold lock:", lock_path)

    else:
        threshold = locked_threshold

        prediction_frame = build_prediction_frame(
            image_ids,
            targets,
            probabilities,
            "test",
            threshold,
        )
        metrics_record = build_metrics_record(
            targets,
            probabilities,
            "test",
            threshold,
            config_hash,
            manifest_hash,
            checkpoint_hash,
        )

        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        prediction_frame.to_csv(prediction_path, index=False)
        pd.DataFrame([metrics_record]).to_csv(metrics_path, index=False)

        print()
        print("Locked validation threshold used:", threshold)
        print("Test predictions:", prediction_path)
        print("Test metrics:", metrics_path)

    print()
    print(pd.Series(metrics_record))


if __name__ == "__main__":
    main()

