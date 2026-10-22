import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.e9_dataset import E9RobustnessDataset
from src.model import MultiModalPneumoniaModel
from src.resnet_model import MultiModalResNet50
from src.train_e7 import calculate_metrics, sha256_file, utc_now
from src.train_e9 import (
    DEFAULT_CONFIG,
    load_e9_config,
    require_hash,
    resolve_repo_path,
    validate_manifest,
)
from src.utils import get_device


SYSTEMS = [
    "baseline_densenet",
    "clahe_densenet",
    "baseline_resnet",
    "clahe_resnet",
    "baseline_ensemble",
    "clahe_ensemble",
]
COMPONENTS = [
    "baseline_densenet",
    "clahe_densenet",
    "baseline_resnet",
    "clahe_resnet",
]
PERTURBED_CONDITIONS = [
    "brightness_low",
    "brightness_high",
    "contrast_low",
    "contrast_high",
    "gamma_low",
    "gamma_high",
    "gaussian_noise",
    "gaussian_blur",
]


def require_available_outputs(paths):
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing:
        listing = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "E9 outputs already exist. Do not overwrite evidence without a documented "
            "protocol amendment:\n" + listing
        )


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


def write_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_static_inputs(config, include_e9_checkpoint=True):
    specifications = [
        (
            "baseline_densenet",
            config["models"]["baseline_densenet"]["checkpoint"],
            config["models"]["baseline_densenet"]["checkpoint_sha256"],
            "E2 checkpoint",
        ),
        (
            "duplicate_densenet",
            config["models"]["baseline_densenet"]["duplicate_checkpoint"],
            config["models"]["baseline_densenet"]["duplicate_checkpoint_sha256"],
            "E6-A duplicate checkpoint",
        ),
        (
            "clahe_densenet",
            config["models"]["clahe_densenet"]["checkpoint"],
            config["models"]["clahe_densenet"]["checkpoint_sha256"],
            "E6-B CLAHE DenseNet checkpoint",
        ),
        (
            "clahe_densenet_metrics",
            config["models"]["clahe_densenet"]["training_metrics"],
            config["models"]["clahe_densenet"]["training_metrics_sha256"],
            "E6-B training metrics",
        ),
        (
            "baseline_resnet",
            config["models"]["baseline_resnet"]["checkpoint"],
            config["models"]["baseline_resnet"]["checkpoint_sha256"],
            "E7 checkpoint",
        ),
        (
            "e7_lock",
            config["models"]["baseline_resnet"]["threshold_lock"],
            config["models"]["baseline_resnet"]["threshold_lock_sha256"],
            "E7 threshold lock",
        ),
        (
            "e8_lock",
            config["models"]["baseline_ensemble"]["threshold_lock"],
            config["models"]["baseline_ensemble"]["threshold_lock_sha256"],
            "E8 threshold lock",
        ),
        (
            "densenet_source",
            config["model_architecture"]["densenet_source"],
            config["model_architecture"]["densenet_source_sha256"],
            "DenseNet source",
        ),
        (
            "resnet_source",
            config["model_architecture"]["resnet_source"],
            config["model_architecture"]["resnet_source_sha256"],
            "ResNet source",
        ),
        (
            "dataset_source",
            config["data"]["dataset_source"],
            config["data"]["dataset_source_sha256"],
            "dataset source",
        ),
    ]
    paths = {}
    hashes = {}

    for key, value, expected, label in specifications:
        paths[key], hashes[key] = require_hash(value, expected, label)

    if include_e9_checkpoint:
        e9_path = resolve_repo_path(config["models"]["clahe_resnet"]["checkpoint"])
        if not e9_path.is_file():
            raise FileNotFoundError(
                "E9 CLAHE ResNet checkpoint is missing. Complete training first: "
                f"{e9_path}"
            )
        paths["clahe_resnet"] = e9_path
        hashes["clahe_resnet"] = sha256_file(e9_path)

    return paths, hashes


def verify_duplicate_densenet_states(first_path, duplicate_path):
    first = torch.load(first_path, map_location="cpu")["model_state"]
    duplicate = torch.load(duplicate_path, map_location="cpu")["model_state"]

    if list(first) != list(duplicate):
        raise ValueError("E2 and E6-A checkpoint keys differ.")

    for key in first:
        if not torch.equal(first[key], duplicate[key]):
            raise ValueError(f"E2 and E6-A are not exact duplicates at tensor {key}.")


def validate_retained_locks(config, paths):
    with paths["e7_lock"].open("r", encoding="utf-8") as handle:
        e7 = json.load(handle)
    with paths["e8_lock"].open("r", encoding="utf-8") as handle:
        e8 = json.load(handle)

    expected_e7 = {
        "experiment": "E7",
        "threshold": float(config["models"]["baseline_resnet"]["threshold"]),
        "manifest_sha256": str(config["data"]["manifest_sha256"]).upper(),
        "checkpoint_sha256": str(
            config["models"]["baseline_resnet"]["checkpoint_sha256"]
        ).upper(),
    }
    expected_e8 = {
        "experiment": "E8",
        "threshold": float(config["models"]["baseline_ensemble"]["threshold"]),
        "manifest_sha256": str(config["data"]["manifest_sha256"]).upper(),
        "densenet_checkpoint_sha256": str(
            config["models"]["baseline_densenet"]["checkpoint_sha256"]
        ).upper(),
        "resnet_checkpoint_sha256": str(
            config["models"]["baseline_resnet"]["checkpoint_sha256"]
        ).upper(),
    }

    for key, expected in expected_e7.items():
        if e7.get(key) != expected:
            raise ValueError(f"Retained E7 lock {key} mismatch.")
    for key, expected in expected_e8.items():
        if e8.get(key) != expected:
            raise ValueError(f"Retained E8 lock {key} mismatch.")


def architecture_kwargs(config):
    architecture = config["model_architecture"]
    return {
        "metadata_dim": int(architecture["metadata_dim"]),
        "image_embedding_dim": int(architecture["image_embedding_dim"]),
        "metadata_embedding_dim": int(architecture["metadata_embedding_dim"]),
        "dropout": float(architecture["dropout"]),
    }


def load_model(config, architecture, checkpoint_path, expected_experiment, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint.get("experiment") != expected_experiment:
        raise ValueError(
            f"Expected {expected_experiment} checkpoint, found {checkpoint.get('experiment')}."
        )

    if expected_experiment == "E9":
        if checkpoint.get("arm") != "clahe_resnet50":
            raise ValueError("E9 checkpoint is not the approved CLAHE ResNet-50 arm.")
        if checkpoint.get("manifest_sha256") != str(
            config["data"]["manifest_sha256"]
        ).upper():
            raise ValueError("E9 checkpoint manifest hash differs from the protocol.")
        if checkpoint.get("preprocessing") != config["preprocessing"]["clahe"]:
            raise ValueError("E9 checkpoint CLAHE configuration differs from the protocol.")

    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{expected_experiment} checkpoint has no model_state.")

    nonfinite = sum(
        int((~torch.isfinite(tensor)).sum().item())
        for tensor in state.values()
        if torch.is_tensor(tensor)
    )
    if nonfinite:
        raise ValueError(f"{expected_experiment} checkpoint contains non-finite values.")

    cls = MultiModalPneumoniaModel if architecture == "densenet" else MultiModalResNet50
    model = cls(pretrained=False, **architecture_kwargs(config)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def validate_e9_training_record(config, config_path, checkpoint_hash):
    metadata_path = resolve_repo_path(config["output"]["run_metadata"])
    if not metadata_path.is_file():
        raise FileNotFoundError(f"E9 training run metadata is missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    expected = {
        "experiment": "E9",
        "arm": "clahe_resnet50",
        "status": "complete",
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": str(config["data"]["manifest_sha256"]).upper(),
        "checkpoint_sha256": checkpoint_hash,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"E9 training record {key} mismatch.")

    for relative, expected_hash in record.get("source_sha256", {}).items():
        source_path = resolve_repo_path(relative)
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise ValueError(f"E9 training source changed after training: {source_path}")

    return metadata_path, sha256_file(metadata_path)


def forward_with_features(model, images, metadata):
    image_features = model.image_encoder(images)
    image_features = model.image_projection(image_features)
    metadata_features = model.metadata_encoder(metadata)
    fused = torch.cat([image_features, metadata_features], dim=1)
    logits = model.classifier(fused).squeeze(1)
    return logits, image_features


def make_loader(config, split, preprocessing, condition):
    dataset = E9RobustnessDataset(
        manifest=str(resolve_repo_path(config["data"]["manifest"])),
        split=split,
        image_size=int(config["data"]["image_size"]),
        augment=False,
        preprocessing=preprocessing,
        image_root=str(resolve_repo_path(config["data"]["image_root"])),
        condition=condition,
        perturbations=config["perturbations"],
        seed=int(config["seed"]),
    )
    expected = int(config["data"]["expected_split_counts"][split])
    if len(dataset) != expected:
        raise ValueError(f"E9 {split} expected {expected} samples, found {len(dataset)}.")
    device = get_device()
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    return dataset, loader, device


def collect_pair(first, second, loader, device, description):
    image_ids = []
    targets = []
    first_probabilities = []
    second_probabilities = []
    first_features = []
    second_features = []

    first.eval()
    second.eval()
    with torch.no_grad():
        for images, metadata, batch_targets, image_paths in tqdm(loader, desc=description):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            first_logits, first_batch_features = forward_with_features(first, images, metadata)
            second_logits, second_batch_features = forward_with_features(second, images, metadata)
            image_ids.extend(Path(path).name for path in image_paths)
            targets.extend(batch_targets.numpy().astype(int).tolist())
            first_probabilities.extend(torch.sigmoid(first_logits).cpu().numpy().tolist())
            second_probabilities.extend(torch.sigmoid(second_logits).cpu().numpy().tolist())
            first_features.append(first_batch_features.cpu().numpy().astype(np.float32))
            second_features.append(second_batch_features.cpu().numpy().astype(np.float32))

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{description} produced duplicate image IDs.")

    return {
        "image_id": np.asarray(image_ids, dtype=str),
        "target": np.asarray(targets, dtype=np.int64),
        "first_probability": np.asarray(first_probabilities, dtype=np.float64),
        "second_probability": np.asarray(second_probabilities, dtype=np.float64),
        "first_feature": np.concatenate(first_features, axis=0),
        "second_feature": np.concatenate(second_features, axis=0),
    }


def align_runs(baseline, clahe):
    baseline_order = np.argsort(baseline["image_id"])
    clahe_order = np.argsort(clahe["image_id"])
    aligned = {}

    if not np.array_equal(
        baseline["image_id"][baseline_order], clahe["image_id"][clahe_order]
    ):
        raise ValueError("Baseline and CLAHE image IDs do not match.")
    if not np.array_equal(
        baseline["target"][baseline_order], clahe["target"][clahe_order]
    ):
        raise ValueError("Baseline and CLAHE targets do not match.")

    for key, value in baseline.items():
        aligned[f"baseline_{key}"] = value[baseline_order]
    for key, value in clahe.items():
        aligned[f"clahe_{key}"] = value[clahe_order]
    return aligned


def analyze_thresholds(targets, probabilities, thresholds):
    frame = pd.DataFrame(
        [calculate_metrics(targets, probabilities, float(value)) for value in thresholds]
    )
    best_f1 = frame["f1"].max()
    selected = frame.loc[frame["f1"] == best_f1].iloc[0].to_dict()
    selected = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in selected.items()
    }
    return frame, selected


def build_prediction_frame(experiment, model, split, ids, targets, probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "experiment": experiment,
            "model": model,
            "split": split,
            "image_id": ids,
            "target": np.asarray(targets, dtype=int),
            "probability": probabilities,
            "prediction": (probabilities >= float(threshold)).astype(int),
            "threshold": float(threshold),
        }
    )


def metric_record(system, condition, targets, probabilities, threshold):
    preprocessing, architecture = system.split("_", 1)
    return {
        "experiment": "E9",
        "system": system,
        "preprocessing": preprocessing,
        "architecture": architecture,
        "condition": condition,
        "split": "test",
        "n": len(targets),
        "positive_n": int(np.sum(np.asarray(targets) == 1)),
        "negative_n": int(np.sum(np.asarray(targets) == 0)),
        **calculate_metrics(targets, probabilities, threshold),
    }


def validate_historical_e6b(config, probabilities, targets):
    expected_auc = float(config["models"]["clahe_densenet"]["expected_best_validation_roc_auc"])
    checkpoint_metrics = calculate_metrics(
        targets,
        probabilities,
        float(config["models"]["clahe_densenet"]["historical_monitor_threshold"]),
    )
    expected = config["models"]["clahe_densenet"]["expected_historical_validation"]
    tolerance = float(config["integrity"]["numeric_tolerance"])

    if not np.isclose(checkpoint_metrics["roc_auc"], expected_auc, rtol=0.0, atol=tolerance):
        raise ValueError("Regenerated E6-B validation ROC-AUC differs from retained history.")

    for key in ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]:
        if not np.isclose(checkpoint_metrics[key], float(expected[key]), rtol=0.0, atol=tolerance):
            raise ValueError(f"Regenerated E6-B validation {key} differs from history.")
    for key in ["tn", "fp", "fn", "tp"]:
        if checkpoint_metrics[key] != int(expected[key]):
            raise ValueError(f"Regenerated E6-B validation {key} differs from history.")


def run_validation(config, config_path):
    _, _, manifest_hash = validate_manifest(config)
    paths, hashes = validate_static_inputs(config, include_e9_checkpoint=True)
    training_metadata_path, training_metadata_hash = validate_e9_training_record(
        config, config_path, hashes["clahe_resnet"]
    )
    verify_duplicate_densenet_states(paths["baseline_densenet"], paths["duplicate_densenet"])
    validate_retained_locks(config, paths)
    output = {key: resolve_repo_path(value) for key, value in config["output"].items()}
    keys = [
        "densenet_val_predictions",
        "densenet_val_metrics",
        "densenet_val_threshold_analysis",
        "resnet_val_predictions",
        "resnet_val_metrics",
        "resnet_val_threshold_analysis",
        "ensemble_val_predictions",
        "ensemble_val_metrics",
        "ensemble_val_threshold_analysis",
        "validation_lock",
    ]
    require_available_outputs([output[key] for key in keys])
    _, loader, device = make_loader(config, "val", "clahe", "clean")
    densenet, _ = load_model(config, "densenet", paths["clahe_densenet"], "E6-B", device)
    resnet, resnet_checkpoint = load_model(
        config, "resnet", paths["clahe_resnet"], "E9", device
    )
    result = collect_pair(densenet, resnet, loader, device, "E9 CLAHE validation")
    ids = result["image_id"]
    targets = result["target"]
    dense_probability = result["first_probability"]
    res_probability = result["second_probability"]
    ensemble_probability = (
        float(config["fusion"]["densenet_weight"]) * dense_probability
        + float(config["fusion"]["resnet_weight"]) * res_probability
    )
    validate_historical_e6b(config, dense_probability, targets)
    grid = config["threshold_selection"]["grid"]
    analyses = {}
    selected = {}
    probabilities = {
        "densenet": dense_probability,
        "resnet": res_probability,
        "ensemble": ensemble_probability,
    }

    for name, values in probabilities.items():
        analyses[name], selected[name] = analyze_thresholds(targets, values, grid)

    frames = {
        "densenet": build_prediction_frame(
            "E6-B",
            config["models"]["clahe_densenet"]["model_name"],
            "val",
            ids,
            targets,
            dense_probability,
            selected["densenet"]["threshold"],
        ),
        "resnet": build_prediction_frame(
            "E9",
            config["models"]["clahe_resnet"]["model_name"],
            "val",
            ids,
            targets,
            res_probability,
            selected["resnet"]["threshold"],
        ),
        "ensemble": build_prediction_frame(
            "E9",
            "CLAHE DenseNet-121 + ResNet-50 equal-weight ensemble",
            "val",
            ids,
            targets,
            ensemble_probability,
            selected["ensemble"]["threshold"],
        ),
    }

    for name in ["densenet", "resnet", "ensemble"]:
        prediction_path = output[f"{name}_val_predictions"]
        metric_path = output[f"{name}_val_metrics"]
        analysis_path = output[f"{name}_val_threshold_analysis"]
        write_csv_atomic(prediction_path, frames[name])
        write_csv_atomic(metric_path, pd.DataFrame([selected[name]]))
        write_csv_atomic(analysis_path, analyses[name])

    lock = {
        "experiment": "E9",
        "stage": "clahe_validation_threshold_lock",
        "selected_at_utc": utc_now(),
        "selection_split": "val",
        "selection_metric": "f1",
        "tie_break": "lowest_threshold",
        "fusion_method": config["fusion"]["method"],
        "thresholds": {name: float(selected[name]["threshold"]) for name in selected},
        "selected_metrics": selected,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest_hash,
        "input_sha256": hashes,
        "training_run_metadata": str(training_metadata_path),
        "training_run_metadata_sha256": training_metadata_hash,
        "evidence_sha256": {
            f"{name}_{kind}": sha256_file(output[f"{name}_val_{kind}"])
            for name in ["densenet", "resnet", "ensemble"]
            for kind in ["predictions", "metrics", "threshold_analysis"]
        },
    }
    write_json_atomic(output["validation_lock"], lock)
    print("E6-B historical validation reproduction passed.")
    print("E9 CLAHE thresholds locked:", lock["thresholds"])
    print("Validation lock:", output["validation_lock"])


def load_validation_lock(config, config_path, static_hashes=None):
    lock_path = resolve_repo_path(config["output"]["validation_lock"])
    if not lock_path.is_file():
        raise FileNotFoundError("E9 validation lock missing. Run --stage val first.")
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    expected = {
        "experiment": "E9",
        "stage": "clahe_validation_threshold_lock",
        "selection_split": "val",
        "selection_metric": "f1",
        "tie_break": "lowest_threshold",
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": str(config["data"]["manifest_sha256"]).upper(),
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"E9 validation lock {key} mismatch.")
    required = {"densenet", "resnet", "ensemble"}
    if set(lock.get("thresholds", {})) != required:
        raise ValueError("E9 validation lock does not contain all CLAHE thresholds.")
    if static_hashes is not None and lock.get("input_sha256") != static_hashes:
        raise ValueError("E9 validation lock input hashes differ from current model evidence.")
    for name, kind_hash in lock.get("evidence_sha256", {}).items():
        system, kind = name.split("_", 1)
        path = resolve_repo_path(config["output"][f"{system}_val_{kind}"])
        if not path.is_file() or sha256_file(path) != kind_hash:
            raise ValueError(f"E9 validation evidence changed: {path}")
    return lock, lock_path


def thresholds_from_locks(config, clahe_lock):
    return {
        "baseline_densenet": float(config["models"]["baseline_densenet"]["threshold"]),
        "clahe_densenet": float(clahe_lock["thresholds"]["densenet"]),
        "baseline_resnet": float(config["models"]["baseline_resnet"]["threshold"]),
        "clahe_resnet": float(clahe_lock["thresholds"]["resnet"]),
        "baseline_ensemble": float(config["models"]["baseline_ensemble"]["threshold"]),
        "clahe_ensemble": float(clahe_lock["thresholds"]["ensemble"]),
    }


def condition_paths(config, condition):
    root = resolve_repo_path(config["output"]["condition_root"]) / condition
    return {
        "root": root,
        "predictions": root / "predictions.csv",
        "metrics": root / "metrics.csv",
        "features": root / "feature_stability.csv",
        "metadata": root / "metadata.json",
    }


def cosine_and_drift(clean, perturbed, epsilon):
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if clean.shape != perturbed.shape or clean.ndim != 2:
        raise ValueError("Feature matrices must have matching two-dimensional shapes.")
    difference = perturbed - clean
    clean_norm = np.linalg.norm(clean, axis=1)
    perturbed_norm = np.linalg.norm(perturbed, axis=1)
    cosine = np.sum(clean * perturbed, axis=1) / np.maximum(
        clean_norm * perturbed_norm, float(epsilon)
    )
    normalized_drift = np.linalg.norm(difference, axis=1) / np.maximum(
        clean_norm, float(epsilon)
    )
    return cosine, normalized_drift


def verify_baseline_reference(config, ids, targets, probability_map):
    tolerance = float(config["integrity"]["numeric_tolerance"])
    mapping = {
        "baseline_densenet": config["models"]["baseline_densenet"],
        "baseline_resnet": config["models"]["baseline_resnet"],
        "baseline_ensemble": config["models"]["baseline_ensemble"],
    }
    order = np.argsort(ids)
    sorted_ids = ids[order]
    sorted_targets = targets[order]

    for system, model_config in mapping.items():
        path, _ = require_hash(
            model_config["test_predictions"],
            model_config["test_predictions_sha256"],
            f"retained {system} test predictions",
        )
        frame = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)
        if not np.array_equal(frame["image_id"].astype(str).to_numpy(), sorted_ids):
            raise ValueError(f"Regenerated {system} image IDs differ from retained E8 evidence.")
        if not np.array_equal(frame["target"].to_numpy(dtype=int), sorted_targets):
            raise ValueError(f"Regenerated {system} targets differ from retained E8 evidence.")
        if not np.allclose(
            frame["probability"].to_numpy(dtype=float),
            probability_map[system][order],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(f"Regenerated {system} probabilities differ from retained evidence.")


def run_condition(config, config_path, condition):
    if condition not in config["perturbations"]:
        raise ValueError(f"Unknown E9 condition: {condition}")
    _, _, manifest_hash = validate_manifest(config)
    paths, hashes = validate_static_inputs(config, include_e9_checkpoint=True)
    validate_e9_training_record(config, config_path, hashes["clahe_resnet"])
    verify_duplicate_densenet_states(paths["baseline_densenet"], paths["duplicate_densenet"])
    validate_retained_locks(config, paths)
    clahe_lock, clahe_lock_path = load_validation_lock(config, config_path, hashes)
    outputs = condition_paths(config, condition)
    require_available_outputs(
        [outputs["predictions"], outputs["metrics"], outputs["features"], outputs["metadata"]]
    )
    if condition == "clean":
        require_available_outputs(
            [
                resolve_repo_path(config["output"]["clean_features"]),
                resolve_repo_path(config["output"]["clean_reference_lock"]),
            ]
        )

    _, baseline_loader, device = make_loader(config, "test", "baseline", condition)
    _, clahe_loader, _ = make_loader(config, "test", "clahe", condition)
    baseline_dense, _ = load_model(
        config, "densenet", paths["baseline_densenet"], "E2", device
    )
    clahe_dense, _ = load_model(config, "densenet", paths["clahe_densenet"], "E6-B", device)
    baseline_res, _ = load_model(config, "resnet", paths["baseline_resnet"], "E7", device)
    clahe_res, _ = load_model(config, "resnet", paths["clahe_resnet"], "E9", device)
    baseline = collect_pair(
        baseline_dense,
        baseline_res,
        baseline_loader,
        device,
        f"E9 baseline {condition}",
    )
    clahe = collect_pair(
        clahe_dense,
        clahe_res,
        clahe_loader,
        device,
        f"E9 CLAHE {condition}",
    )
    aligned = align_runs(baseline, clahe)
    ids = aligned["baseline_image_id"]
    targets = aligned["baseline_target"]
    probability_map = {
        "baseline_densenet": aligned["baseline_first_probability"],
        "clahe_densenet": aligned["clahe_first_probability"],
        "baseline_resnet": aligned["baseline_second_probability"],
        "clahe_resnet": aligned["clahe_second_probability"],
    }
    probability_map["baseline_ensemble"] = (
        probability_map["baseline_densenet"] + probability_map["baseline_resnet"]
    ) / 2.0
    probability_map["clahe_ensemble"] = (
        probability_map["clahe_densenet"] + probability_map["clahe_resnet"]
    ) / 2.0
    thresholds = thresholds_from_locks(config, clahe_lock)

    if condition == "clean":
        verify_baseline_reference(config, ids, targets, probability_map)

    prediction_frame = pd.DataFrame({"image_id": ids, "target": targets})
    for system in SYSTEMS:
        prediction_frame[f"{system}_probability"] = probability_map[system]
        prediction_frame[f"{system}_prediction"] = (
            probability_map[system] >= thresholds[system]
        ).astype(int)
        prediction_frame[f"{system}_threshold"] = thresholds[system]

    metric_frame = pd.DataFrame(
        [
            metric_record(system, condition, targets, probability_map[system], thresholds[system])
            for system in SYSTEMS
        ]
    )
    current_features = {
        "baseline_densenet": aligned["baseline_first_feature"],
        "clahe_densenet": aligned["clahe_first_feature"],
        "baseline_resnet": aligned["baseline_second_feature"],
        "clahe_resnet": aligned["clahe_second_feature"],
    }
    clean_feature_path = resolve_repo_path(config["output"]["clean_features"])
    clean_lock_path = resolve_repo_path(config["output"]["clean_reference_lock"])

    if condition == "clean":
        feature_values = {
            f"{system}_{metric}": np.ones(len(ids)) if metric == "cosine" else np.zeros(len(ids))
            for system in COMPONENTS
            for metric in ["cosine", "normalized_l2_drift"]
        }
    else:
        if not clean_feature_path.is_file() or not clean_lock_path.is_file():
            raise FileNotFoundError("E9 clean reference is missing. Run the clean condition first.")
        with clean_lock_path.open("r", encoding="utf-8") as handle:
            clean_lock = json.load(handle)
        expected_clean_lock = {
            "experiment": "E9",
            "condition": "clean",
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": manifest_hash,
            "validation_lock_sha256": sha256_file(clahe_lock_path),
            "clean_features_sha256": sha256_file(clean_feature_path),
            "baseline_reproduction": "passed",
        }
        for key, value in expected_clean_lock.items():
            if clean_lock.get(key) != value:
                raise ValueError(f"E9 clean reference lock {key} mismatch.")
        clean = np.load(clean_feature_path, allow_pickle=False)
        if not np.array_equal(clean["image_id"].astype(str), ids):
            raise ValueError("E9 clean and perturbed feature IDs do not match.")
        feature_values = {}
        epsilon = float(config["feature_stability"]["epsilon"])
        for system in COMPONENTS:
            cosine, drift = cosine_and_drift(clean[system], current_features[system], epsilon)
            feature_values[f"{system}_cosine"] = cosine
            feature_values[f"{system}_normalized_l2_drift"] = drift

    feature_frame = pd.DataFrame({"image_id": ids, "target": targets})
    for key, values in feature_values.items():
        feature_frame[key] = values
    feature_frame["baseline_ensemble_cosine"] = (
        feature_frame["baseline_densenet_cosine"]
        + feature_frame["baseline_resnet_cosine"]
    ) / 2.0
    feature_frame["clahe_ensemble_cosine"] = (
        feature_frame["clahe_densenet_cosine"] + feature_frame["clahe_resnet_cosine"]
    ) / 2.0
    feature_frame["baseline_ensemble_normalized_l2_drift"] = (
        feature_frame["baseline_densenet_normalized_l2_drift"]
        + feature_frame["baseline_resnet_normalized_l2_drift"]
    ) / 2.0
    feature_frame["clahe_ensemble_normalized_l2_drift"] = (
        feature_frame["clahe_densenet_normalized_l2_drift"]
        + feature_frame["clahe_resnet_normalized_l2_drift"]
    ) / 2.0

    write_csv_atomic(outputs["predictions"], prediction_frame)
    write_csv_atomic(outputs["metrics"], metric_frame)
    write_csv_atomic(outputs["features"], feature_frame)

    if condition == "clean":
        write_npz_atomic(
            clean_feature_path,
            image_id=ids,
            target=targets,
            **current_features,
        )
        clean_lock = {
            "experiment": "E9",
            "condition": "clean",
            "created_at_utc": utc_now(),
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": manifest_hash,
            "validation_lock_sha256": sha256_file(clahe_lock_path),
            "clean_features_sha256": sha256_file(clean_feature_path),
            "baseline_reproduction": "passed",
            "feature_layer": config["feature_stability"]["layer"],
            "feature_dimension": int(config["feature_stability"]["dimension"]),
        }
        write_json_atomic(clean_lock_path, clean_lock)

    metadata = {
        "experiment": "E9",
        "condition": condition,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest_hash,
        "input_sha256": hashes,
        "validation_lock_sha256": sha256_file(clahe_lock_path),
        "prediction_sha256": sha256_file(outputs["predictions"]),
        "metrics_sha256": sha256_file(outputs["metrics"]),
        "features_sha256": sha256_file(outputs["features"]),
        "samples": len(ids),
        "thresholds": thresholds,
    }
    write_json_atomic(outputs["metadata"], metadata)
    print(f"E9 condition complete: {condition}")
    print(metric_frame[["system", "roc_auc", "pr_auc", "f1", "sensitivity", "specificity"]].to_string(index=False))
    print("Condition evidence:", outputs["root"])


def validate_condition_evidence(config, config_path, condition):
    paths = condition_paths(config, condition)
    for key in ["predictions", "metrics", "features", "metadata"]:
        if not paths[key].is_file():
            raise FileNotFoundError(f"E9 {condition} evidence missing: {paths[key]}")
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "experiment": "E9",
        "condition": condition,
        "status": "complete",
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": str(config["data"]["manifest_sha256"]).upper(),
        "prediction_sha256": sha256_file(paths["predictions"]),
        "metrics_sha256": sha256_file(paths["metrics"]),
        "features_sha256": sha256_file(paths["features"]),
        "validation_lock_sha256": sha256_file(
            resolve_repo_path(config["output"]["validation_lock"])
        ),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"E9 {condition} metadata {key} mismatch.")
    return paths


def safe_auc(targets, probabilities):
    if len(np.unique(targets)) < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(targets, probabilities))


def summarize_point_estimates(config, config_path):
    evidence = {
        condition: validate_condition_evidence(config, config_path, condition)
        for condition in ["clean", *PERTURBED_CONDITIONS]
    }
    clean_metrics = pd.read_csv(evidence["clean"]["metrics"]).set_index("system")
    rows = []

    for condition in PERTURBED_CONDITIONS:
        condition_metrics = pd.read_csv(evidence[condition]["metrics"]).set_index("system")
        for system in SYSTEMS:
            row = condition_metrics.loc[system].to_dict()
            row.update(
                {
                    "record_type": "condition",
                    "system": system,
                    "condition": condition,
                    "roc_auc_delta_from_clean": row["roc_auc"] - clean_metrics.loc[system, "roc_auc"],
                    "pr_auc_delta_from_clean": row["pr_auc"] - clean_metrics.loc[system, "pr_auc"],
                    "f1_delta_from_clean": row["f1"] - clean_metrics.loc[system, "f1"],
                    "sensitivity_delta_from_clean": row["sensitivity"] - clean_metrics.loc[system, "sensitivity"],
                    "specificity_delta_from_clean": row["specificity"] - clean_metrics.loc[system, "specificity"],
                    "accuracy_delta_from_clean": row["accuracy"] - clean_metrics.loc[system, "accuracy"],
                }
            )
            rows.append(row)

    condition_frame = pd.DataFrame(rows)
    mean_rows = []
    numeric = [
        "roc_auc_delta_from_clean",
        "pr_auc_delta_from_clean",
        "f1_delta_from_clean",
        "sensitivity_delta_from_clean",
        "specificity_delta_from_clean",
        "accuracy_delta_from_clean",
    ]
    for system in SYSTEMS:
        subset = condition_frame.loc[condition_frame["system"] == system]
        record = {
            "record_type": "mean_across_8_perturbations",
            "system": system,
            "condition": "mean_8_perturbations",
        }
        record.update({column: float(subset[column].mean()) for column in numeric})
        mean_rows.append(record)

    for architecture in ["densenet", "resnet", "ensemble"]:
        baseline = next(row for row in mean_rows if row["system"] == f"baseline_{architecture}")
        clahe = next(row for row in mean_rows if row["system"] == f"clahe_{architecture}")
        contrast = {
            "record_type": "clahe_minus_baseline_robustness_contrast",
            "system": f"clahe_minus_baseline_{architecture}",
            "condition": "mean_8_perturbations",
        }
        contrast.update({column: clahe[column] - baseline[column] for column in numeric})
        mean_rows.append(contrast)

    feature_rows = []
    for condition in PERTURBED_CONDITIONS:
        frame = pd.read_csv(evidence[condition]["features"])
        for system in SYSTEMS:
            feature_rows.append(
                {
                    "record_type": "feature_condition",
                    "system": system,
                    "condition": condition,
                    "feature_cosine_mean": float(frame[f"{system}_cosine"].mean()),
                    "feature_normalized_l2_drift_mean": float(
                        frame[f"{system}_normalized_l2_drift"].mean()
                    ),
                    "feature_note": (
                        "mean component proxy" if system.endswith("ensemble") else "post-ReLU image_projection"
                    ),
                }
            )
    feature_frame = pd.DataFrame(feature_rows)
    feature_means = []
    for system in SYSTEMS:
        subset = feature_frame.loc[feature_frame["system"] == system]
        feature_means.append(
            {
                "record_type": "feature_mean_across_8_perturbations",
                "system": system,
                "condition": "mean_8_perturbations",
                "feature_cosine_mean": float(subset["feature_cosine_mean"].mean()),
                "feature_normalized_l2_drift_mean": float(
                    subset["feature_normalized_l2_drift_mean"].mean()
                ),
                "feature_note": subset["feature_note"].iloc[0],
            }
        )

    baseline_architecture_cosine = float(
        np.mean(
            [
                next(x for x in feature_means if x["system"] == "baseline_densenet")[
                    "feature_cosine_mean"
                ],
                next(x for x in feature_means if x["system"] == "baseline_resnet")[
                    "feature_cosine_mean"
                ],
            ]
        )
    )
    clahe_architecture_cosine = float(
        np.mean(
            [
                next(x for x in feature_means if x["system"] == "clahe_densenet")[
                    "feature_cosine_mean"
                ],
                next(x for x in feature_means if x["system"] == "clahe_resnet")[
                    "feature_cosine_mean"
                ],
            ]
        )
    )
    baseline_architecture_drift = float(
        np.mean(
            [
                next(x for x in feature_means if x["system"] == "baseline_densenet")[
                    "feature_normalized_l2_drift_mean"
                ],
                next(x for x in feature_means if x["system"] == "baseline_resnet")[
                    "feature_normalized_l2_drift_mean"
                ],
            ]
        )
    )
    clahe_architecture_drift = float(
        np.mean(
            [
                next(x for x in feature_means if x["system"] == "clahe_densenet")[
                    "feature_normalized_l2_drift_mean"
                ],
                next(x for x in feature_means if x["system"] == "clahe_resnet")[
                    "feature_normalized_l2_drift_mean"
                ],
            ]
        )
    )
    feature_means.extend(
        [
            {
                "record_type": "primary_feature_endpoint",
                "system": "baseline_mean_across_densenet_and_resnet",
                "condition": "mean_8_perturbations",
                "feature_cosine_mean": baseline_architecture_cosine,
                "feature_normalized_l2_drift_mean": baseline_architecture_drift,
                "feature_note": "mean across the two learned component representations",
            },
            {
                "record_type": "primary_feature_endpoint",
                "system": "clahe_mean_across_densenet_and_resnet",
                "condition": "mean_8_perturbations",
                "feature_cosine_mean": clahe_architecture_cosine,
                "feature_normalized_l2_drift_mean": clahe_architecture_drift,
                "feature_note": "mean across the two learned component representations",
            },
            {
                "record_type": "primary_feature_endpoint_contrast",
                "system": "clahe_minus_baseline_mean_across_densenet_and_resnet",
                "condition": "mean_8_perturbations",
                "feature_cosine_mean": clahe_architecture_cosine - baseline_architecture_cosine,
                "feature_normalized_l2_drift_mean": clahe_architecture_drift - baseline_architecture_drift,
                "feature_note": "positive cosine contrast and negative drift contrast favour CLAHE",
            },
        ]
    )
    for architecture in ["densenet", "resnet", "ensemble"]:
        baseline = next(x for x in feature_means if x["system"] == f"baseline_{architecture}")
        clahe = next(x for x in feature_means if x["system"] == f"clahe_{architecture}")
        feature_means.append(
            {
                "record_type": "clahe_minus_baseline_feature_contrast",
                "system": f"clahe_minus_baseline_{architecture}",
                "condition": "mean_8_perturbations",
                "feature_cosine_mean": clahe["feature_cosine_mean"] - baseline["feature_cosine_mean"],
                "feature_normalized_l2_drift_mean": (
                    clahe["feature_normalized_l2_drift_mean"]
                    - baseline["feature_normalized_l2_drift_mean"]
                ),
                "feature_note": "positive cosine contrast favours CLAHE; negative drift contrast favours CLAHE",
            }
        )
    summary = pd.concat(
        [condition_frame, pd.DataFrame(mean_rows), feature_frame, pd.DataFrame(feature_means)],
        ignore_index=True,
        sort=False,
    )
    return summary, evidence


def stratified_indices(targets, rng):
    targets = np.asarray(targets, dtype=int)
    negative = np.flatnonzero(targets == 0)
    positive = np.flatnonzero(targets == 1)
    return np.concatenate(
        [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
    )


def percentile_interval(values, confidence):
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1.0 - alpha))


def bootstrap_intervals(config, evidence):
    clean_predictions = pd.read_csv(evidence["clean"]["predictions"]).sort_values("image_id")
    ids = clean_predictions["image_id"].astype(str).to_numpy()
    targets = clean_predictions["target"].to_numpy(dtype=int)
    condition_predictions = {}
    condition_features = {}

    for condition in PERTURBED_CONDITIONS:
        predictions = pd.read_csv(evidence[condition]["predictions"]).sort_values("image_id")
        features = pd.read_csv(evidence[condition]["features"]).sort_values("image_id")
        if not np.array_equal(predictions["image_id"].astype(str).to_numpy(), ids):
            raise ValueError(f"E9 {condition} prediction IDs differ during bootstrap.")
        if not np.array_equal(features["image_id"].astype(str).to_numpy(), ids):
            raise ValueError(f"E9 {condition} feature IDs differ during bootstrap.")
        condition_predictions[condition] = predictions
        condition_features[condition] = features

    resamples = int(config["bootstrap"]["resamples"])
    confidence = float(config["bootstrap"]["confidence_level"])
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    values = {}

    for system in SYSTEMS:
        values[(system, "mean_roc_auc_delta")] = []
        values[(system, "mean_pr_auc_delta")] = []
        values[(system, "mean_f1_delta")] = []
        values[(system, "mean_feature_cosine")] = []
        values[(system, "mean_feature_normalized_l2_drift")] = []

    thresholds = {
        system: float(clean_predictions[f"{system}_threshold"].iloc[0]) for system in SYSTEMS
    }

    for _ in tqdm(range(resamples), desc="E9 stratified bootstrap"):
        sample = stratified_indices(targets, rng)
        sample_targets = targets[sample]
        for system in SYSTEMS:
            clean_probability = clean_predictions[f"{system}_probability"].to_numpy()[sample]
            clean_metrics = calculate_metrics(sample_targets, clean_probability, thresholds[system])
            roc_deltas = []
            pr_deltas = []
            f1_deltas = []
            cosines = []
            drifts = []
            for condition in PERTURBED_CONDITIONS:
                probability = condition_predictions[condition][f"{system}_probability"].to_numpy()[sample]
                metrics = calculate_metrics(sample_targets, probability, thresholds[system])
                roc_deltas.append(metrics["roc_auc"] - clean_metrics["roc_auc"])
                pr_deltas.append(metrics["pr_auc"] - clean_metrics["pr_auc"])
                f1_deltas.append(metrics["f1"] - clean_metrics["f1"])
                cosines.append(
                    float(condition_features[condition][f"{system}_cosine"].to_numpy()[sample].mean())
                )
                drifts.append(
                    float(
                        condition_features[condition][
                            f"{system}_normalized_l2_drift"
                        ].to_numpy()[sample].mean()
                    )
                )
            values[(system, "mean_roc_auc_delta")].append(float(np.mean(roc_deltas)))
            values[(system, "mean_pr_auc_delta")].append(float(np.mean(pr_deltas)))
            values[(system, "mean_f1_delta")].append(float(np.mean(f1_deltas)))
            values[(system, "mean_feature_cosine")].append(float(np.mean(cosines)))
            values[(system, "mean_feature_normalized_l2_drift")].append(float(np.mean(drifts)))

    rows = []
    for (system, endpoint), endpoint_values in values.items():
        lower, upper = percentile_interval(endpoint_values, confidence)
        rows.append(
            {
                "experiment": "E9",
                "system": system,
                "endpoint": endpoint,
                "estimate": float(np.mean(endpoint_values)),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence,
                "resamples": resamples,
                "seed": int(config["bootstrap"]["seed"]),
                "stratified": True,
            }
        )

    # Paired CLAHE-minus-baseline bootstrap contrasts preserve each resample pairing.
    for architecture in ["densenet", "resnet", "ensemble"]:
        for endpoint in [
            "mean_roc_auc_delta",
            "mean_pr_auc_delta",
            "mean_f1_delta",
            "mean_feature_cosine",
            "mean_feature_normalized_l2_drift",
        ]:
            contrast = np.asarray(values[(f"clahe_{architecture}", endpoint)]) - np.asarray(
                values[(f"baseline_{architecture}", endpoint)]
            )
            lower, upper = percentile_interval(contrast, confidence)
            rows.append(
                {
                    "experiment": "E9",
                    "system": f"clahe_minus_baseline_{architecture}",
                    "endpoint": endpoint,
                    "estimate": float(np.mean(contrast)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence,
                    "resamples": resamples,
                    "seed": int(config["bootstrap"]["seed"]),
                    "stratified": True,
                }
            )

    for endpoint in ["mean_feature_cosine", "mean_feature_normalized_l2_drift"]:
        baseline_average = (
            np.asarray(values[("baseline_densenet", endpoint)])
            + np.asarray(values[("baseline_resnet", endpoint)])
        ) / 2.0
        clahe_average = (
            np.asarray(values[("clahe_densenet", endpoint)])
            + np.asarray(values[("clahe_resnet", endpoint)])
        ) / 2.0
        for system, endpoint_values in [
            ("baseline_mean_across_densenet_and_resnet", baseline_average),
            ("clahe_mean_across_densenet_and_resnet", clahe_average),
            (
                "clahe_minus_baseline_mean_across_densenet_and_resnet",
                clahe_average - baseline_average,
            ),
        ]:
            lower, upper = percentile_interval(endpoint_values, confidence)
            rows.append(
                {
                    "experiment": "E9",
                    "system": system,
                    "endpoint": endpoint,
                    "estimate": float(np.mean(endpoint_values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence,
                    "resamples": resamples,
                    "seed": int(config["bootstrap"]["seed"]),
                    "stratified": True,
                }
            )
    return pd.DataFrame(rows)


def run_summary(config, config_path):
    output_summary = resolve_repo_path(config["output"]["robustness_summary"])
    output_bootstrap = resolve_repo_path(config["output"]["bootstrap_intervals"])
    require_available_outputs([output_summary, output_bootstrap])
    _, hashes = validate_static_inputs(config, include_e9_checkpoint=True)
    validate_e9_training_record(config, config_path, hashes["clahe_resnet"])
    load_validation_lock(config, config_path, hashes)
    summary, evidence = summarize_point_estimates(config, config_path)
    intervals = bootstrap_intervals(config, evidence)
    write_csv_atomic(output_summary, summary)
    write_csv_atomic(output_bootstrap, intervals)
    print("E9 robustness summary complete.")
    print("Summary:", output_summary)
    print("Bootstrap intervals:", output_bootstrap)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate E9 validation, one test condition, or the final robustness summary."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", required=True, choices=["val", "condition", "summarize"])
    parser.add_argument("--condition", choices=["clean", *PERTURBED_CONDITIONS])
    args = parser.parse_args()
    config, config_path = load_e9_config(args.config)

    if args.stage == "val":
        if args.condition is not None:
            parser.error("--condition is only valid with --stage condition.")
        run_validation(config, config_path)
    elif args.stage == "condition":
        if args.condition is None:
            parser.error("--stage condition requires --condition.")
        run_condition(config, config_path, args.condition)
    else:
        if args.condition is not None:
            parser.error("--condition is only valid with --stage condition.")
        run_summary(config, config_path)


if __name__ == "__main__":
    main()
