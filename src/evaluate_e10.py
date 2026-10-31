import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.e10_protocol import (
    FEEDBACK_COLUMNS,
    build_partitions,
    load_checkpoint_model,
    load_e10_config,
    make_dataset,
    output_path,
    require_outputs_absent,
    sha256_file,
    source_hashes,
    validate_e4_is_test_only,
    validate_manifest,
    validate_partition_outputs,
    validate_static_inputs,
    write_csv_atomic,
    write_json_atomic,
)
from src.train_e7 import calculate_metrics, configure_determinism, utc_now
from src.utils import get_device


ARM_LABELS = {
    "pre": "Pre-feedback DenseNet-121",
    "control": "No-feedback continued-training control",
    "feedback": "Simulated-feedback adapted DenseNet-121",
}
ARM_OUTPUTS = {
    "pre": ("pre_checkpoint", "pre_run_metadata"),
    "control": ("control_checkpoint", "control_run_metadata"),
    "feedback": ("feedback_checkpoint", "feedback_run_metadata"),
}


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_complete_checkpoint(config, arm):
    checkpoint_key, metadata_key = ARM_OUTPUTS[arm]
    checkpoint_path = output_path(config, checkpoint_key)
    metadata_path = output_path(config, metadata_key)
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Complete E10 {arm} training is required.")
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"E10 {arm} training metadata is not complete.")
    digest = sha256_file(checkpoint_path)
    if metadata.get("checkpoint_sha256") != digest:
        raise ValueError(f"E10 {arm} checkpoint differs from its training record.")
    return checkpoint_path, digest


def _loader(config, manifest_key, split, role=None):
    dataset = make_dataset(config, output_path(config, manifest_key), split, augment=False)
    if role is not None:
        if "e10_role" not in dataset.df.columns:
            raise ValueError("E10 role filtering requires the partition manifest.")
        dataset.df = dataset.df[dataset.df["e10_role"] == role].reset_index(drop=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=get_device().type == "cuda",
    )
    return dataset, loader


def _collect(model, loader, device, description):
    model.eval()
    image_ids = []
    image_paths = []
    targets = []
    probabilities = []
    with torch.no_grad():
        for images, metadata, batch_targets, paths in tqdm(loader, desc=description):
            logits = model(images.to(device), metadata.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            targets.extend(batch_targets.numpy().astype(int).tolist())
            image_paths.extend(map(str, paths))
            image_ids.extend(Path(str(path)).name for path in paths)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("E10 inference produced duplicate image IDs.")
    return {
        "image_id": np.asarray(image_ids),
        "image_path": np.asarray(image_paths),
        "target": np.asarray(targets, dtype=np.int64),
        "probability": np.asarray(probabilities, dtype=np.float64),
    }


def analyze_thresholds(targets, probabilities, grid):
    rows = [calculate_metrics(targets, probabilities, float(value)) for value in grid]
    analysis = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_f1 = analysis["f1"].max()
    selected = analysis[np.isclose(analysis["f1"], best_f1)].iloc[0].to_dict()
    return analysis, selected


def prediction_frame(arm, split, result, threshold):
    return pd.DataFrame(
        {
            "experiment": "E10",
            "arm": arm,
            "model": ARM_LABELS[arm],
            "split": split,
            "image_id": result["image_id"],
            "target": result["target"],
            "probability": result["probability"],
            "prediction": (result["probability"] >= float(threshold)).astype(int),
            "threshold": float(threshold),
        }
    )


def metric_frame(arm, split, result, threshold):
    return pd.DataFrame(
        [
            {
                "experiment": "E10",
                "arm": arm,
                "model": ARM_LABELS[arm],
                "split": split,
                "n": len(result["target"]),
                "positive_n": int(result["target"].sum()),
                "negative_n": int(len(result["target"]) - result["target"].sum()),
                **calculate_metrics(result["target"], result["probability"], threshold),
            }
        ]
    )


def run_prepare(config, config_path):
    static = validate_static_inputs(config)
    manifest, _, manifest_hash = validate_manifest(config)
    validate_e4_is_test_only(config, manifest)
    output_keys = [
        "partition_manifest",
        "partition_metadata",
        "pre_manifest",
        "control_manifest",
        "feedback_manifest",
    ]
    require_outputs_absent([output_path(config, key) for key in output_keys])
    partition, pre, control, feedback, counts = build_partitions(config, manifest)
    frames = {
        "partition_manifest": partition,
        "pre_manifest": pre,
        "control_manifest": control,
        "feedback_manifest": feedback,
    }
    for key, frame in frames.items():
        write_csv_atomic(output_path(config, key), frame)
    metadata = {
        "experiment": "E10",
        "stage": "partition_lock",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": manifest_hash,
        "feedback_fraction": float(config["partition"]["feedback_fraction"]),
        "seed": int(config["seed"]),
        "stratification": "patient_has_positive",
        "counts": counts,
        "e4_retained_evidence": "four cases; all original test split; not used for training",
        "static_input_sha256": {label: digest for label, (_, digest) in static.items()},
        "source_sha256": source_hashes(),
        "output_sha256": {
            key: sha256_file(output_path(config, key)) for key in frames
        },
    }
    write_json_atomic(output_path(config, "partition_metadata"), metadata)
    print("E10 deterministic partitions prepared and locked.")
    print("Partition metadata:", output_path(config, "partition_metadata"))
    print(pd.Series(counts))


def _evaluate_arm(config, arm, manifest_key, split, description):
    checkpoint_path, checkpoint_hash = _require_complete_checkpoint(config, arm)
    device = get_device()
    model, checkpoint = load_checkpoint_model(config, checkpoint_path, arm, device)
    if arm != "pre":
        pre_hash = sha256_file(output_path(config, "pre_checkpoint"))
        if checkpoint.get("pre_checkpoint_sha256") != pre_hash:
            raise ValueError(f"E10 {arm} checkpoint did not start from the locked pre checkpoint.")
    _, loader = _loader(config, manifest_key, split)
    result = _collect(model, loader, device, description)
    return result, checkpoint_hash


def run_feedback(config, config_path):
    validate_static_inputs(config)
    validate_manifest(config)
    partition_metadata = validate_partition_outputs(config)
    keys = [
        "pre_val_predictions",
        "pre_val_metrics",
        "pre_threshold_analysis",
        "pre_feedback_lock",
        "feedback_pool_predictions",
        "simulated_feedback",
        "feedback_lock",
    ]
    require_outputs_absent([output_path(config, key) for key in keys])
    pre_checkpoint_path, pre_hash = _require_complete_checkpoint(config, "pre")
    device = get_device()
    model, _ = load_checkpoint_model(config, pre_checkpoint_path, "pre", device)

    _, val_loader = _loader(config, "pre_manifest", "val")
    val = _collect(model, val_loader, device, "E10 pre-feedback val")
    analysis, selected = analyze_thresholds(
        val["target"], val["probability"], config["threshold_selection"]["grid"]
    )
    threshold = float(selected["threshold"])
    write_csv_atomic(output_path(config, "pre_val_predictions"), prediction_frame("pre", "val", val, threshold))
    write_csv_atomic(output_path(config, "pre_val_metrics"), metric_frame("pre", "val", val, threshold))
    write_csv_atomic(output_path(config, "pre_threshold_analysis"), analysis)
    pre_lock = {
        "experiment": "E10",
        "stage": "pre_feedback_validation_lock",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "partition_metadata_sha256": sha256_file(output_path(config, "partition_metadata")),
        "pre_checkpoint_sha256": pre_hash,
        "threshold": threshold,
        "selection_split": "val",
        "selection_metric": "f1",
        "tie_break": "lowest_threshold",
        "pre_val_predictions_sha256": sha256_file(output_path(config, "pre_val_predictions")),
        "pre_val_metrics_sha256": sha256_file(output_path(config, "pre_val_metrics")),
        "pre_threshold_analysis_sha256": sha256_file(output_path(config, "pre_threshold_analysis")),
    }
    write_json_atomic(output_path(config, "pre_feedback_lock"), pre_lock)

    _, pool_loader = _loader(config, "partition_manifest", "train", role="feedback_pool")
    pool = _collect(model, pool_loader, device, "E10 feedback pool")
    pool_predictions = prediction_frame("pre", "feedback_pool", pool, threshold)
    write_csv_atomic(output_path(config, "feedback_pool_predictions"), pool_predictions)
    partition = pd.read_csv(output_path(config, "partition_manifest"))
    lookup = partition.set_index("image_id")
    rows = []
    for item in pool_predictions.itertuples(index=False):
        reference = int(item.target)
        prediction = int(item.prediction)
        source = lookup.loc[item.image_id]
        rows.append(
            {
                "experiment": "E10",
                "schema_version": int(config["feedback"]["schema_version"]),
                "feedback_source": config["feedback"]["source"],
                "model_arm": "pre",
                "model_checkpoint_sha256": pre_hash,
                "image_id": item.image_id,
                "image_path": str(source["image_path"]),
                "patient_id": source["Patient ID"],
                "original_split": "train",
                "model_probability": float(item.probability),
                "model_prediction": prediction,
                "reviewer_decision": "agree" if prediction == reference else "disagree",
                "corrected_label": reference,
                "reference_label": reference,
            }
        )
    feedback = pd.DataFrame(rows, columns=FEEDBACK_COLUMNS)
    if len(feedback) != int(config["partition"]["expected"]["feedback_images"]):
        raise ValueError("E10 simulated feedback row count differs from the locked pool.")
    write_csv_atomic(output_path(config, "simulated_feedback"), feedback)
    feedback_lock = {
        "experiment": "E10",
        "stage": "simulated_feedback_lock",
        "created_at_utc": utc_now(),
        "feedback_source": config["feedback"]["source"],
        "clinical_reviewer": False,
        "localization_feedback": False,
        "config_sha256": sha256_file(config_path),
        "partition_metadata_sha256": sha256_file(output_path(config, "partition_metadata")),
        "pre_feedback_lock_sha256": sha256_file(output_path(config, "pre_feedback_lock")),
        "pre_checkpoint_sha256": pre_hash,
        "feedback_pool_predictions_sha256": sha256_file(output_path(config, "feedback_pool_predictions")),
        "simulated_feedback_sha256": sha256_file(output_path(config, "simulated_feedback")),
        "records": len(feedback),
        "agree": int((feedback["reviewer_decision"] == "agree").sum()),
        "disagree": int((feedback["reviewer_decision"] == "disagree").sum()),
    }
    write_json_atomic(output_path(config, "feedback_lock"), feedback_lock)
    print("E10 pre-feedback validation threshold locked:", threshold)
    print("E10 simulated oracle feedback locked:", output_path(config, "feedback_lock"))


def _validate_feedback_lock(config):
    lock_path = output_path(config, "feedback_lock")
    if not lock_path.is_file():
        raise FileNotFoundError("E10 feedback lock is missing.")
    lock = _read_json(lock_path)
    checks = {
        "pre_checkpoint_sha256": output_path(config, "pre_checkpoint"),
        "pre_feedback_lock_sha256": output_path(config, "pre_feedback_lock"),
        "feedback_pool_predictions_sha256": output_path(config, "feedback_pool_predictions"),
        "simulated_feedback_sha256": output_path(config, "simulated_feedback"),
    }
    for field, path in checks.items():
        if not path.is_file() or lock.get(field) != sha256_file(path):
            raise ValueError(f"E10 feedback lock mismatch: {field}.")
    return lock


def run_val(config, config_path):
    validate_static_inputs(config)
    validate_manifest(config)
    validate_partition_outputs(config)
    _validate_feedback_lock(config)
    keys = [
        "control_val_predictions",
        "control_val_metrics",
        "control_threshold_analysis",
        "feedback_val_predictions",
        "feedback_val_metrics",
        "feedback_threshold_analysis",
        "validation_lock",
    ]
    require_outputs_absent([output_path(config, key) for key in keys])
    pre_lock = _read_json(output_path(config, "pre_feedback_lock"))
    checkpoint_hashes = {"pre": sha256_file(output_path(config, "pre_checkpoint"))}
    thresholds = {"pre": float(pre_lock["threshold"])}
    output_hashes = {}
    for arm, prefix in (("control", "control"), ("feedback", "feedback")):
        result, checkpoint_hash = _evaluate_arm(
            config, arm, f"{prefix}_manifest", "val", f"E10 {arm} val"
        )
        checkpoint_hashes[arm] = checkpoint_hash
        analysis, selected = analyze_thresholds(
            result["target"], result["probability"], config["threshold_selection"]["grid"]
        )
        threshold = float(selected["threshold"])
        thresholds[arm] = threshold
        prediction_key = f"{prefix}_val_predictions"
        metrics_key = f"{prefix}_val_metrics"
        analysis_key = f"{prefix}_threshold_analysis"
        write_csv_atomic(output_path(config, prediction_key), prediction_frame(arm, "val", result, threshold))
        write_csv_atomic(output_path(config, metrics_key), metric_frame(arm, "val", result, threshold))
        write_csv_atomic(output_path(config, analysis_key), analysis)
        for key in (prediction_key, metrics_key, analysis_key):
            output_hashes[key] = sha256_file(output_path(config, key))
    lock = {
        "experiment": "E10",
        "stage": "three_arm_validation_lock",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "feedback_lock_sha256": sha256_file(output_path(config, "feedback_lock")),
        "pre_feedback_lock_sha256": sha256_file(output_path(config, "pre_feedback_lock")),
        "checkpoint_sha256": checkpoint_hashes,
        "thresholds": thresholds,
        "selection_split": "val",
        "selection_metric": "f1",
        "tie_break": "lowest_threshold",
        "output_sha256": output_hashes,
    }
    write_json_atomic(output_path(config, "validation_lock"), lock)
    print("E10 three-arm validation thresholds locked:", thresholds)
    print("Validation lock:", output_path(config, "validation_lock"))


def _load_validation_lock(config, config_path):
    path = output_path(config, "validation_lock")
    if not path.is_file():
        raise FileNotFoundError("Run evaluate_e10 --stage val before test evaluation.")
    lock = _read_json(path)
    if lock.get("stage") != "three_arm_validation_lock":
        raise ValueError("E10 validation lock has an unexpected stage.")
    if lock.get("config_sha256") != sha256_file(config_path):
        raise ValueError("E10 validation lock references a different config.")
    if lock.get("feedback_lock_sha256") != sha256_file(output_path(config, "feedback_lock")):
        raise ValueError("E10 validation lock references different feedback evidence.")
    if lock.get("pre_feedback_lock_sha256") != sha256_file(output_path(config, "pre_feedback_lock")):
        raise ValueError("E10 validation lock references different pre-feedback evidence.")
    for arm, digest in lock.get("checkpoint_sha256", {}).items():
        checkpoint_path, current = _require_complete_checkpoint(config, arm)
        if digest != current:
            raise ValueError(f"E10 validation lock checkpoint mismatch: {arm}.")
    if set(lock.get("thresholds", {})) != {"pre", "control", "feedback"}:
        raise ValueError("E10 validation lock does not contain all arm thresholds.")
    for key, digest in lock.get("output_sha256", {}).items():
        path = output_path(config, key)
        if not path.is_file() or digest != sha256_file(path):
            raise ValueError(f"E10 validation output mismatch: {key}.")
    return lock


def _assert_aligned(results):
    reference = results["pre"]
    for arm in ("control", "feedback"):
        current = results[arm]
        if not np.array_equal(reference["image_id"], current["image_id"]):
            raise ValueError(f"E10 test image IDs are not aligned for {arm}.")
        if not np.array_equal(reference["target"], current["target"]):
            raise ValueError(f"E10 test targets are not aligned for {arm}.")


def run_test(config, config_path):
    validate_static_inputs(config)
    validate_manifest(config)
    validate_partition_outputs(config)
    lock = _load_validation_lock(config, config_path)
    keys = [
        "pre_test_predictions",
        "pre_test_metrics",
        "control_test_predictions",
        "control_test_metrics",
        "feedback_test_predictions",
        "feedback_test_metrics",
        "test_comparison",
    ]
    require_outputs_absent([output_path(config, key) for key in keys])
    results = {}
    for arm, prefix in (("pre", "pre"), ("control", "control"), ("feedback", "feedback")):
        results[arm], _ = _evaluate_arm(
            config, arm, f"{prefix}_manifest", "test", f"E10 {arm} test"
        )
    _assert_aligned(results)
    metric_frames = []
    for arm, prefix in (("pre", "pre"), ("control", "control"), ("feedback", "feedback")):
        threshold = float(lock["thresholds"][arm])
        predictions = prediction_frame(arm, "test", results[arm], threshold)
        metrics = metric_frame(arm, "test", results[arm], threshold)
        write_csv_atomic(output_path(config, f"{prefix}_test_predictions"), predictions)
        write_csv_atomic(output_path(config, f"{prefix}_test_metrics"), metrics)
        metric_frames.append(metrics)
    comparison = pd.concat(metric_frames, ignore_index=True)
    comparison["validation_lock_sha256"] = sha256_file(output_path(config, "validation_lock"))
    for arm, prefix in (("pre", "pre"), ("control", "control"), ("feedback", "feedback")):
        comparison[f"{arm}_test_predictions_sha256"] = sha256_file(
            output_path(config, f"{prefix}_test_predictions")
        )
        comparison[f"{arm}_test_metrics_sha256"] = sha256_file(
            output_path(config, f"{prefix}_test_metrics")
        )
    write_csv_atomic(output_path(config, "test_comparison"), comparison)
    print("E10 held-out test evaluation complete.")
    print(comparison.to_string(index=False))
    print("Test comparison:", output_path(config, "test_comparison"))


def _bootstrap_indices(targets, rng):
    negative = np.flatnonzero(targets == 0)
    positive = np.flatnonzero(targets == 1)
    sampled = np.concatenate(
        [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
    )
    rng.shuffle(sampled)
    return sampled


def run_summarize(config):
    output = output_path(config, "bootstrap_intervals")
    require_outputs_absent([output])
    frames = {
        arm: pd.read_csv(output_path(config, f"{prefix}_test_predictions")).sort_values("image_id").reset_index(drop=True)
        for arm, prefix in (("pre", "pre"), ("control", "control"), ("feedback", "feedback"))
    }
    ids = frames["pre"]["image_id"].to_numpy()
    targets = frames["pre"]["target"].to_numpy(dtype=int)
    for arm in ("control", "feedback"):
        if not np.array_equal(ids, frames[arm]["image_id"].to_numpy()) or not np.array_equal(
            targets, frames[arm]["target"].to_numpy(dtype=int)
        ):
            raise ValueError(f"E10 paired bootstrap inputs are not aligned for {arm}.")
    thresholds = {arm: float(frames[arm]["threshold"].iloc[0]) for arm in frames}
    probabilities = {arm: frames[arm]["probability"].to_numpy(dtype=float) for arm in frames}
    metric_names = ["roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "accuracy"]
    comparisons = (("feedback_minus_control", "control"), ("feedback_minus_pre", "pre"))
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    samples = {(name, metric): [] for name, _ in comparisons for metric in metric_names}
    for _ in tqdm(range(int(config["bootstrap"]["resamples"])), desc="E10 paired bootstrap"):
        index = _bootstrap_indices(targets, rng)
        sampled_targets = targets[index]
        metrics = {
            arm: calculate_metrics(sampled_targets, values[index], thresholds[arm])
            for arm, values in probabilities.items()
        }
        for comparison, comparator in comparisons:
            for metric in metric_names:
                samples[(comparison, metric)].append(metrics["feedback"][metric] - metrics[comparator][metric])
    confidence = float(config["bootstrap"]["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    point = {
        arm: calculate_metrics(targets, values, thresholds[arm])
        for arm, values in probabilities.items()
    }
    rows = []
    input_hashes = {
        f"{arm}_test_predictions_sha256": sha256_file(
            output_path(config, f"{prefix}_test_predictions")
        )
        for arm, prefix in (("pre", "pre"), ("control", "control"), ("feedback", "feedback"))
    }
    comparison_hash = sha256_file(output_path(config, "test_comparison"))
    for comparison, comparator in comparisons:
        for metric in metric_names:
            values = np.asarray(samples[(comparison, metric)])
            rows.append(
                {
                    "experiment": "E10",
                    "comparison": comparison,
                    "metric": metric,
                    "point_difference": point["feedback"][metric] - point[comparator][metric],
                    "ci_lower": float(np.quantile(values, alpha)),
                    "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                    "confidence_level": confidence,
                    "resamples": int(config["bootstrap"]["resamples"]),
                    "seed": int(config["bootstrap"]["seed"]),
                    "stratified": True,
                    "test_comparison_sha256": comparison_hash,
                    **input_hashes,
                }
            )
    write_csv_atomic(output, pd.DataFrame(rows))
    print("E10 paired, stratified bootstrap complete.")
    print("Bootstrap intervals:", output)


def main():
    parser = argparse.ArgumentParser(description="Run a staged E10 evaluation step.")
    parser.add_argument("--config", default="config_e10.yaml")
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "feedback", "val", "test", "summarize")
    )
    args = parser.parse_args()
    config, config_path = load_e10_config(args.config)
    configure_determinism(int(config["seed"]), True)
    if args.stage == "prepare":
        run_prepare(config, config_path)
    elif args.stage == "feedback":
        run_feedback(config, config_path)
    elif args.stage == "val":
        run_val(config, config_path)
    elif args.stage == "test":
        run_test(config, config_path)
    else:
        run_summarize(config)


if __name__ == "__main__":
    main()
