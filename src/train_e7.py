import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import NIHPneumoniaDataset
from src.resnet_model import MultiModalResNet50
from src.utils import get_device, seed_everything


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config_e7.yaml"

LEGACY_OUTPUTS = {
    "outputs/checkpoints/e1_image_only.pt",
    "outputs/checkpoints/e2_multimodal.pt",
    "outputs/checkpoints/e6_a_multimodal.pt",
    "outputs/checkpoints/e6_b_multimodal.pt",
    "outputs/metrics/e1_image_only_metrics.csv",
    "outputs/metrics/e1_image_only_test_metrics.csv",
    "outputs/metrics/e2_multimodal_metrics.csv",
    "outputs/metrics/e6_a_metrics.csv",
    "outputs/metrics/e6_b_metrics.csv",
    "outputs/metrics/e6_preprocessing_metrics.csv",
}

E7_SOURCE_FILES = [
    "config_e7.yaml",
    "src/dataset.py",
    "src/resnet_model.py",
    "src/train_e7.py",
    "src/utils.py",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest().upper()


def load_e7_config(path=DEFAULT_CONFIG):
    config_path = resolve_repo_path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    validate_config(config)
    return config, config_path


def validate_config(config):
    if config.get("experiment", {}).get("id") != "E7":
        raise ValueError("E7 configuration must declare experiment.id as E7.")

    if config.get("model", {}).get("backbone") != "resnet50":
        raise ValueError("E7 backbone must be resnet50.")

    if config.get("data", {}).get("preprocessing") != "baseline":
        raise ValueError("E7 is restricted to baseline preprocessing; CLAHE is E9 scope.")

    image_root = config.get("data", {}).get("image_root")

    if not image_root:
        raise ValueError("E7 configuration must declare data.image_root.")

    if Path(image_root).is_absolute():
        raise ValueError("E7 data.image_root must be repository-relative.")

    if config.get("threshold_selection", {}).get("split") != "val":
        raise ValueError("E7 threshold selection must use the validation split.")

    if config.get("threshold_selection", {}).get("metric") != "f1":
        raise ValueError("E7 threshold selection metric must be f1.")

    if config.get("threshold_selection", {}).get("tie_break") != "lowest_threshold":
        raise ValueError("E7 threshold ties must select the lowest threshold.")

    if config.get("training", {}).get("checkpoint_metric") != "roc_auc":
        raise ValueError("E7 checkpoint selection metric must be roc_auc.")

    required_outputs = {
        "checkpoint",
        "training_metrics",
        "run_metadata",
        "val_predictions",
        "val_metrics",
        "threshold_analysis",
        "locked_threshold",
        "test_predictions",
        "test_metrics",
    }
    missing_outputs = sorted(required_outputs.difference(config.get("output", {})))

    if missing_outputs:
        raise ValueError("E7 configuration is missing outputs: " + ", ".join(missing_outputs))

    output_values = list(config.get("output", {}).values())

    if len(output_values) != len(set(output_values)):
        raise ValueError("E7 output paths must be unique.")

    normalized_outputs = {
        Path(value).as_posix()
        for value in output_values
    }

    collisions = normalized_outputs.intersection(LEGACY_OUTPUTS)

    if collisions:
        raise ValueError(
            "E7 output paths collide with preserved E1-E6 artefacts: "
            + ", ".join(sorted(collisions))
        )

    grid = config.get("threshold_selection", {}).get("grid", [])

    if not grid or any(float(value) <= 0 or float(value) >= 1 for value in grid):
        raise ValueError("E7 threshold grid must contain values strictly between 0 and 1.")

    if list(map(float, grid)) != sorted(set(map(float, grid))):
        raise ValueError("E7 threshold grid must be unique and ascending.")


def validate_manifest(config):
    manifest_path = resolve_repo_path(config["data"]["manifest"])

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    actual_hash = sha256_file(manifest_path)
    expected_hash = str(config["data"]["manifest_sha256"]).upper()

    if actual_hash != expected_hash:
        raise ValueError(
            "Manifest hash differs from the approved E7 protocol. "
            f"Expected {expected_hash}, found {actual_hash}."
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
        raise ValueError("Manifest is missing columns: " + ", ".join(missing))

    if frame["image_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate image paths.")

    image_ids = frame["image_path"].map(lambda value: Path(str(value)).name)

    if image_ids.duplicated().any():
        raise ValueError(
            "Manifest contains duplicate image filenames; E7/E8 require unique image IDs."
        )

    patient_split_counts = frame.groupby("Patient ID")["split"].nunique()

    if int((patient_split_counts > 1).sum()) != 0:
        raise ValueError("Manifest contains patients assigned to multiple splits.")

    targets = set(frame["target"].dropna().astype(int).unique().tolist())

    if targets != {0, 1}:
        raise ValueError(f"Manifest target values must be {{0, 1}}, found {targets}.")

    actual_counts = {
        str(split): int(count)
        for split, count in frame.groupby("split").size().items()
    }
    expected_counts = {
        key: int(value)
        for key, value in config["data"]["expected_split_counts"].items()
    }

    if actual_counts != expected_counts:
        raise ValueError(
            "Manifest split counts differ from the approved E7 protocol. "
            f"Expected {expected_counts}, found {actual_counts}."
        )

    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": actual_hash,
        "rows": int(len(frame)),
        "patients": int(frame["Patient ID"].nunique()),
        "split_counts": actual_counts,
        "positive_counts": {
            split: int(group["target"].sum())
            for split, group in frame.groupby("split")
        },
    }

    return frame, summary


def validate_image_inventory(config, manifest_frame):
    image_root = resolve_repo_path(config["data"]["image_root"])

    if not image_root.is_dir():
        raise FileNotFoundError(f"Configured E7 image root not found: {image_root}")

    image_files = [
        *image_root.glob("*.png"),
        *image_root.glob("*/*.png"),
    ]
    paths_by_name = {}

    for image_path in image_files:
        paths_by_name.setdefault(image_path.name, []).append(image_path)

    manifest_names = manifest_frame["image_path"].map(
        lambda value: Path(str(value)).name
    )
    missing = sorted(
        name
        for name in manifest_names
        if name not in paths_by_name
    )
    ambiguous = sorted(
        name
        for name in manifest_names
        if len(paths_by_name.get(name, [])) > 1
    )

    if missing:
        raise FileNotFoundError(
            "E7 image inventory is missing manifest files, including: "
            + ", ".join(missing[:10])
        )

    if ambiguous:
        raise ValueError(
            "E7 image inventory contains non-unique manifest filenames, including: "
            + ", ".join(ambiguous[:10])
        )

    return {
        "image_root": str(image_root),
        "inventory_png_files": int(len(image_files)),
        "manifest_images_resolved": int(len(manifest_names)),
        "missing_manifest_images": 0,
        "ambiguous_manifest_images": 0,
    }


def calculate_metrics(targets, probabilities, threshold):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= float(threshold)).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "sensitivity": float(recall_score(targets, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(model, loader, device, threshold):
    model.eval()
    targets = []
    probabilities = []

    with torch.no_grad():
        for images, metadata, batch_targets, _ in loader:
            images = images.to(device)
            metadata = metadata.to(device)

            logits = model(images, metadata)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()

            targets.extend(batch_targets.numpy().tolist())
            probabilities.extend(batch_probabilities.tolist())

    return calculate_metrics(targets, probabilities, threshold)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    temporary.replace(path)


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def current_git_status():
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def source_hashes():
    hashes = {}

    for relative_path in E7_SOURCE_FILES:
        path = ROOT / relative_path

        if not path.exists():
            raise FileNotFoundError(f"Required E7 source file is missing: {path}")

        hashes[relative_path] = sha256_file(path)

    return hashes


def environment_record(device):
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
    }


def require_available_outputs(paths, overwrite):
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        joined = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "E7 training outputs already exist. Archive them before a retest, "
            "or use --overwrite only under a documented protocol amendment:\n"
            + joined
        )


def configure_determinism(seed, enabled):
    seed_everything(seed)

    if enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(config):
    model_config = config["model"]

    return MultiModalResNet50(
        pretrained=bool(model_config["pretrained"]),
        metadata_dim=int(model_config["metadata_dim"]),
        image_embedding_dim=int(model_config["image_embedding_dim"]),
        metadata_embedding_dim=int(model_config["metadata_embedding_dim"]),
        dropout=float(model_config["dropout"]),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train E7: metadata-fused ResNet-50."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the approved E7 YAML configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing E7 outputs. Archive prior evidence before using this.",
    )
    args = parser.parse_args()

    config, config_path = load_e7_config(args.config)
    manifest_frame, manifest_summary = validate_manifest(config)
    image_inventory = validate_image_inventory(config, manifest_frame)

    checkpoint_path = resolve_repo_path(config["output"]["checkpoint"])
    metrics_path = resolve_repo_path(config["output"]["training_metrics"])
    metadata_path = resolve_repo_path(config["output"]["run_metadata"])

    require_available_outputs(
        [checkpoint_path, metrics_path, metadata_path],
        args.overwrite,
    )

    seed = int(config["seed"])
    configure_determinism(
        seed,
        bool(config["training"].get("deterministic", False)),
    )

    device = get_device()
    config_hash = sha256_file(config_path)
    git_commit = current_git_commit()
    git_status = current_git_status()
    e7_source_hashes = source_hashes()

    run_record = {
        "experiment": "E7",
        "status": "started",
        "started_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git_commit": git_commit,
        "git_status": git_status,
        "source_sha256": e7_source_hashes,
        "config": str(config_path),
        "config_sha256": config_hash,
        "manifest": manifest_summary,
        "image_inventory": image_inventory,
        "environment": environment_record(device),
        "seed": seed,
    }

    write_json(metadata_path, run_record)

    try:
        print("Device:", device)
        print("Experiment: E7 - Multi-modal ResNet-50 + metadata")
        print("Config SHA-256:", config_hash)
        print("Manifest SHA-256:", manifest_summary["manifest_sha256"])

        manifest_path = str(resolve_repo_path(config["data"]["manifest"]))
        image_root = str(resolve_repo_path(config["data"]["image_root"]))

        train_dataset = NIHPneumoniaDataset(
            manifest_path,
            "train",
            int(config["data"]["image_size"]),
            augment=True,
            preprocessing=config["data"]["preprocessing"],
            image_root=image_root,
        )
        validation_dataset = NIHPneumoniaDataset(
            manifest_path,
            "val",
            int(config["data"]["image_size"]),
            augment=False,
            preprocessing=config["data"]["preprocessing"],
            image_root=image_root,
        )

        print("Training samples:", len(train_dataset))
        print("Validation samples:", len(validation_dataset))

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )

        model = build_model(config).to(device)
        model.freeze_backbone()

        training_targets = train_dataset.df["target"]
        positives = int(training_targets.sum())
        negatives = int(len(training_targets) - positives)

        if positives <= 0 or negatives <= 0:
            raise ValueError("E7 training split must contain both target classes.")

        positive_weight = torch.tensor(
            [negatives / positives],
            device=device,
            dtype=torch.float32,
        )

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        epochs = int(config["training"]["epochs"])
        frozen_epochs = int(config["training"]["freeze_backbone_epochs"])
        monitor_threshold = float(
            config["training"]["validation_monitor_threshold"]
        )

        best_auc = float("-inf")
        best_epoch = None
        history = []

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            if epoch == frozen_epochs:
                model.unfreeze_backbone()

            model.train()
            running_loss = 0.0

            for images, metadata, targets, _ in tqdm(
                train_loader,
                desc=f"E7 Epoch {epoch + 1}/{epochs}",
            ):
                images = images.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images, metadata)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())

            train_loss = running_loss / len(train_loader)
            metrics = evaluate(
                model,
                validation_loader,
                device,
                monitor_threshold,
            )

            epoch_record = {
                "experiment": "E7",
                "model": "Multi-modal ResNet-50 + metadata",
                "epoch": epoch + 1,
                "seed": seed,
                "training_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "train_loss": train_loss,
                **metrics,
            }
            history.append(epoch_record)
            pd.DataFrame(history).to_csv(metrics_path, index=False)

            print(
                f"Epoch {epoch + 1}: "
                f"loss={train_loss:.4f}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"specificity={metrics['specificity']:.4f}, "
                f"f1={metrics['f1']:.4f}, "
                f"roc_auc={metrics['roc_auc']:.4f}, "
                f"pr_auc={metrics['pr_auc']:.4f}"
            )

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_epoch = epoch + 1

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "experiment": "E7",
                        "model": "Multi-modal ResNet-50 + metadata",
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_auc": best_auc,
                        "git_commit": git_commit,
                        "git_status": git_status,
                        "source_sha256": e7_source_hashes,
                        "config_sha256": config_hash,
                        "manifest_sha256": manifest_summary["manifest_sha256"],
                    },
                    checkpoint_path,
                )

                print("Saved best checkpoint:", checkpoint_path)

            run_record.update(
                {
                    "status": "running",
                    "last_completed_epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_val_auc": best_auc,
                }
            )
            write_json(metadata_path, run_record)

        if best_epoch is None or not checkpoint_path.exists():
            raise RuntimeError("E7 training completed without a valid checkpoint.")

        run_record.update(
            {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "best_epoch": best_epoch,
                "best_val_auc": best_auc,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "training_metrics": str(metrics_path),
                "training_metrics_sha256": sha256_file(metrics_path),
            }
        )
        write_json(metadata_path, run_record)

        print()
        print("E7 training complete.")
        print("Best epoch:", best_epoch)
        print("Best validation ROC-AUC:", best_auc)
        print("Checkpoint:", checkpoint_path)
        print("Training metrics:", metrics_path)
        print("Run metadata:", metadata_path)

    except KeyboardInterrupt as error:
        run_record.update(
            {
                "status": "interrupted",
                "stopped_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": "Training was interrupted by the user.",
            }
        )
        write_json(metadata_path, run_record)
        raise

    except Exception as error:
        run_record.update(
            {
                "status": "failed",
                "stopped_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(metadata_path, run_record)
        raise


if __name__ == "__main__":
    main()

