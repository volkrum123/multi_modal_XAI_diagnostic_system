import argparse
import platform
import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.e9_dataset import E9RobustnessDataset
from src.resnet_model import MultiModalResNet50
from src.train_e7 import (
    calculate_metrics,
    configure_determinism,
    current_git_commit,
    current_git_status,
    sha256_file,
    utc_now,
    write_json,
)
from src.utils import get_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config_e9.yaml"

E9_SOURCE_FILES = [
    "config_e9.yaml",
    "src/dataset.py",
    "src/e9_dataset.py",
    "src/resnet_model.py",
    "src/train_e9.py",
    "src/utils.py",
]

EXPECTED_CONDITIONS = {
    "clean",
    "brightness_low",
    "brightness_high",
    "contrast_low",
    "contrast_high",
    "gamma_low",
    "gamma_high",
    "gaussian_noise",
    "gaussian_blur",
}


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_e9_config(path=DEFAULT_CONFIG):
    config_path = resolve_repo_path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    validate_config(config)
    return config, config_path


def validate_config(config):
    if config.get("experiment", {}).get("id") != "E9":
        raise ValueError("E9 configuration must declare experiment.id as E9.")

    data = config.get("data", {})

    if not data.get("image_root") or Path(data["image_root"]).is_absolute():
        raise ValueError("E9 data.image_root must be repository-relative.")

    clahe = config.get("preprocessing", {}).get("clahe", {})

    if clahe.get("name") != "clahe":
        raise ValueError("E9 must identify the experimental preprocessing arm as CLAHE.")

    if float(clahe.get("clip_limit", -1)) != 2.0:
        raise ValueError("E9 CLAHE clip_limit is locked at 2.0.")

    if list(clahe.get("tile_grid_size", [])) != [8, 8]:
        raise ValueError("E9 CLAHE tile_grid_size is locked at [8, 8].")

    fusion = config.get("fusion", {})

    if fusion.get("method") != "equal_weight_probability_average":
        raise ValueError("E9 fusion must use equal-weight probability averaging.")

    if float(fusion.get("densenet_weight", -1)) != 0.5 or float(
        fusion.get("resnet_weight", -1)
    ) != 0.5:
        raise ValueError("E9 fusion weights are locked at 0.50 and 0.50.")

    selection = config.get("threshold_selection", {})

    if selection.get("split") != "val" or selection.get("metric") != "f1":
        raise ValueError("E9 thresholds must be selected on validation F1.")

    if selection.get("tie_break") != "lowest_threshold":
        raise ValueError("E9 threshold ties must use the lowest threshold.")

    grid = list(map(float, selection.get("grid", [])))

    if not grid or grid != sorted(set(grid)) or any(x <= 0 or x >= 1 for x in grid):
        raise ValueError("E9 threshold grid must be unique, ascending, and inside (0, 1).")

    if set(config.get("perturbations", {})) != EXPECTED_CONDITIONS:
        raise ValueError("E9 perturbation names differ from the approved nine conditions.")

    if int(config.get("training", {}).get("epochs", -1)) != 5:
        raise ValueError("E9 CLAHE ResNet-50 training is locked at five epochs.")

    if int(config.get("training", {}).get("freeze_backbone_epochs", -1)) != 1:
        raise ValueError("E9 must freeze the ResNet backbone for exactly one epoch.")

    if config.get("training", {}).get("checkpoint_metric") != "roc_auc":
        raise ValueError("E9 checkpoint selection must use validation ROC-AUC.")

    if config.get("feature_stability", {}).get("layer") != "image_projection_post_relu":
        raise ValueError("E9 feature extraction must use post-ReLU image_projection.")

    required_outputs = {
        "checkpoint",
        "training_metrics",
        "run_metadata",
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
        "clean_features",
        "clean_reference_lock",
        "condition_root",
        "robustness_summary",
        "bootstrap_intervals",
    }
    outputs = config.get("output", {})
    missing = sorted(required_outputs.difference(outputs))

    if missing:
        raise ValueError("E9 configuration is missing outputs: " + ", ".join(missing))

    output_paths = [Path(value).as_posix() for value in outputs.values()]

    if len(output_paths) != len(set(output_paths)):
        raise ValueError("E9 output paths must be unique.")

    if any(not value.startswith("outputs/") for value in output_paths):
        raise ValueError("Every E9 output must be below outputs/.")

    if any("e9_" not in value for value in output_paths):
        raise ValueError("Every E9 output must be isolated under an e9_* name.")

    retained_inputs = set()

    for model_config in config.get("models", {}).values():
        if not isinstance(model_config, dict):
            continue
        for key, value in model_config.items():
            if key in {
                "checkpoint",
                "duplicate_checkpoint",
                "training_metrics",
                "threshold_lock",
                "test_predictions",
            } and value:
                retained_inputs.add(Path(value).as_posix())

    collisions = retained_inputs.intersection(output_paths)

    # The new E9 checkpoint is both a declared model input and a future output.
    collisions.discard(Path(outputs["checkpoint"]).as_posix())

    if collisions:
        raise ValueError("E9 outputs collide with retained evidence: " + ", ".join(collisions))


def require_hash(path, expected_hash, label):
    resolved = resolve_repo_path(path)

    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")

    actual = sha256_file(resolved)
    expected = str(expected_hash).upper()

    if actual != expected:
        raise ValueError(f"{label} hash mismatch. Expected {expected}, found {actual}.")

    return resolved, actual


def validate_manifest(config):
    manifest_path, manifest_hash = require_hash(
        config["data"]["manifest"],
        config["data"]["manifest_sha256"],
        "E9 manifest",
    )
    frame = pd.read_csv(manifest_path)
    required = {"image_path", "Patient ID", "target", "age", "sex", "view", "split"}
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError("E9 manifest is missing columns: " + ", ".join(missing))

    image_ids = frame["image_path"].map(lambda value: Path(str(value)).name)

    if image_ids.duplicated().any():
        raise ValueError("E9 requires globally unique image filenames.")

    if int((frame.groupby("Patient ID")["split"].nunique() > 1).sum()) != 0:
        raise ValueError("E9 manifest contains patients assigned to multiple splits.")

    expected_counts = {
        key: int(value) for key, value in config["data"]["expected_split_counts"].items()
    }
    actual_counts = {
        str(key): int(value) for key, value in frame.groupby("split").size().items()
    }

    if actual_counts != expected_counts:
        raise ValueError(f"E9 split counts differ: expected {expected_counts}, found {actual_counts}.")

    expected_positive = {
        key: int(value) for key, value in config["data"]["expected_positive_counts"].items()
    }
    actual_positive = {
        str(key): int(group["target"].sum()) for key, group in frame.groupby("split")
    }

    if actual_positive != expected_positive:
        raise ValueError(
            f"E9 positive counts differ: expected {expected_positive}, found {actual_positive}."
        )

    return frame, manifest_path, manifest_hash


def validate_image_inventory(config, manifest_frame):
    image_root = resolve_repo_path(config["data"]["image_root"])

    if not image_root.is_dir():
        raise FileNotFoundError(f"E9 image root not found: {image_root}")

    files = [*image_root.glob("*.png"), *image_root.glob("*/*.png")]
    by_name = {}

    for path in files:
        by_name.setdefault(path.name, []).append(path)

    names = manifest_frame["image_path"].map(lambda value: Path(str(value)).name)
    missing = [name for name in names if name not in by_name]
    ambiguous = [name for name in names if len(by_name.get(name, [])) > 1]

    if missing:
        raise FileNotFoundError("E9 image inventory is missing: " + ", ".join(missing[:10]))

    if ambiguous:
        raise ValueError("E9 image inventory has ambiguous names: " + ", ".join(ambiguous[:10]))

    return {
        "image_root": str(image_root),
        "inventory_png_files": len(files),
        "manifest_images_resolved": len(names),
        "missing_manifest_images": 0,
        "ambiguous_manifest_images": 0,
    }


def source_hashes():
    hashes = {}

    for relative in E9_SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required E9 source file is missing: {path}")
        hashes[relative] = sha256_file(path)

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
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
    }


def build_model(config, pretrained=True):
    architecture = config["model_architecture"]
    return MultiModalResNet50(
        pretrained=pretrained,
        metadata_dim=int(architecture["metadata_dim"]),
        image_embedding_dim=int(architecture["image_embedding_dim"]),
        metadata_embedding_dim=int(architecture["metadata_embedding_dim"]),
        dropout=float(architecture["dropout"]),
    )


def evaluate(model, loader, device, threshold):
    model.eval()
    targets = []
    probabilities = []

    with torch.no_grad():
        for images, metadata, batch_targets, _ in loader:
            logits = model(images.to(device), metadata.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            targets.extend(batch_targets.numpy().tolist())

    return calculate_metrics(targets, probabilities, threshold)


def require_available_outputs(paths, overwrite):
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        listing = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "E9 training outputs already exist. Archive them before rerunning, or use "
            "--overwrite only under a documented protocol amendment:\n" + listing
        )


def main():
    parser = argparse.ArgumentParser(description="Train the E9 CLAHE ResNet-50 arm.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config, config_path = load_e9_config(args.config)
    manifest, manifest_path, manifest_hash = validate_manifest(config)
    image_inventory = validate_image_inventory(config, manifest)
    require_hash(
        config["data"]["dataset_source"],
        config["data"]["dataset_source_sha256"],
        "E9 retained dataset source",
    )
    require_hash(
        config["model_architecture"]["resnet_source"],
        config["model_architecture"]["resnet_source_sha256"],
        "E9 ResNet source",
    )

    checkpoint_path = resolve_repo_path(config["output"]["checkpoint"])
    metrics_path = resolve_repo_path(config["output"]["training_metrics"])
    metadata_path = resolve_repo_path(config["output"]["run_metadata"])
    require_available_outputs([checkpoint_path, metrics_path, metadata_path], args.overwrite)

    seed = int(config["seed"])
    configure_determinism(seed, bool(config["training"]["deterministic"]))
    device = get_device()
    config_hash = sha256_file(config_path)
    sources = source_hashes()
    run_record = {
        "experiment": "E9",
        "arm": "clahe_resnet50",
        "status": "started",
        "started_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git_commit": current_git_commit(),
        "git_status": current_git_status(),
        "source_sha256": sources,
        "config": str(config_path),
        "config_sha256": config_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "image_inventory": image_inventory,
        "environment": environment_record(device),
        "seed": seed,
        "preprocessing": config["preprocessing"]["clahe"],
    }
    write_json(metadata_path, run_record)

    try:
        print("Device:", device)
        print("Experiment: E9 - CLAHE ResNet-50 + metadata")
        print("Config SHA-256:", config_hash)
        print("Manifest SHA-256:", manifest_hash)

        common = {
            "manifest": str(manifest_path),
            "image_size": int(config["data"]["image_size"]),
            "preprocessing": "clahe",
            "image_root": str(resolve_repo_path(config["data"]["image_root"])),
            "condition": "clean",
            "perturbations": config["perturbations"],
            "seed": seed,
        }
        train_dataset = E9RobustnessDataset(split="train", augment=True, **common)
        validation_dataset = E9RobustnessDataset(split="val", augment=False, **common)
        print("Training samples:", len(train_dataset))
        print("Validation samples:", len(validation_dataset))

        generator = torch.Generator().manual_seed(seed)
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

        model = build_model(config, pretrained=True).to(device)
        model.freeze_backbone()
        positives = int(train_dataset.df["target"].sum())
        negatives = len(train_dataset) - positives

        if positives <= 0 or negatives <= 0:
            raise ValueError("E9 training split must contain both target classes.")

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        epochs = int(config["training"]["epochs"])
        frozen_epochs = int(config["training"]["freeze_backbone_epochs"])
        monitor_threshold = float(config["training"]["validation_monitor_threshold"])
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
                train_loader, desc=f"E9 CLAHE ResNet Epoch {epoch + 1}/{epochs}"
            ):
                images = images.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images, metadata), targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())

            metrics = evaluate(model, validation_loader, device, monitor_threshold)
            record = {
                "experiment": "E9",
                "arm": "clahe_resnet50",
                "epoch": epoch + 1,
                "seed": seed,
                "training_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "train_loss": running_loss / len(train_loader),
                **metrics,
            }
            history.append(record)
            pd.DataFrame(history).to_csv(metrics_path, index=False)
            print(
                f"Epoch {epoch + 1}: loss={record['train_loss']:.4f}, "
                f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
                f"pr_auc={metrics['pr_auc']:.4f}"
            )

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "experiment": "E9",
                        "arm": "clahe_resnet50",
                        "model": "Multi-modal ResNet-50 + metadata, CLAHE",
                        "preprocessing": config["preprocessing"]["clahe"],
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_auc": best_auc,
                        "git_commit": run_record["git_commit"],
                        "git_status": run_record["git_status"],
                        "source_sha256": sources,
                        "config_sha256": config_hash,
                        "manifest_sha256": manifest_hash,
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

        if best_epoch is None or not checkpoint_path.is_file():
            raise RuntimeError("E9 training completed without a valid checkpoint.")

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
        print("\nE9 CLAHE ResNet-50 training complete.")
        print("Best epoch:", best_epoch)
        print("Best validation ROC-AUC:", best_auc)
        print("Checkpoint:", checkpoint_path)

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
