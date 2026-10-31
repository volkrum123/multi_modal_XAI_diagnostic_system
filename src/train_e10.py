import argparse
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.e10_protocol import (
    ARMS,
    FEEDBACK_COLUMNS,
    build_model,
    configure_adaptation_layers,
    load_checkpoint_model,
    load_e10_config,
    make_dataset,
    output_path,
    require_outputs_absent,
    source_hashes,
    validate_manifest,
    validate_partition_outputs,
    validate_static_inputs,
    write_json_atomic,
)
from src.train_e7 import (
    calculate_metrics,
    configure_determinism,
    current_git_commit,
    current_git_status,
    environment_record,
    sha256_file,
    utc_now,
)
from src.utils import get_device


ARM_SETTINGS = {
    "pre": {
        "manifest": "pre_manifest",
        "checkpoint": "pre_checkpoint",
        "metrics": "pre_training_metrics",
        "metadata": "pre_run_metadata",
        "training": "pre_training",
        "description": "Pre-feedback DenseNet-121",
    },
    "control": {
        "manifest": "control_manifest",
        "checkpoint": "control_checkpoint",
        "metrics": "control_training_metrics",
        "metadata": "control_run_metadata",
        "training": "adaptation_training",
        "description": "No-feedback continued-training control",
    },
    "feedback": {
        "manifest": "feedback_manifest",
        "checkpoint": "feedback_checkpoint",
        "metrics": "feedback_training_metrics",
        "metadata": "feedback_run_metadata",
        "training": "adaptation_training",
        "description": "Simulated-feedback adapted DenseNet-121",
    },
}


def _require_pre_checkpoint(config):
    import json

    checkpoint_path = output_path(config, "pre_checkpoint")
    metadata_path = output_path(config, "pre_run_metadata")
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Complete E10 pre-feedback training is required first.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete":
        raise ValueError("E10 pre-feedback training metadata is not complete.")
    if metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("E10 pre-feedback checkpoint differs from its training record.")
    return checkpoint_path, metadata


def _require_feedback_lock(config, pre_checkpoint_path):
    import json

    lock_path = output_path(config, "feedback_lock")
    feedback_path = output_path(config, "simulated_feedback")
    if not lock_path.is_file() or not feedback_path.is_file():
        raise FileNotFoundError("Run evaluate_e10 --stage feedback before the feedback arm.")
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("stage") != "simulated_feedback_lock":
        raise ValueError("E10 feedback lock has an unexpected stage.")
    if lock.get("pre_checkpoint_sha256") != sha256_file(pre_checkpoint_path):
        raise ValueError("E10 feedback lock references a different pre-feedback checkpoint.")
    if lock.get("simulated_feedback_sha256") != sha256_file(feedback_path):
        raise ValueError("E10 simulated feedback differs from its lock.")
    feedback = pd.read_csv(feedback_path)
    if list(feedback.columns) != FEEDBACK_COLUMNS or feedback["image_id"].duplicated().any():
        raise ValueError("E10 simulated feedback schema or unique-image contract failed.")
    if set(feedback["feedback_source"]) != {config["feedback"]["source"]}:
        raise ValueError("E10 simulated feedback source is not the approved oracle source.")
    if set(feedback["original_split"]) != {"train"}:
        raise ValueError("E10 simulated feedback contains non-training cases.")
    partition = pd.read_csv(output_path(config, "partition_manifest"))
    pool = partition[partition["e10_role"] == "feedback_pool"].set_index("image_id")
    if set(feedback["image_id"]) != set(pool.index):
        raise ValueError("E10 simulated feedback IDs differ from the locked feedback pool.")
    labels = feedback.set_index("image_id")["reference_label"].astype(int).sort_index()
    if not labels.equals(pool["target"].astype(int).sort_index()):
        raise ValueError("E10 simulated feedback labels differ from the locked manifest labels.")


def _atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _evaluate(model, loader, device, threshold):
    model.eval()
    targets = []
    probabilities = []
    with torch.no_grad():
        for images, metadata, batch_targets, _ in loader:
            logits = model(images.to(device), metadata.to(device))
            targets.extend(batch_targets.numpy().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return calculate_metrics(targets, probabilities, threshold)


def main():
    parser = argparse.ArgumentParser(description="Train one approved E10 arm.")
    parser.add_argument("--config", default="config_e10.yaml")
    parser.add_argument("--arm", required=True, choices=ARMS)
    args = parser.parse_args()

    config, config_path = load_e10_config(args.config)
    validate_static_inputs(config)
    _, _, manifest_sha = validate_manifest(config)
    partition_metadata = validate_partition_outputs(config)

    arm = args.arm
    settings = ARM_SETTINGS[arm]
    training = config[settings["training"]]
    checkpoint_path = output_path(config, settings["checkpoint"])
    metrics_path = output_path(config, settings["metrics"])
    metadata_path = output_path(config, settings["metadata"])
    require_outputs_absent([checkpoint_path, metrics_path, metadata_path])

    pre_checkpoint_path = None
    if arm != "pre":
        pre_checkpoint_path, _ = _require_pre_checkpoint(config)
    if arm == "feedback":
        _require_feedback_lock(config, pre_checkpoint_path)

    seed = int(config["seed"])
    configure_determinism(seed, bool(training["deterministic"]))
    device = get_device()
    config_hash = sha256_file(config_path)
    manifest_path = output_path(config, settings["manifest"])
    source_sha = source_hashes()
    run_record = {
        "experiment": "E10",
        "arm": arm,
        "description": settings["description"],
        "status": "started",
        "started_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "config": str(config_path),
        "config_sha256": config_hash,
        "source_manifest_sha256": manifest_sha,
        "arm_manifest": str(manifest_path),
        "arm_manifest_sha256": sha256_file(manifest_path),
        "partition_metadata_sha256": sha256_file(output_path(config, "partition_metadata")),
        "source_sha256": source_sha,
        "git_commit": current_git_commit(),
        "git_status": current_git_status(),
        "environment": environment_record(device),
        "seed": seed,
    }
    write_json_atomic(metadata_path, run_record)

    try:
        train_dataset = make_dataset(config, manifest_path, "train", augment=True)
        val_dataset = make_dataset(config, manifest_path, "val", augment=False)
        generator = torch.Generator().manual_seed(seed)
        loader_args = {
            "batch_size": int(config["data"]["batch_size"]),
            "num_workers": int(config["data"]["num_workers"]),
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_args)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

        if arm == "pre":
            model = build_model(config, pretrained=True).to(device)
            model.freeze_backbone()
            trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        else:
            model, _ = load_checkpoint_model(config, pre_checkpoint_path, "pre", device)
            trainable_names = configure_adaptation_layers(
                model, config["adaptation_training"]["trainable_prefixes"]
            )

        targets = train_dataset.df["target"].astype(int)
        positives = int(targets.sum())
        negatives = int(len(targets) - positives)
        if positives <= 0 or negatives <= 0:
            raise ValueError("E10 training data must contain both target classes.")
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=device)
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad or arm == "pre"],
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        epochs = int(training["epochs"])
        monitor_threshold = float(training["validation_monitor_threshold"])
        best_auc = float("-inf")
        best_epoch = None
        history = []
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        print("Device:", device)
        print("Experiment: E10 -", settings["description"])
        print("Training samples:", len(train_dataset))
        print("Validation samples:", len(val_dataset))

        for epoch in range(epochs):
            if arm == "pre" and epoch == int(training["freeze_backbone_epochs"]):
                model.unfreeze_backbone()
                trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]

            model.train()
            running_loss = 0.0
            for images, metadata, batch_targets, _ in tqdm(
                train_loader, desc=f"E10 {arm} Epoch {epoch + 1}/{epochs}"
            ):
                images = images.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images, metadata), batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())

            metrics = _evaluate(model, val_loader, device, monitor_threshold)
            row = {
                "experiment": "E10",
                "arm": arm,
                "epoch": epoch + 1,
                "training_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "train_loss": running_loss / len(train_loader),
                **metrics,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(metrics_path, index=False)
            print(
                f"Epoch {epoch + 1}: loss={row['train_loss']:.4f}, "
                f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
                f"pr_auc={metrics['pr_auc']:.4f}"
            )

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_epoch = epoch + 1
                _atomic_torch_save(
                    checkpoint_path,
                    {
                        "experiment": "E10",
                        "arm": arm,
                        "model_state": model.state_dict(),
                        "config_sha256": config_hash,
                        "source_manifest_sha256": manifest_sha,
                        "arm_manifest_sha256": sha256_file(manifest_path),
                        "pre_checkpoint_sha256": (
                            sha256_file(pre_checkpoint_path) if pre_checkpoint_path else None
                        ),
                        "best_epoch": best_epoch,
                        "best_val_auc": best_auc,
                        "trainable_parameters": trainable_names,
                        "seed": seed,
                    },
                )
                print("Saved best checkpoint:", checkpoint_path)

            run_record.update(
                status="running",
                last_completed_epoch=epoch + 1,
                best_epoch=best_epoch,
                best_val_auc=best_auc,
            )
            write_json_atomic(metadata_path, run_record)

        if best_epoch is None or not checkpoint_path.is_file():
            raise RuntimeError("E10 training completed without a checkpoint.")
        run_record.update(
            status="complete",
            completed_at_utc=utc_now(),
            best_epoch=best_epoch,
            best_val_auc=best_auc,
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            training_metrics=str(metrics_path),
            training_metrics_sha256=sha256_file(metrics_path),
            trainable_parameters=trainable_names,
        )
        write_json_atomic(metadata_path, run_record)
        print("E10 training complete.")
        print("Best epoch:", best_epoch)
        print("Best validation ROC-AUC:", best_auc)

    except KeyboardInterrupt as error:
        run_record.update(
            status="interrupted",
            stopped_at_utc=utc_now(),
            error_type=type(error).__name__,
            error="Training was interrupted by the user.",
        )
        write_json_atomic(metadata_path, run_record)
        raise
    except Exception as error:
        run_record.update(
            status="failed",
            stopped_at_utc=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json_atomic(metadata_path, run_record)
        raise


if __name__ == "__main__":
    main()
