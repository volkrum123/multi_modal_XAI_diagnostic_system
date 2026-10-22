import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluate_e7 import (
    PREDICTION_COLUMNS,
    analyze_thresholds,
    build_prediction_frame,
    load_locked_threshold,
)
from src.resnet_model import MultiModalResNet50
from src.dataset import NIHPneumoniaDataset
from src.train_e7 import (
    DEFAULT_CONFIG,
    LEGACY_OUTPUTS,
    load_e7_config,
    resolve_repo_path,
    validate_image_inventory,
    validate_manifest,
)


class E7ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_path = load_e7_config(DEFAULT_CONFIG)

    def test_configuration_is_e7_and_outputs_are_isolated(self):
        self.assertEqual(self.config["experiment"]["id"], "E7")
        self.assertEqual(self.config["model"]["backbone"], "resnet50")
        self.assertEqual(self.config["data"]["preprocessing"], "baseline")
        self.assertFalse(Path(self.config["data"]["image_root"]).is_absolute())

        output_paths = {
            Path(value).as_posix()
            for value in self.config["output"].values()
        }

        self.assertEqual(len(output_paths), len(self.config["output"]))
        self.assertTrue(output_paths.isdisjoint(LEGACY_OUTPUTS))

    def test_manifest_identity_and_patient_split_contract(self):
        _, summary = validate_manifest(self.config)

        self.assertEqual(summary["rows"], 61792)
        self.assertEqual(summary["patients"], 25052)
        self.assertEqual(
            summary["split_counts"],
            {"test": 9326, "train": 43020, "val": 9446},
        )
        self.assertEqual(
            summary["positive_counts"],
            {"test": 228, "train": 995, "val": 208},
        )

    def test_repository_image_root_resolves_inventory_and_real_sample(self):
        frame, _ = validate_manifest(self.config)
        inventory = validate_image_inventory(self.config, frame)

        self.assertEqual(inventory["inventory_png_files"], 112120)
        self.assertEqual(inventory["manifest_images_resolved"], 61792)
        self.assertEqual(inventory["missing_manifest_images"], 0)
        self.assertEqual(inventory["ambiguous_manifest_images"], 0)

        dataset = NIHPneumoniaDataset(
            str(resolve_repo_path(self.config["data"]["manifest"])),
            "train",
            int(self.config["data"]["image_size"]),
            augment=False,
            preprocessing=self.config["data"]["preprocessing"],
            image_root=str(resolve_repo_path(self.config["data"]["image_root"])),
        )
        image, metadata, target, manifest_path = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(tuple(metadata.shape), (3,))
        self.assertEqual(tuple(target.shape), ())
        self.assertTrue(
            dataset.resolve_image_path(manifest_path).is_file()
        )

    def test_resnet_forward_and_backbone_freeze_contract(self):
        model = MultiModalResNet50(
            pretrained=False,
            metadata_dim=3,
            image_embedding_dim=256,
            metadata_embedding_dim=32,
            dropout=0.30,
        )
        model.eval()

        images = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        metadata = torch.zeros((2, 3), dtype=torch.float32)

        with torch.no_grad():
            logits = model(images, metadata)

        self.assertEqual(tuple(logits.shape), (2,))

        model.freeze_backbone()
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.image_encoder.parameters()
            )
        )

        model.unfreeze_backbone()
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.image_encoder.parameters()
            )
        )

    def test_threshold_selection_uses_lowest_threshold_for_f1_tie(self):
        targets = [0, 0, 1, 1]
        probabilities = [0.10, 0.40, 0.60, 0.90]

        analysis, selected = analyze_thresholds(
            targets,
            probabilities,
            [0.50, 0.60],
        )

        self.assertEqual(len(analysis), 2)
        self.assertEqual(float(selected["threshold"]), 0.50)
        self.assertEqual(float(selected["f1"]), 1.0)

    def test_prediction_export_schema(self):
        frame = build_prediction_frame(
            image_ids=["image-a.png", "image-b.png"],
            targets=[0, 1],
            probabilities=[0.20, 0.80],
            split="val",
            threshold=0.50,
        )

        self.assertEqual(frame.columns.tolist(), PREDICTION_COLUMNS)
        self.assertEqual(frame["prediction"].tolist(), [0, 1])
        self.assertFalse(frame["image_id"].duplicated().any())

    def test_locked_threshold_requires_matching_evidence_hashes(self):
        expected = {
            "experiment": "E7",
            "selection_split": "val",
            "selection_metric": "f1",
            "config_sha256": "CONFIG",
            "manifest_sha256": "MANIFEST",
            "checkpoint_sha256": "CHECKPOINT",
            "threshold": 0.65,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "locked.json"
            lock_path.write_text(
                json.dumps(expected),
                encoding="utf-8",
            )

            threshold, record = load_locked_threshold(
                lock_path,
                "CONFIG",
                "MANIFEST",
                "CHECKPOINT",
            )

            self.assertEqual(threshold, 0.65)
            self.assertEqual(record["experiment"], "E7")

            with self.assertRaises(ValueError):
                load_locked_threshold(
                    lock_path,
                    "DIFFERENT-CONFIG",
                    "MANIFEST",
                    "CHECKPOINT",
                )


if __name__ == "__main__":
    unittest.main()

