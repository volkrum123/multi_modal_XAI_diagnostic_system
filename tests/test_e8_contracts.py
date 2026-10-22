import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluate_e8 import (
    COMPONENT_COLUMNS,
    DEFAULT_CONFIG,
    align_components,
    analyze_thresholds,
    build_component_frame,
    build_ensemble_frame,
    load_densenet_model,
    load_e8_config,
    resolve_repo_path,
    sha256_file,
    validate_config,
    validate_e2_validation_reproduction,
    validate_lock_record,
    validate_manifest,
    validate_static_inputs,
)


class E8ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_path = load_e8_config(DEFAULT_CONFIG)

    def test_configuration_fixes_equal_weights_and_isolates_outputs(self):
        self.assertEqual(self.config["experiment"]["id"], "E8")
        self.assertEqual(
            self.config["fusion"]["method"],
            "equal_weight_probability_average",
        )
        self.assertEqual(float(self.config["fusion"]["densenet_weight"]), 0.5)
        self.assertEqual(float(self.config["fusion"]["resnet_weight"]), 0.5)
        self.assertEqual(
            self.config["threshold_selection"]["split"],
            "val",
        )

        outputs = [Path(value).as_posix() for value in self.config["output"].values()]
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertTrue(all("e8_" in value for value in outputs))

        changed = copy.deepcopy(self.config)
        changed["fusion"]["densenet_weight"] = 0.6

        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_manifest_and_retained_input_hash_contracts(self):
        frame, _, manifest_hash = validate_manifest(self.config)
        _, hashes, _ = validate_static_inputs(self.config, "val")

        self.assertEqual(len(frame), 61792)
        self.assertEqual(
            manifest_hash,
            self.config["data"]["manifest_sha256"],
        )
        self.assertEqual(
            hashes["densenet_checkpoint"],
            self.config["densenet"]["checkpoint_sha256"],
        )
        self.assertEqual(
            hashes["e7_predictions"],
            self.config["resnet"]["val_predictions_sha256"],
        )

    def test_e2_checkpoint_loads_into_approved_architecture(self):
        checkpoint_path = resolve_repo_path(self.config["densenet"]["checkpoint"])
        model, checkpoint = load_densenet_model(
            self.config,
            checkpoint_path,
            torch.device("cpu"),
        )

        self.assertEqual(checkpoint["experiment"], "E2")
        self.assertEqual(len(checkpoint["model_state"]), 735)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            7236001,
        )

    def test_alignment_and_equal_weight_probability_contract(self):
        densenet = build_component_frame(
            "E2",
            "DenseNet",
            "val",
            ["a.png", "b.png"],
            [0, 1],
            [0.2, 0.8],
            0.75,
        )
        resnet = build_component_frame(
            "E7",
            "ResNet",
            "val",
            ["b.png", "a.png"],
            [1, 0],
            [0.6, 0.4],
            0.85,
        )
        aligned = align_components(densenet, resnet, "val")
        ensemble = build_ensemble_frame(aligned, "val", 0.5, 0.5, 0.5)

        self.assertEqual(ensemble["image_id"].tolist(), ["a.png", "b.png"])
        np.testing.assert_allclose(
            ensemble["probability"].to_numpy(),
            np.asarray([0.3, 0.7]),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(ensemble["prediction"].tolist(), [0, 1])

    def test_alignment_rejects_id_or_target_mismatch(self):
        first = build_component_frame(
            "E2", "DenseNet", "val", ["a.png"], [0], [0.2], 0.75
        )
        wrong_id = build_component_frame(
            "E7", "ResNet", "val", ["b.png"], [0], [0.2], 0.85
        )
        wrong_target = build_component_frame(
            "E7", "ResNet", "val", ["a.png"], [1], [0.8], 0.85
        )

        with self.assertRaises(ValueError):
            align_components(first, wrong_id, "val")

        with self.assertRaises(ValueError):
            align_components(first, wrong_target, "val")

    def test_threshold_selection_uses_lowest_threshold_for_f1_tie(self):
        analysis, selected = analyze_thresholds(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
            [0.5, 0.6],
        )

        self.assertEqual(len(analysis), 2)
        self.assertEqual(float(selected["threshold"]), 0.5)
        self.assertEqual(float(selected["f1"]), 1.0)

    def test_historical_e2_validation_contract_uses_threshold_075(self):
        historical = pd.read_csv(
            resolve_repo_path(
                self.config["densenet"]["validation_threshold_analysis"]
            )
        )
        reproduction = historical.copy()
        reproduction["recall"] = reproduction["sensitivity"]
        reproduction["roc_auc"] = self.config["densenet"][
            "best_validation_roc_auc"
        ]
        reproduction["pr_auc"] = 0.0
        reproduction = reproduction[
            [
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
        ]

        validate_e2_validation_reproduction(self.config, reproduction)
        selected = reproduction.loc[reproduction["f1"].idxmax()]
        self.assertEqual(float(selected["threshold"]), 0.75)

    def test_lock_rejects_changed_validation_evidence(self):
        config_hash = sha256_file(self.config_path)

        with tempfile.TemporaryDirectory(
            dir=resolve_repo_path("outputs")
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            val_paths = {}
            record = {
                "experiment": "E8",
                "selection_split": "val",
                "selection_metric": "f1",
                "tie_break": "lowest_threshold",
                "fusion_method": "equal_weight_probability_average",
                "densenet_weight": 0.5,
                "resnet_weight": 0.5,
                "threshold": 0.7,
                "config_sha256": config_hash,
                "manifest_sha256": self.config["data"]["manifest_sha256"],
                "densenet_checkpoint_sha256": self.config["densenet"][
                    "checkpoint_sha256"
                ],
                "resnet_checkpoint_sha256": self.config["resnet"][
                    "checkpoint_sha256"
                ],
                "resnet_validation_predictions_sha256": self.config["resnet"][
                    "val_predictions_sha256"
                ],
            }

            for key in [
                "densenet_val_predictions",
                "densenet_val_metrics",
                "ensemble_val_predictions",
                "ensemble_val_metrics",
                "ensemble_val_threshold_analysis",
            ]:
                path = temporary / f"{key}.csv"
                path.write_text(key, encoding="utf-8")
                val_paths[key] = path

            record.update(
                {
                    "densenet_validation_predictions_sha256": sha256_file(
                        val_paths["densenet_val_predictions"]
                    ),
                    "densenet_validation_metrics_sha256": sha256_file(
                        val_paths["densenet_val_metrics"]
                    ),
                    "ensemble_validation_predictions_sha256": sha256_file(
                        val_paths["ensemble_val_predictions"]
                    ),
                    "ensemble_validation_metrics_sha256": sha256_file(
                        val_paths["ensemble_val_metrics"]
                    ),
                    "ensemble_threshold_analysis_sha256": sha256_file(
                        val_paths["ensemble_val_threshold_analysis"]
                    ),
                }
            )

            threshold = validate_lock_record(
                record,
                self.config,
                config_hash,
                val_paths,
            )
            self.assertEqual(threshold, 0.7)

            val_paths["ensemble_val_metrics"].write_text(
                "changed",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                validate_lock_record(
                    record,
                    self.config,
                    config_hash,
                    val_paths,
                )


if __name__ == "__main__":
    unittest.main()

