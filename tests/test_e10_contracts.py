import unittest
from pathlib import Path

import numpy as np

from src.e10_protocol import (
    FEEDBACK_COLUMNS,
    build_model,
    build_partitions,
    configure_adaptation_layers,
    load_e10_config,
    validate_e4_is_test_only,
    validate_manifest,
    validate_static_inputs,
)
from src.evaluate_e10 import analyze_thresholds


class E10ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_e10_config("config_e10.yaml")
        cls.manifest, _, _ = validate_manifest(cls.config)
        (
            cls.partition,
            cls.pre_manifest,
            cls.control_manifest,
            cls.feedback_manifest,
            cls.counts,
        ) = build_partitions(cls.config, cls.manifest)

    def test_configuration_locks_protocol_and_isolates_outputs(self):
        self.assertEqual(self.config["experiment"]["id"], "E10")
        self.assertEqual(self.config["model"]["backbone"], "densenet121")
        self.assertEqual(self.config["model"]["preprocessing"], "baseline")
        self.assertEqual(self.config["pre_training"]["epochs"], 5)
        self.assertEqual(self.config["adaptation_training"]["epochs"], 3)
        self.assertEqual(self.config["bootstrap"]["resamples"], 1000)
        paths = [Path(value).as_posix() for value in self.config["output"].values()]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(path.startswith("outputs/") and "e10_" in path for path in paths))

    def test_retained_inputs_and_e4_test_only_contract(self):
        validated = validate_static_inputs(self.config)
        self.assertEqual(len(validated), 7)
        validate_e4_is_test_only(self.config, self.manifest)

    def test_patient_partition_and_exact_counts(self):
        expected = {key: int(value) for key, value in self.config["partition"]["expected"].items()}
        self.assertEqual(self.counts, expected)
        base_patients = set(
            self.partition.loc[self.partition["e10_role"] == "base_train", "Patient ID"]
        )
        feedback_patients = set(
            self.partition.loc[self.partition["e10_role"] == "feedback_pool", "Patient ID"]
        )
        self.assertTrue(base_patients.isdisjoint(feedback_patients))
        self.assertEqual(len(feedback_patients), expected["feedback_patients"])

    def test_training_manifests_preserve_held_out_splits(self):
        original_held_out = self.manifest[self.manifest["split"].isin(["val", "test"])].sort_values("image_id")
        for frame in (self.pre_manifest, self.control_manifest, self.feedback_manifest):
            held_out = frame[frame["split"].isin(["val", "test"])].sort_values("image_id")
            self.assertTrue(np.array_equal(original_held_out["image_id"], held_out["image_id"]))
            self.assertTrue(np.array_equal(original_held_out["target"], held_out["target"]))

    def test_continuation_arms_are_balanced_and_disjoint_as_approved(self):
        control = self.control_manifest[self.control_manifest["split"] == "train"]
        feedback = self.feedback_manifest[self.feedback_manifest["split"] == "train"]
        expected = self.config["partition"]["expected"]
        self.assertEqual(len(control), 2 * int(expected["replay_images_per_set"]))
        self.assertEqual(len(feedback), int(expected["feedback_images"]) + int(expected["replay_images_per_set"]))
        self.assertEqual(int(control["target"].sum()), 2 * int(expected["replay_positives_per_set"]))
        self.assertEqual(int(feedback["target"].sum()), int(expected["feedback_positives"]) + int(expected["replay_positives_per_set"]))
        feedback_pool_ids = set(
            self.partition.loc[self.partition["e10_role"] == "feedback_pool", "image_id"]
        )
        self.assertTrue(feedback_pool_ids.isdisjoint(set(control["image_id"])))
        self.assertTrue(feedback_pool_ids.issubset(set(feedback["image_id"])))

    def test_adaptation_trainable_parameter_contract(self):
        model = build_model(self.config, pretrained=False)
        prefixes = self.config["adaptation_training"]["trainable_prefixes"]
        trainable = configure_adaptation_layers(model, prefixes)
        self.assertTrue(trainable)
        self.assertTrue(all(any(name.startswith(prefix) for prefix in prefixes) for name in trainable))
        early = [
            name
            for name, parameter in model.named_parameters()
            if name.startswith("image_encoder.features.denseblock1") and parameter.requires_grad
        ]
        self.assertEqual(early, [])

    def test_threshold_tie_uses_lowest_threshold(self):
        targets = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        analysis, selected = analyze_thresholds(targets, probabilities, [0.3, 0.4, 0.5])
        self.assertTrue((analysis["f1"] == 1.0).all())
        self.assertEqual(float(selected["threshold"]), 0.3)

    def test_simulated_feedback_schema_is_explicit(self):
        self.assertEqual(self.config["feedback"]["source"], "simulated_oracle_manifest_label")
        self.assertIn("feedback_source", FEEDBACK_COLUMNS)
        self.assertIn("reference_label", FEEDBACK_COLUMNS)
        self.assertIn("reviewer_decision", FEEDBACK_COLUMNS)


if __name__ == "__main__":
    unittest.main()
