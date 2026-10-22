import copy
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.e9_dataset import E9RobustnessDataset
from src.evaluate_e9 import (
    PERTURBED_CONDITIONS,
    align_runs,
    analyze_thresholds,
    condition_paths,
    cosine_and_drift,
    forward_with_features,
    validate_static_inputs,
    verify_duplicate_densenet_states,
)
from src.model import MultiModalPneumoniaModel
from src.resnet_model import MultiModalResNet50
from src.train_e9 import (
    DEFAULT_CONFIG,
    EXPECTED_CONDITIONS,
    load_e9_config,
    resolve_repo_path,
    validate_config,
    validate_manifest,
)


class E9ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_path = load_e9_config(DEFAULT_CONFIG)

    def test_configuration_locks_protocol_and_isolates_outputs(self):
        self.assertEqual(self.config["experiment"]["id"], "E9")
        self.assertEqual(set(self.config["perturbations"]), EXPECTED_CONDITIONS)
        self.assertEqual(PERTURBED_CONDITIONS, [
            "brightness_low",
            "brightness_high",
            "contrast_low",
            "contrast_high",
            "gamma_low",
            "gamma_high",
            "gaussian_noise",
            "gaussian_blur",
        ])
        self.assertEqual(int(self.config["training"]["epochs"]), 5)
        self.assertEqual(float(self.config["fusion"]["densenet_weight"]), 0.5)
        self.assertEqual(float(self.config["fusion"]["resnet_weight"]), 0.5)
        self.assertEqual(
            self.config["feature_stability"]["layer"],
            "image_projection_post_relu",
        )

        outputs = [Path(value).as_posix() for value in self.config["output"].values()]
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertTrue(all(value.startswith("outputs/") for value in outputs))
        self.assertTrue(all("e9_" in value for value in outputs))

        changed = copy.deepcopy(self.config)
        changed["training"]["epochs"] = 6
        with self.assertRaises(ValueError):
            validate_config(changed)

        changed = copy.deepcopy(self.config)
        changed["fusion"]["densenet_weight"] = 0.6
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_manifest_and_retained_input_hash_contracts(self):
        frame, _, manifest_hash = validate_manifest(self.config)
        paths, hashes = validate_static_inputs(self.config, include_e9_checkpoint=False)

        self.assertEqual(len(frame), 61792)
        self.assertEqual(manifest_hash, self.config["data"]["manifest_sha256"])
        self.assertEqual(
            hashes["baseline_densenet"],
            self.config["models"]["baseline_densenet"]["checkpoint_sha256"],
        )
        self.assertEqual(
            hashes["clahe_densenet"],
            self.config["models"]["clahe_densenet"]["checkpoint_sha256"],
        )
        self.assertEqual(
            hashes["baseline_resnet"],
            self.config["models"]["baseline_resnet"]["checkpoint_sha256"],
        )
        verify_duplicate_densenet_states(
            paths["baseline_densenet"], paths["duplicate_densenet"]
        )

    def test_e6b_checkpoint_is_distinct_and_finite(self):
        baseline = torch.load(
            resolve_repo_path(self.config["models"]["baseline_densenet"]["checkpoint"]),
            map_location="cpu",
        )["model_state"]
        clahe = torch.load(
            resolve_repo_path(self.config["models"]["clahe_densenet"]["checkpoint"]),
            map_location="cpu",
        )["model_state"]
        self.assertEqual(list(baseline), list(clahe))
        self.assertTrue(
            all(torch.isfinite(value).all().item() for value in clahe.values())
        )
        self.assertTrue(any(not torch.equal(baseline[key], clahe[key]) for key in baseline))

    def test_perturbations_are_deterministic_and_have_expected_direction(self):
        base = np.full((32, 32, 3), 128, dtype=np.uint8)
        base[:, :16] = 64
        image = Image.fromarray(base)

        dataset = E9RobustnessDataset.__new__(E9RobustnessDataset)
        dataset.seed = 42
        dataset.perturbation = self.config["perturbations"]["gaussian_noise"]
        first = np.asarray(dataset.apply_perturbation(image, "sample.png"))
        second = np.asarray(dataset.apply_perturbation(image, "sample.png"))
        third = np.asarray(dataset.apply_perturbation(image, "other.png"))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

        dataset.perturbation = self.config["perturbations"]["brightness_low"]
        darker = np.asarray(dataset.apply_perturbation(image, "sample.png"))
        dataset.perturbation = self.config["perturbations"]["brightness_high"]
        brighter = np.asarray(dataset.apply_perturbation(image, "sample.png"))
        self.assertLess(float(darker.mean()), float(base.mean()))
        self.assertGreater(float(brighter.mean()), float(base.mean()))

    def test_perturbation_precedes_clahe(self):
        dataset = E9RobustnessDataset.__new__(E9RobustnessDataset)
        dataset.preprocessing = "clahe"
        calls = []

        def perturb(image, image_id):
            calls.append(("perturb", image_id))
            return image

        def clahe(image):
            calls.append(("clahe", None))
            return image

        dataset.apply_perturbation = perturb
        dataset.apply_clahe = clahe
        dataset.prepare_image(Image.new("RGB", (4, 4)), "x.png")
        self.assertEqual(calls, [("perturb", "x.png"), ("clahe", None)])

    def test_forward_with_features_matches_model_forward(self):
        torch.manual_seed(1)
        images = torch.randn(2, 3, 64, 64)
        metadata = torch.randn(2, 3)

        for model in [
            MultiModalPneumoniaModel(pretrained=False),
            MultiModalResNet50(pretrained=False),
        ]:
            model.eval()
            with torch.no_grad():
                direct = model(images, metadata)
                logits, features = forward_with_features(model, images, metadata)
            torch.testing.assert_close(logits, direct, rtol=0.0, atol=0.0)
            self.assertEqual(tuple(features.shape), (2, 256))
            self.assertTrue(torch.all(features >= 0).item())

    def test_threshold_tie_and_equal_weight_alignment(self):
        analysis, selected = analyze_thresholds(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], [0.5, 0.6]
        )
        self.assertEqual(len(analysis), 2)
        self.assertEqual(float(selected["threshold"]), 0.5)

        baseline = {
            "image_id": np.asarray(["b.png", "a.png"]),
            "target": np.asarray([1, 0]),
            "first_probability": np.asarray([0.8, 0.2]),
            "second_probability": np.asarray([0.6, 0.4]),
            "first_feature": np.asarray([[2.0], [1.0]]),
            "second_feature": np.asarray([[4.0], [3.0]]),
        }
        clahe = {
            "image_id": np.asarray(["a.png", "b.png"]),
            "target": np.asarray([0, 1]),
            "first_probability": np.asarray([0.3, 0.7]),
            "second_probability": np.asarray([0.5, 0.5]),
            "first_feature": np.asarray([[1.0], [2.0]]),
            "second_feature": np.asarray([[3.0], [4.0]]),
        }
        aligned = align_runs(baseline, clahe)
        np.testing.assert_array_equal(aligned["baseline_image_id"], ["a.png", "b.png"])
        ensemble = (
            aligned["baseline_first_probability"]
            + aligned["baseline_second_probability"]
        ) / 2.0
        np.testing.assert_allclose(ensemble, [0.3, 0.7], rtol=0.0, atol=1e-15)

    def test_feature_stability_known_values(self):
        clean = np.asarray([[1.0, 0.0], [1.0, 1.0]])
        same = clean.copy()
        cosine, drift = cosine_and_drift(clean, same, 1e-12)
        np.testing.assert_allclose(cosine, [1.0, 1.0], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(drift, [0.0, 0.0], rtol=0.0, atol=1e-15)

        changed = np.asarray([[0.0, 1.0], [2.0, 2.0]])
        cosine, drift = cosine_and_drift(clean, changed, 1e-12)
        np.testing.assert_allclose(cosine, [0.0, 1.0], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(drift, [np.sqrt(2.0), 1.0], rtol=0.0, atol=1e-15)

    def test_condition_outputs_are_isolated(self):
        seen = set()
        for condition in ["clean", *PERTURBED_CONDITIONS]:
            paths = condition_paths(self.config, condition)
            self.assertEqual(paths["root"].name, condition)
            for key in ["predictions", "metrics", "features", "metadata"]:
                self.assertNotIn(paths[key], seen)
                seen.add(paths[key])
                self.assertTrue("e9_conditions" in paths[key].as_posix())


if __name__ == "__main__":
    unittest.main()
