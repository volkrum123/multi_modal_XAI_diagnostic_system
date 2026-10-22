import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.dataset import NIHPneumoniaDataset


class E9RobustnessDataset(NIHPneumoniaDataset):
    """NIH dataset with deterministic perturbation before preprocessing."""

    def __init__(
        self,
        manifest,
        split,
        image_size,
        augment,
        preprocessing,
        image_root,
        condition,
        perturbations,
        seed=42,
    ):
        super().__init__(
            manifest=manifest,
            split=split,
            image_size=image_size,
            augment=augment,
            preprocessing=preprocessing,
            image_root=image_root,
        )

        if condition not in perturbations:
            raise ValueError(f"Unknown E9 perturbation condition: {condition}")

        self.condition = condition
        self.perturbation = dict(perturbations[condition])
        self.seed = int(seed)

    def _noise_seed(self, image_id):
        payload = f"{self.seed}:{Path(str(image_id)).name}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def apply_perturbation(self, image, image_id):
        kind = self.perturbation["kind"]

        if kind == "clean":
            return image.copy()

        if kind == "brightness":
            return ImageEnhance.Brightness(image).enhance(
                float(self.perturbation["factor"])
            )

        if kind == "contrast":
            return ImageEnhance.Contrast(image).enhance(
                float(self.perturbation["factor"])
            )

        if kind == "gamma":
            gamma = float(self.perturbation["gamma"])
            values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            adjusted = np.power(values, gamma)
            adjusted = np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)
            return Image.fromarray(adjusted, mode="RGB")

        if kind == "gaussian_noise":
            standard_deviation = float(
                self.perturbation["standard_deviation"]
            )
            values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            generator = np.random.default_rng(self._noise_seed(image_id))
            noise = generator.normal(
                loc=0.0,
                scale=standard_deviation,
                size=values.shape,
            )
            adjusted = np.clip(values + noise, 0.0, 1.0)
            adjusted = np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)
            return Image.fromarray(adjusted, mode="RGB")

        if kind == "gaussian_blur":
            return image.filter(
                ImageFilter.GaussianBlur(radius=float(self.perturbation["radius"]))
            )

        raise ValueError(f"Unsupported E9 perturbation kind: {kind}")

    def prepare_image(self, image, image_id):
        image = self.apply_perturbation(image.convert("RGB"), image_id)

        if self.preprocessing == "clahe":
            image = self.apply_clahe(image)

        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.resolve_image_path(row["image_path"])

        with Image.open(path) as source:
            image = self.prepare_image(source.convert("RGB"), row["image_path"])

        image = self.transform(image)
        metadata = torch.tensor(
            [
                float(row["age"]) / 100.0,
                float(row["sex"]),
                float(row["view"]),
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(float(row["target"]), dtype=torch.float32)

        return image, metadata, target, row["image_path"]

