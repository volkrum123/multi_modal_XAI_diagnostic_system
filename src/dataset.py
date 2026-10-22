from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class NIHPneumoniaDataset(Dataset):
    def __init__(
        self,
        manifest,
        split,
        image_size=224,
        augment=False,
        preprocessing="baseline",
        image_root=None,
    ):
        df = pd.read_csv(manifest)

        self.df = df[
            df["split"] == split
        ].reset_index(drop=True)

        self.preprocessing = preprocessing
        self.image_root = (
            Path(image_root).resolve()
            if image_root is not None
            else None
        )
        self._resolved_paths = {}

        if self.image_root is not None and not self.image_root.is_dir():
            raise FileNotFoundError(
                f"Image root not found: {self.image_root}"
            )

        if preprocessing not in [
            "baseline",
            "clahe",
        ]:
            raise ValueError(
                f"Unsupported preprocessing: {preprocessing}"
            )

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])

        # CLAHE configuration.
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

    def __len__(self):
        return len(self.df)

    def resolve_image_path(self, manifest_path):
        """Resolve a manifest path, optionally against a portable image root."""

        cache_key = str(manifest_path)

        if cache_key in self._resolved_paths:
            return self._resolved_paths[cache_key]

        path = Path(cache_key)

        if path.is_file():
            resolved = path

        else:
            if self.image_root is not None:
                direct_candidate = self.image_root / path.name
                candidates = (
                    [direct_candidate]
                    if direct_candidate.is_file()
                    else list(
                        self.image_root.glob(
                            f"*/{path.name}"
                        )
                    )
                )
            else:
                candidates = list(
                    path.parent.glob(
                        f"*/{path.name}"
                    )
                )

            candidates = [
                candidate
                for candidate in candidates
                if candidate.is_file()
            ]

            if not candidates:
                raise FileNotFoundError(
                    f"Image not found: {manifest_path}"
                )

            if len(candidates) > 1:
                raise RuntimeError(
                    "Image filename is not unique under the configured root: "
                    f"{path.name}"
                )

            resolved = candidates[0]

        self._resolved_paths[cache_key] = resolved
        return resolved

    def apply_clahe(self, image):
        """
        Apply CLAHE to the luminance channel
        of a chest X-ray image.
        """

        image_np = np.array(
            image.convert("RGB")
        )

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY,
        )

        enhanced = self.clahe.apply(gray)

        enhanced_rgb = cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2RGB,
        )

        return Image.fromarray(
            enhanced_rgb
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = self.resolve_image_path(
            row["image_path"]
        )

        image = Image.open(
            path
        ).convert("RGB")

        # Apply preprocessing before
        # the standard DenseNet transforms.
        if self.preprocessing == "clahe":
            image = self.apply_clahe(
                image
            )

        image = self.transform(
            image
        )

        # Structured modality:
        # age, sex, view.
        metadata = torch.tensor(
            [
                float(row["age"]) / 100.0,
                float(row["sex"]),
                float(row["view"]),
            ],
            dtype=torch.float32,
        )

        target = torch.tensor(
            float(row["target"]),
            dtype=torch.float32,
        )

        return (
            image,
            metadata,
            target,
            row["image_path"],
        )
