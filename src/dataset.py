from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class NIHPneumoniaDataset(Dataset):
    def __init__(self, manifest, split, image_size=224, augment=False):
        df = pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["image_path"])
        if not path.exists():
            # Support the common NIH layout where images are inside numbered folders.
            candidates = list(path.parent.glob(f"*/{path.name}"))
            if candidates:
                path = candidates[0]
            else:
                raise FileNotFoundError(f"Image not found: {row['image_path']}")

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        # Structured modality: age, sex, view.
        metadata = torch.tensor([
            float(row["age"]) / 100.0,
            float(row["sex"]),
            float(row["view"]),
        ], dtype=torch.float32)

        target = torch.tensor(float(row["target"]), dtype=torch.float32)
        return image, metadata, target, row["image_path"]
