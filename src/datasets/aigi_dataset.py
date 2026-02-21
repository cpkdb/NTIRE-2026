from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class AIGIDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        shards: Optional[List[int]] = None,
        transform=None,
        is_test: bool = False,
        val_dir: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.is_test = is_test
        self.samples: List[Tuple[Path, Optional[int]]] = []

        if val_dir is not None:
            img_dir = Path(val_dir)
            for img_path in sorted(img_dir.glob("*.jpg")):
                self.samples.append((img_path, None))
            self.is_test = True
            print(f"Found {len(self.samples)} validation images from {val_dir}")
            return

        if shards is None:
            shards = list(range(6))

        for shard_idx in shards:
            shard_dir = self._find_shard_dir(shard_idx)
            if shard_dir is None:
                continue

            images_dir = shard_dir / "images"
            labels_file = shard_dir / "labels.csv"

            if is_test or not labels_file.exists():
                for img_path in sorted(images_dir.glob("*.jpg")):
                    self.samples.append((img_path, None))
            else:
                df = pd.read_csv(labels_file, index_col=0)
                for _, row in df.iterrows():
                    img_path = images_dir / row["image_name"]
                    if img_path.exists():
                        self.samples.append((img_path, int(row["label"])))

        print(f"Found {len(self.samples)} images from shards {shards}")

    def _find_shard_dir(self, shard_idx: int) -> Optional[Path]:
        """Handle both flat (shard_X/images/) and nested (shard_X/shard_X/images/) layouts."""
        flat = self.data_root / f"shard_{shard_idx}"
        if not flat.exists():
            return None
        nested = flat / f"shard_{shard_idx}"
        if nested.exists() and (nested / "images").exists():
            return nested
        if (flat / "images").exists():
            return flat
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_path.name
        return image, torch.tensor(label, dtype=torch.long)


class ToyDataset(Dataset):
    def __init__(self, data_root: str, transform=None):
        self.data_root = Path(data_root) / "toy_dataset" / "images"
        self.transform = transform
        self.samples = sorted(self.data_root.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path.name
