from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class Config:
    data_root: str = "/data"
    shards: List[int] = field(default_factory=lambda: list(range(6)))
    model_name: str = "resnet50"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    val_ratio: float = 0.1
    output_dir: str = "/workspace/experiments"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
