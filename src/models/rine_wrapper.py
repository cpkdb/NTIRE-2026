import importlib.util
from pathlib import Path
import torch
import torch.nn as nn

_rine_src = Path(__file__).resolve().parents[2] / "baselines" / "rine" / "src" / "models.py"
_spec = importlib.util.spec_from_file_location("rine_models", str(_rine_src))
_rine_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rine_mod)
_RINEModel = _rine_mod.Model

from .base_classifier import BaseClassifier

_BACKBONE_DIM = {"ViT-L/14": 1024, "ViT-B/16": 768, "ViT-B/32": 768}


class RINEClassifier(BaseClassifier):
    def __init__(self, backbone="ViT-L/14", nproj=3, proj_dim=256, device="cuda",
                 download_root="/root/autodl-tmp/clip_models", num_hooks=None):
        super().__init__()
        dim = _BACKBONE_DIM.get(backbone, 768)
        self.rine = _RINEModel(
            backbone=[backbone, dim], nproj=nproj, proj_dim=proj_dim, device=device,
            download_root=download_root, num_hooks=num_hooks,
        )

    def forward(self, x):
        p, _ = self.rine(x)
        return p.squeeze(-1)

    def get_features(self, x):
        p, z = self.rine(x)
        return p.squeeze(-1), z

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_trainable(self, path):
        state = {k: v for k, v in self.state_dict().items() if "clip" not in k}
        torch.save(state, path)

    def load_trainable(self, path, device="cpu"):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state, strict=False)
