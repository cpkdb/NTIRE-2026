import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_classifier import BaseClassifier


class _Hook:
    __slots__ = ("output", "_handle")

    def __init__(self, module):
        self.output = None
        self._handle = module.register_forward_hook(self._fn)

    def _fn(self, _module, _input, output):
        self.output = output

    def close(self):
        self._handle.remove()


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=8):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.detach(), requires_grad=False) if linear.bias is not None else None
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(x, self.lora_A), self.lora_B)


class DINOv3Classifier(BaseClassifier):

    def __init__(self, nproj=3, proj_dim=256, num_hooks=None,
                 model_path="/root/autodl-tmp/dinov3_model",
                 lora_layers=0, lora_rank=8):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._lora_params = []
        if lora_layers > 0:
            self._inject_lora(lora_layers, lora_rank)

        dim = self.backbone.config.hidden_size  # 1024

        # Hook norm2 in each transformer layer
        all_hooks = [_Hook(layer.norm2) for layer in self.backbone.layer]
        if num_hooks is not None and 0 < num_hooks < len(all_hooks):
            for h in all_hooks[:-num_hooks]:
                h.close()
            all_hooks = all_hooks[-num_hooks:]
        self.hooks = all_hooks
        n_hooks = len(self.hooks)

        self.alpha = nn.Parameter(torch.randn(1, n_hooks, proj_dim))

        layers1 = [nn.Dropout()]
        for i in range(nproj):
            layers1 += [nn.Linear(dim if i == 0 else proj_dim, proj_dim), nn.ReLU(), nn.Dropout()]
        self.proj1 = nn.Sequential(*layers1)

        layers2 = [nn.Dropout()]
        for _ in range(nproj):
            layers2 += [nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout()]
        self.proj2 = nn.Sequential(*layers2)

        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(),
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(),
            nn.Linear(proj_dim, 1),
        )

    def _inject_lora(self, n_layers, rank):
        layers = list(self.backbone.layer)
        for layer in layers[-min(len(layers), n_layers):]:
            attn = layer.attention
            for name in ("q_proj", "v_proj"):
                linear = getattr(attn, name, None)
                if isinstance(linear, nn.Linear):
                    lora = LoRALinear(linear, rank=rank)
                    setattr(attn, name, lora)
                    self._lora_params.extend([lora.lora_A, lora.lora_B])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract(self, x):
        if self._lora_params:
            self.backbone(x)
        else:
            with torch.no_grad():
                self.backbone(x)
        g = torch.stack([h.output[:, 0, :] for h in self.hooks], dim=1)
        return g

    def forward(self, x):
        g = self.proj1(self._extract(x).float())
        z = (torch.softmax(self.alpha, dim=1) * g).sum(dim=1)
        z = self.proj2(z)
        return self.head(z).squeeze(-1)

    def get_features(self, x):
        g = self.proj1(self._extract(x).float())
        z = (torch.softmax(self.alpha, dim=1) * g).sum(dim=1)
        z = self.proj2(z)
        return self.head(z).squeeze(-1), z

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def trainable_params(self):
        seen = set()
        params = []
        for p in list(self.proj1.parameters()) + list(self.proj2.parameters()) + \
                 list(self.head.parameters()) + [self.alpha] + self._lora_params:
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
        return params

    def save_trainable(self, path):
        state = {}
        for k, v in self.state_dict().items():
            if not k.startswith("backbone.") or ".lora_" in k:
                state[k] = v
        torch.save(state, path)

    def load_trainable(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device), strict=False)
