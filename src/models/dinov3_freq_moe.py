import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_wrapper import DINOv3Classifier

_NORM_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class DINOv3FreqMoE(DINOv3Classifier):

    def __init__(self, num_experts=3, nproj=3, proj_dim=256, num_hooks=None,
                 model_path="/root/autodl-tmp/dinov3_model",
                 lora_layers=0, lora_rank=8):
        super().__init__(nproj=nproj, proj_dim=proj_dim, num_hooks=num_hooks,
                         model_path=model_path, lora_layers=lora_layers,
                         lora_rank=lora_rank)

        mean, std = _NORM_IMAGENET
        self.register_buffer("_freq_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("_freq_std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("_haar", self._make_haar())

        self.freq_enc = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d(1)

        self.cls_router_proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.router = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_experts),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.tau = nn.Parameter(torch.ones(1))
        self._use_gumbel = False
        self._gumbel_tau = 1.0

        self.experts = nn.ModuleList([self._make_expert(proj_dim) for _ in range(num_experts)])

        del self.head

        self._num_experts = num_experts

    @staticmethod
    def _make_haar():
        ll = torch.tensor([[.5, .5], [.5, .5]])
        lh = torch.tensor([[-.5, -.5], [.5, .5]])
        hl = torch.tensor([[-.5, .5], [-.5, .5]])
        hh = torch.tensor([[.5, -.5], [-.5, .5]])
        return torch.stack([ll, lh, hl, hh]).unsqueeze(1).repeat(3, 1, 1, 1)

    @staticmethod
    def _make_expert(proj_dim):
        return nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(),
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(),
            nn.Linear(proj_dim, 1),
        )

    def _freq_token(self, x):
        raw = x * self._freq_std + self._freq_mean
        c = F.conv2d(raw, self._haar.to(raw.dtype), stride=2, groups=3)
        hf = torch.cat([c[:, 1:4], c[:, 5:8], c[:, 9:12]], dim=1)
        return self.freq_pool(self.freq_enc(hf)).flatten(1)

    def _build_router_input(self, f_tok, h_raw):
        cls_tok = self.cls_router_proj(h_raw[:, -1, :].detach())
        return torch.cat([f_tok, cls_tok], dim=-1)

    def _compute_gates(self, router_logits):
        if self.training:
            if self._use_gumbel:
                return F.gumbel_softmax(router_logits, tau=self._gumbel_tau, hard=True, dim=-1)
            tau = self.tau.clamp(0.1, 10.0)
            return torch.softmax(router_logits / tau, dim=-1)
        return F.one_hot(router_logits.argmax(dim=-1), num_classes=self._num_experts).to(router_logits.dtype)

    def forward(self, x):
        f_tok = self._freq_token(x)
        h_raw = self._extract(x).float()
        g = self.proj1(h_raw)

        z = (torch.softmax(self.alpha, dim=1) * g).sum(dim=1)
        z = self.proj2(z)

        gates = self._compute_gates(self.router(self._build_router_input(f_tok, h_raw)))
        expert_logits = torch.stack([e(z).squeeze(-1) for e in self.experts], dim=-1)
        return (gates * expert_logits).sum(dim=-1)

    def forward_with_aux(self, x):
        f_tok = self._freq_token(x)
        h_raw = self._extract(x).float()
        g = self.proj1(h_raw)

        z = (torch.softmax(self.alpha, dim=1) * g).sum(dim=1)
        z = self.proj2(z)

        gates = self._compute_gates(self.router(self._build_router_input(f_tok, h_raw)))
        expert_logits = torch.stack([e(z).squeeze(-1) for e in self.experts], dim=-1)
        logit = (gates * expert_logits).sum(dim=-1)
        return logit, z, gates, expert_logits

    def get_features(self, x):
        logit, z, _, _ = self.forward_with_aux(x)
        return logit, z

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def trainable_params(self):
        seen = set()
        params = []
        sources = (
            list(self.proj1.parameters()) +
            list(self.proj2.parameters()) +
            list(self.experts.parameters()) +
            list(self.freq_enc.parameters()) +
            list(self.freq_pool.parameters()) +
            list(self.cls_router_proj.parameters()) +
            list(self.router.parameters()) +
            [self.tau, self.alpha] +
            self._lora_params
        )
        for p in sources:
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

    def load_freq_encoder(self, freq_ckpt_path, device="cpu"):
        ckpt = torch.load(freq_ckpt_path, map_location="cpu")
        state = ckpt.get("trainable_model", ckpt.get("model", ckpt))
        mapping = {}
        for k, v in state.items():
            if k.startswith("enc."):
                mapping["freq_enc." + k[4:]] = v
        self.load_state_dict(mapping, strict=False)

    def load_trainable(self, path, device="cpu"):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "trainable_model" in state:
            state = state["trainable_model"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        head_keys = {k: v for k, v in state.items() if k.startswith("head.")}
        if head_keys and not any(k.startswith("experts.") for k in state):
            for i in range(self._num_experts):
                for hk, hv in head_keys.items():
                    w = hv.clone()
                    if i > 0 and w.is_floating_point():
                        scale = max(w.std(unbiased=False).item(), w.abs().mean().item(), 1e-6)
                        w = w + torch.randn_like(w) * scale * 0.01
                    state[f"experts.{i}.{hk[5:]}"] = w
            for hk in head_keys:
                state.pop(hk)
        if "alpha" in state and state["alpha"].shape != self.alpha.shape:
            state.pop("alpha")
        self.load_state_dict(state, strict=False)
