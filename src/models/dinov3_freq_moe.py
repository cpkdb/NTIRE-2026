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

        # --- Frequency encoder (from FreqClassifier) ---
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

        # --- Dynamic alpha generator ---
        n_hooks = len(self.hooks)
        self.alpha_gen = nn.Linear(128, n_hooks * proj_dim)

        # --- Router ---
        self.router = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_experts),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.tau = nn.Parameter(torch.ones(1))

        # --- Experts (independently initialized, cold-start) ---
        self.experts = nn.ModuleList([self._make_expert(proj_dim) for _ in range(num_experts)])

        # Remove single head/static alpha (replaced by experts + alpha_gen)
        del self.head
        del self.alpha

        self._num_experts = num_experts
        self._proj_dim = proj_dim
        self._n_hooks = n_hooks

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
        return self.freq_pool(self.freq_enc(hf)).flatten(1)  # [B,128]

    def forward(self, x):
        f_tok = self._freq_token(x)
        g = self.proj1(self._extract(x).float())

        B = g.shape[0]
        alpha_dyn = self.alpha_gen(f_tok).view(B, self._n_hooks, self._proj_dim)
        z = (torch.softmax(alpha_dyn, dim=1) * g).sum(dim=1)
        z = self.proj2(z)

        tau = self.tau.clamp(0.1, 10.0)
        gates = torch.softmax(self.router(f_tok) / tau, dim=-1)  # [B, E]
        expert_logits = torch.stack([e(z).squeeze(-1) for e in self.experts], dim=-1)  # [B, E]
        return (gates * expert_logits).sum(dim=-1)  # [B]

    def forward_with_aux(self, x):
        f_tok = self._freq_token(x)
        g = self.proj1(self._extract(x).float())

        B = g.shape[0]
        alpha_dyn = self.alpha_gen(f_tok).view(B, self._n_hooks, self._proj_dim)
        z = (torch.softmax(alpha_dyn, dim=1) * g).sum(dim=1)
        z = self.proj2(z)

        tau = self.tau.clamp(0.1, 10.0)
        gates = torch.softmax(self.router(f_tok) / tau, dim=-1)
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
            list(self.alpha_gen.parameters()) +
            list(self.router.parameters()) +
            [self.tau] +
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
        """Load pretrained FreqClassifier weights into freq_enc."""
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
        # Backward compat: map legacy single-head checkpoints (head.*) to all experts
        # Expert Mutation: inject tiny noise to break symmetry for MoE routing
        head_keys = {k: v for k, v in state.items() if k.startswith("head.")}
        if head_keys and not any(k.startswith("experts.") for k in state):
            for i in range(self._num_experts):
                for hk, hv in head_keys.items():
                    w = hv.clone()
                    if i > 0 and w.is_floating_point():
                        w = w + torch.randn_like(w) * 1e-3
                    state[f"experts.{i}.{hk[5:]}"] = w
            for hk in head_keys:
                state.pop(hk)
        state.pop("alpha", None)
        self.load_state_dict(state, strict=False)
