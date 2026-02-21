import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_classifier import BaseClassifier

_NORM = {
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class FreqClassifier(BaseClassifier):

    def __init__(self, backbone_type="imagenet", hidden=(32, 64, 96, 128)):
        super().__init__()
        mean, std = _NORM.get(backbone_type, _NORM["imagenet"])
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("_haar", self._make_haar())

        c1, c2, c3, c4 = hidden
        self.enc = nn.Sequential(
            nn.Conv2d(9, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(True),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c4), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, 1)

    @staticmethod
    def _make_haar():
        ll = torch.tensor([[.5, .5], [.5, .5]])
        lh = torch.tensor([[-.5, -.5], [.5, .5]])
        hl = torch.tensor([[-.5, .5], [-.5, .5]])
        hh = torch.tensor([[.5, -.5], [-.5, .5]])
        return torch.stack([ll, lh, hl, hh]).unsqueeze(1).repeat(3, 1, 1, 1)  # [12,1,2,2]

    def forward(self, x):
        x = x * self._std + self._mean  # denormalize
        c = F.conv2d(x, self._haar.to(x.dtype), stride=2, groups=3)  # [B,12,H/2,W/2]
        hf = c[:, [1, 2, 3, 4, 5, 6, 7, 8, 9], :, :]  # skip LL per channel (indices 0,4,8)

        # Correct: for groups=3, output order is [LL_R,LH_R,HL_R,HH_R, LL_G,LH_G,HL_G,HH_G, LL_B,LH_B,HL_B,HH_B]
        # We want LH,HL,HH for each channel = indices [1,2,3, 5,6,7, 9,10,11]
        hf = torch.cat([c[:, 1:4], c[:, 5:8], c[:, 9:12]], dim=1)  # [B,9,H/2,W/2]
        return self.fc(self.pool(self.enc(hf)).flatten(1)).squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def trainable_params(self):
        return list(self.parameters())

    def save_trainable(self, path):
        torch.save(self.state_dict(), path)

    def load_trainable(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device), strict=False)
