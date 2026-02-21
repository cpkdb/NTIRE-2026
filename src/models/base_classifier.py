from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseClassifier(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)[:, 1]


class TimmClassifier(BaseClassifier):

    def __init__(self, model_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
