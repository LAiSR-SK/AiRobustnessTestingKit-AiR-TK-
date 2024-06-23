# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from typing import Tuple

import torch
from torch import nn
from torchvision import models as torchvision_models

# from .trades_wrn import TradesWideResNet


class FeatureModel(nn.Module):
    """
    A classifier model which can produce layer features, output logits, or
    both.
    """

    normalizer: nn.Module
    model: nn.Module

    def __init__(self):
        super().__init__()
        self._allow_training = False
        self.eval()

    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Should return a tuple of features (layer1, layer2, ...).
        """

        raise NotImplementedError()

    def classifier(self, last_layer: torch.Tensor) -> torch.Tensor:
        """
        Given the final activation, returns the output logits.
        """

        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for the given inputs.
        """

        return self.classifier(self.features(x)[-1])

    def features_logits(
        self,
        x: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Returns a tuple (features, logits) for the given inputs.
        """

        features = self.features(x)
        logits = self.classifier(features[-1])
        return features, logits

    def allow_train(self):
        self._allow_training = True

    def train(self, mode=True):
        if mode is True and not self._allow_training:
            raise RuntimeError("should not be in train mode")
        super().train(mode)


class ImageNetNormalizer(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        mean = torch.tensor(self.mean, device=x.device)
        std = torch.tensor(self.std, device=x.device)

        return (x - mean[None, :, None, None]) / std[None, :, None, None]


class AlexNetFeatureModel(FeatureModel):
    model: torchvision_models.AlexNet

    def __init__(self, alexnet_model: torchvision_models.AlexNet):
        super().__init__()
        self.normalizer = ImageNetNormalizer()
        self.model = alexnet_model.eval()

        assert len(self.model.features) == 13
        self.layer1 = nn.Sequential(self.model.features[:2])
        self.layer2 = nn.Sequential(self.model.features[2:5])
        self.layer3 = nn.Sequential(self.model.features[5:8])
        self.layer4 = nn.Sequential(self.model.features[8:10])
        self.layer5 = nn.Sequential(self.model.features[10:12])
        self.layer6 = self.model.features[12]

    def features(self, x):
        x = self.normalizer(x)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer5 = self.layer5(x_layer4)

        return (x_layer1, x_layer2, x_layer3, x_layer4, x_layer5)

    def classifier(self, last_layer):
        x = self.layer6(last_layer)
        if isinstance(self.model, CifarAlexNet):
            x = x.view(x.size(0), 256 * 2 * 2)
        else:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x


class CifarAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
