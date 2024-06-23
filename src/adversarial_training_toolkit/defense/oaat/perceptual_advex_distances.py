# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.autograd
import torchvision.models as torchvision_models
from adversarial_training_toolkit.defense.oaat.perceptual_advex_models import (
    AlexNetFeatureModel,
    FeatureModel,
)
from torch import nn
from typing_extensions import Literal


class LPIPSDistance(nn.Module):
    """
    Calculates the square root of the Learned Perceptual Image Patch Similarity
    (LPIPS) between two images, using a given neural network.
    """

    model: FeatureModel

    def __init__(
        self,
        model: Optional[Union[FeatureModel, nn.DataParallel]] = None,
        activation_distance: Literal["l2"] = "l2",
        include_image_as_activation: bool = False,
    ):
        """
        Constructs an LPIPS distance metric. The given network should return a
        tuple of (activations, logits). If a network is not specified, AlexNet
        will be used. activation_distance can be 'l2' or 'cw_ssim'.
        """

        super().__init__()

        if model is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            self.model = AlexNetFeatureModel(alexnet_model)
        elif isinstance(model, nn.DataParallel):
            self.model = cast(FeatureModel, model.module)
        else:
            self.model = model

        self.activation_distance = activation_distance
        self.include_image_as_activation = include_image_as_activation

        self.eval()

    def features(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.model.features(image)
        if self.include_image_as_activation:
            features = (image,) + features
        return features

    def forward(self, image1, image2):
        features1 = self.features(image1)
        features2 = self.features(image2)

        if self.activation_distance == "l2":
            return (
                normalize_flatten_features(features1)
                - normalize_flatten_features(features2)
            ).norm(dim=1)
        else:
            raise ValueError(
                f'Invalid activation_distance "{self.activation_distance}"'
            )


def normalize_flatten_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
) -> torch.Tensor:
    """
    Given a tuple of features (layer1, layer2, layer3, ...) from a network,
    flattens those features into a single vector per batch input. The
    features are also scaled such that the L2 distance between features
    for two different inputs is the LPIPS distance between those inputs.
    """

    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = (
            torch.sqrt(torch.sum(feature_layer**2, dim=1, keepdim=True)) + eps
        )
        normalized_features.append(
            (
                feature_layer
                / (
                    norm_factor
                    * np.sqrt(
                        feature_layer.size()[2] * feature_layer.size()[3]
                    )
                )
            ).view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)
