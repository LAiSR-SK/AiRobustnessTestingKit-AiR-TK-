# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from typing import Optional, Union

import torch
import torchvision.models as torchvision_models

# from .distances_adv import normalize_flatten_features, LPIPSDistance
# from .utilities import MarginLoss
from adversarial_training_toolkit.defense.oaat.perceptual_advex_models import (
    AlexNetFeatureModel,
    CifarAlexNet,
    FeatureModel,
)

# from . import utilities
from adversarial_training_toolkit.model import ResNet18, WideResNet
from torch.hub import load_state_dict_from_url
from typing_extensions import Literal

_cached_alexnet: Optional[AlexNetFeatureModel] = None
_cached_alexnet_cifar: Optional[AlexNetFeatureModel] = None



def get_lpips_model(
    lpips_model_spec: Union[
        Literal["self", "resnet"],
        FeatureModel,
    ],
    model: Optional[FeatureModel] = None,
    load_state_dict=0,
) -> FeatureModel:
    global _cached_alexnet, _cached_alexnet_cifar

    lpips_model: FeatureModel

    if lpips_model_spec == "self":
        if model is None:
            raise ValueError(
                'Specified "self" for LPIPS model but no model passed'
            )
        return model
    elif lpips_model_spec == "alexnet":
        if _cached_alexnet is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            _cached_alexnet = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet
        if torch.cuda.is_available():
            lpips_model.cuda()
    elif lpips_model_spec == "alexnet_cifar":
        if _cached_alexnet_cifar is None:
            alexnet_model = CifarAlexNet()
            _cached_alexnet_cifar = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet_cifar
        if torch.cuda.is_available():
            lpips_model.cuda()
        try:
            state = torch.load("./alexnet_cifar.pt")
        except FileNotFoundError:
            state = load_state_dict_from_url(
                "https://perceptual-advex.s3.us-east-2.amazonaws.com/"
                "alexnet_cifar.pt",
                progress=True,
            )
        lpips_model.load_state_dict(state["model"])
    elif lpips_model_spec == "ResNet18":
        lpips_model = ResNet18(num_classes=10)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif lpips_model_spec == "WideResNet34":
        lpips_model = WideResNet(depth=34, num_classes=10)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    # elif lpips_model_spec == 'PreActResNet18':
    # lpips_model = PreActResNet18(num_classes = 10)
    # lpips_model = torch.nn.DataParallel(lpips_model)
    # lpips_model.load_state_dict(torch.load(load_state_dict))
    # lpips_model.eval()

    elif isinstance(lpips_model_spec, str):
        raise ValueError(f'Invalid LPIPS model "{lpips_model_spec}"')
    else:
        lpips_model = lpips_model_spec

    lpips_model.eval()
    return lpips_model


def get_lpips_model_100classes(
    lpips_model_spec: Union[
        Literal["self", "resnet"],
        FeatureModel,
    ],
    model: Optional[FeatureModel] = None,
    load_state_dict=0,
) -> FeatureModel:
    global _cached_alexnet, _cached_alexnet_cifar

    lpips_model: FeatureModel

    if lpips_model_spec == "self":
        if model is None:
            raise ValueError(
                'Specified "self" for LPIPS model but no model passed'
            )
        return model
    elif lpips_model_spec == "alexnet":
        if _cached_alexnet is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            _cached_alexnet = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet
        if torch.cuda.is_available():
            lpips_model.cuda()
    elif lpips_model_spec == "alexnet_cifar":
        if _cached_alexnet_cifar is None:
            alexnet_model = CifarAlexNet()
            _cached_alexnet_cifar = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet_cifar
        if torch.cuda.is_available():
            lpips_model.cuda()
        try:
            state = torch.load("./alexnet_cifar.pt")
        except FileNotFoundError:
            state = load_state_dict_from_url(
                "https://perceptual-advex.s3.us-east-2.amazonaws.com/"
                "alexnet_cifar.pt",
                progress=True,
            )
        lpips_model.load_state_dict(state["model"])
    elif lpips_model_spec == "ResNet18":
        lpips_model = ResNet18(num_classes=100)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    elif lpips_model_spec == "WideResNet34":
        lpips_model = WideResNet(depth=34, num_classes=100)
        lpips_model = torch.nn.DataParallel(lpips_model)
        lpips_model.load_state_dict(torch.load(load_state_dict))
        lpips_model.eval()

    # elif lpips_model_spec == 'PreActResNet18':
    # lpips_model = PreActResNet18(num_classes = 100)
    # lpips_model = torch.nn.DataParallel(lpips_model)
    # lpips_model.load_state_dict(torch.load(load_state_dict))
    # lpips_model.eval()

    elif isinstance(lpips_model_spec, str):
        raise ValueError(f'Invalid LPIPS model "{lpips_model_spec}"')
    else:
        lpips_model = lpips_model_spec

    lpips_model.eval()
    return lpips_model


# NOTE remove PreActResNet18option
