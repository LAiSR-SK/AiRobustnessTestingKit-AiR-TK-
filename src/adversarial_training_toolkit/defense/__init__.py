# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).

from adversarial_training_toolkit.defense.adt import AdtTraining
from adversarial_training_toolkit.defense.adtpp import AdtppTraining
from adversarial_training_toolkit.defense.fat import (
    FatTraining,
)
from adversarial_training_toolkit.defense.gairat import (
    GairatTraining,
)
from adversarial_training_toolkit.defense.trades import (
    TradesTraining,
)
from adversarial_training_toolkit.defense.tradesawp import TradesawpTraining
from adversarial_training_toolkit.defense.va import VaTraining

__all__ = [
    "AdtTraining",
    "AdtppTraining",
    # CurratTraining,  #! TODO: implement (this one is especially bad)
    "FatTraining",
    # "FeatureScatterTraining",  #! TODO: implement
    "GairatTraining",
    # "LasatTraining",  #! TODO: implement
    # "OaatTraining",  #! TODO: implement
    "TradesTraining",  #! TODO: needs tested (has some kind of demon in it)
    "TradesawpTraining",
    "VaTraining",
    # "YopoTraining",  #! TODO: implement
]
