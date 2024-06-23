# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from adversarial_training_toolkit.defense.atawp.at_awp_train_cifar10 import (
    main_atawp10,
)
from adversarial_training_toolkit.defense.atawp.at_awp_train_cifar100 import (
    main_atawp100,
)


def main_atawp(ds_name):
    if ds_name == "cifar10":
        main_atawp10()
    elif ds_name == "cifar100":
        main_atawp100()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main_atawp("cifar10")
