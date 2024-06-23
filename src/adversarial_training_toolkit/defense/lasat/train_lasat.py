# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from adversarial_training_toolkit.defense.lasat.las_at_train_cifar10 import (
    main_lasat10,
)
from adversarial_training_toolkit.defense.lasat.las_at_train_cifar100 import (
    main_lasat100,
)


def main_lasat(ds_name):
    if ds_name == "cifar10":
        main_lasat10()
    elif ds_name == "cifar100":
        main_lasat100()


if __name__ == "__main__":
    main_lasat("cifar10")
