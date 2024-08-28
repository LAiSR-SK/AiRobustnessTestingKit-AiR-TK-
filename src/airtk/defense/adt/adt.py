# (c) 2024 LAiSR-SK

# This code is licensed under the MIT license (see LICENSE.md).


import copy
import os
import time

from collections import namedtuple

import torch
from torch import optim

from torch.nn import Module

from airtk.helper_functions import (
    adjust_learning_rate,
    eval_clean,
    load_data,
    robust_eval,
)
from airtk.loss import adt_loss
from airtk.model import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    WideResNet,
)


class AdtTraining:
    def __init__(
        self,
        ds_name: str,
        model_name: str,
        batch_size: int = 128,
        epochs: int = 76,
        weight_decay: float = 2e-4,
        lr: float = 0.1,
        momentum: float = 0.9,
        seed: int = 0,
        model_dir: str = "data/model",
        lr_schedule: str = "decay",
    ) -> None:
        ArgsPrototype = namedtuple(
            "Prototype",
            [
                "batch_size",
                "test_batch_size",
                "epochs",
                "weight_decay",
                "lr",
                "momentum",
                "seed",
                "model_dir",
                "save_freq",
                "lr_schedule",
                "log_interval",
            ],
        )

        self._args = ArgsPrototype(
            batch_size,
            batch_size,
            epochs,
            weight_decay,
            lr,
            momentum,
            seed,
            model_dir,
            1,
            seed,
            1,
        )

        self._ds_name: str = ds_name

        self._model_name: str = model_name

    def __call__(self) -> Module:
        model_dir: str = self._args.model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.manual_seed(self._args.seed)

        main_adt(self._ds_name, self._model_name, self._args)


def train(args, model, device, train_loader, optimizer, ds_name, epoch):
    """

    Trains one epoch by calculating the loss for each batch and updating the model


    :param args: set of arguments including learning rate, log interval, etc

    :param model: model to train

    :param device: the current device

    :param train_loader: data loader containing the training dataset

    :param optimizer: optimizer to train

    :param epoch: current epoch of training
    """

    model.train()  # set the  model to training mode

    for batch_idx, sample in enumerate(train_loader):
        # Set the data and target (x and y) to the device

        if ds_name == "cifar100":
            data, target, coarse = sample

        else:
            data, target = sample

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # zero out the gradients

        # Calculate the ADT loss

        loss = adt_loss(model, data, target, optimizer)

        # Update the model based on the loss

        loss.backward()

        optimizer.step()

        # print progress

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def main_adt(ds_name, mod_name, args):
    """

    Main training method, which establishes the model/dataset before conducting training and

    clean testing for each epoch. Then reports final training time and conducts robustness tests
    on the final model.


    :param ds_name: the dataset to use for training

    :param mod_name: the model to use for training
    """

    # Set up file for printing the output

    filename = f"log/adt-{ds_name}-{mod_name}-output.txt"
    f = open(filename, "a")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the desired model

    start_tot = time.time()  # start timing the full method

    if mod_name == "res18":
        model = ResNet18(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )

    elif mod_name == "res34":
        model = ResNet34(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )

    elif mod_name == "res50":
        model = ResNet50(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )

    elif mod_name == "res101":
        model = ResNet101(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )

    elif mod_name == "res152":
        model = ResNet152(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )

    elif mod_name == "wideres34":
        model = WideResNet(
            depth=34, num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)

    else:
        raise NotImplementedError

    # Set up the optimizer to the arguments

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Set up the dataloaders

    kwargs = (
        {"num_workers": 1, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )

    train_loader, test_loader = load_data(ds_name, args, kwargs, coarse=True)

    # Save the model and optimizer

    torch.save(
        model.state_dict(),
        os.path.join(
            args.model_dir, f"model-adt-{ds_name}-{mod_name}-start.pt"
        ),
    )

    torch.save(
        optimizer.state_dict(),
        os.path.join(
            args.model_dir, f"opt-adt-{ds_name}-{mod_name}-start.tar"
        ),
    )

    # Begin training for the designated number of epochs

    for epoch in range(1, args.epochs + 1):
        # Adjust the learning rate for the SGD

        adjust_learning_rate(optimizer, epoch, args)

        # Commence training, timing the method and recording the time

        start = time.time()  # start timing the method

        train(args, model, device, train_loader, optimizer, ds_name, epoch)

        end = time.time()  # finish timing the method

        epoch_time = end - start

        # evaluation on natural examples (training and testing)
        print(
            "================================================================"
        )

        f.write("Epoch " + str(epoch) + "\n")

        print("Time for Training: " + str(epoch_time))

        f.write("Time for Training: " + str(epoch_time) + "\n")

        eval_clean(model, device, train_loader, "train", ds_name, f)

        eval_clean(model, device, test_loader, "test", ds_name, f)

        robust_eval(model, device, test_loader, ds_name, f)
        print(
            "================================================================"
        )

        # Save the model (if designated)

        if epoch % args.save_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.model_dir,
                    f"model-adt-{ds_name}-{mod_name}-epoch{epoch}.pt",
                ),
            )

            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    args.model_dir,
                    f"opt-adt-{ds_name}-{mod_name}-epoch{epoch}.tar",
                ),
            )

    # Report the time taken to fully train the model
    end_tot = time.time()

    total_time = end_tot - start_tot

    print("Total training time: " + str(total_time))

    f.write("Total training time: " + str(total_time) + "\n")

    # Load the final trained model, make a deepcopy, and evaluate against FGSM and PGD

    if mod_name == "res18":
        trained_model = ResNet18(
            num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)

    elif mod_name == "res34":
        trained_model = ResNet34(
            num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)

    elif mod_name == "res50":
        trained_model = ResNet50(
            num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)

    elif mod_name == "res101":
        trained_model = ResNet101(
            num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)

    elif mod_name == "res152":
        trained_model = ResNet152(
            num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)

    elif mod_name == "wideres34":
        trained_model = WideResNet(
            depth=34, num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)

    else:
        raise NotImplementedError
    trained_model.load_state_dict(
        torch.load(
            f"data/model/model-adt-{ds_name}-{mod_name}-epoch{args.epochs}.pt"
        )
    )

    model_copy = copy.deepcopy(trained_model)

    robust_eval(model_copy, device, test_loader, ds_name, f)

    f.close()  # close the output file
