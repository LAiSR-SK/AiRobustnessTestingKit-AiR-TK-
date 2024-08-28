# (c) 2024 LAiSR-SK

# This code is licensed under the MIT license (see LICENSE.md).

import copy
import os
import time

from collections import namedtuple

from typing import Any, Dict, NamedTuple


import numpy as np
import torch
from torch import optim

from torch.nn import Module

from torch.optim import Optimizer

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.optim.swa_utils import SWALR, AveragedModel

from torch.utils.data import DataLoader

from airtk.helper_functions import (
    class_define_attacks,
    eval_clean,
    get_model,
    load_data,
    robust_eval,
)
from airtk.loss import adt_va_loss


class AdtppTraining:
    """Class which provides a simple interface for the ADT++ training method."""

    def __init__(
        self,
        dataset: str,
        model: str,
        batch_size: int = 128,
        epochs: int = 200,
        weight_decay: float = 2e-4,
        lr: float = 0.1,
        momentum: float = 0.9,
        seed: int = 1,
        log_interval: int = 1,
        model_dir: str = "data/model",
        swa_warmup: int = 50,
        swa: bool = True,
        lr_schedule: str = "decay",
    ) -> None:
        """Creates a new `AdtppTraining` instance.


        :param dataset: the dataset to be used as a str. Should be either

            "cifar10" or "cifar100".

        :param model: the model to train. Should be one of

            "res18", "res34", "res50", "res101", "res152", or "wideres34".

        :param batch_size: the size of batches for training and testing.

            Defaults to 128.

        :param epochs: the number of epochs to train for.

            Defaults to 200.

        :param weight_decay: a weight decay constant for regularization.

            Defaults to 2e-4.

        :param lr: the initial learning rate. Defaults to 0.1.

        :param momentum: Nesterov momentum constant. Defaults to 0.9.

        :param seed: the random seed to use for training. Defaults to 1.

        :param log_interval: the number of batches to log per. Defaults to 1.

        :param model_dir: the the directory to output models to.

            Defaults to "data/model".

        :param swa_warmup: the number of epochs to apply

            stochastic weight averaging to. Defaults to 50.

        :param swa: whether or not to apply stochastic weight averaging.

            Defaults to True.

        :param lr_schedule: the learning rate schedule. Should be one of

            "decay", "schedules", "cosine", "cyclic-10", or "cyclic-5".

            Defaults to "decay".
        """

        ArgsPrototype = namedtuple(
            "ArgsPrototype",
            [
                "batch_size",
                "test_batch_size",
                "epochs",
                "weight_decay",
                "lr",
                "momentum",
                "no_cuda",
                "seed",
                "log_interval",
                "model_dir",
                "swa_warmup",
                "swa",
                "lr_schedule",
                "save_freq",
            ],
        )

        self._args = ArgsPrototype(
            batch_size,
            batch_size,
            epochs,
            weight_decay,
            lr,
            momentum,
            False,
            seed,
            log_interval,
            model_dir,
            swa_warmup,
            swa,
            lr_schedule,
            1,
        )
        self._dataset = dataset
        self._model = model

    def __call__(self) -> None:
        """Trains the model defined by __init__."""

        main_va_epochs(self._dataset, self._model, self._args, 0)


def train(
    args: NamedTuple,
    model: Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    attacks: Dict[int, str],
    ds_name: str,
    f: str,
    swa_model: Module,
    swa_scheduler: Any,
    scheduler: Any,
):
    """

    Trains one epoch (not the first) by calculating the loss for each batch

    based on the given set of atts, and updating the model


    :param args: set of arguments including learning rate, log interval, etc

    :param model: model to train

    :param device: the current device

    :param train_loader: data loader containing the training dataset

    :param optimizer: optimizer to train

    :param epoch: current epoch of training

    :param attacks: dictionary of attack types based on class

    :param ds_name: name of dataset
    """

    model.train()  # Set the model to training mode

    # For each batch of images in the train loader (size 128)

    for batch_idx, sample in enumerate(train_loader):
        # Extract the image and classification details to determine the class

        if ds_name == "cifar100":
            data, target, coarse = sample

            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )

            avg_cls = int(torch.mode(coarse)[0])

        else:
            data, target = sample

            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )

            avg_cls = int(torch.mode(target)[0])

        # Determine the attack based on the most common class

        attack = attacks[avg_cls]

        # Zero out the gradients of the optimizer

        optimizer.zero_grad()

        # Determine the loss to the batch based on the attack

        loss = adt_va_loss(
            model, data, target, attack, ds_name, optimizer, device
        )

        # Update the model parameters based on the calculated class

        loss.backward()

        optimizer.step()

        # Print the training progress

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}"
                + f"/{len(train_loader.dataset)} "
                + f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                + f" Loss: {loss.item():.6f}"
            )

            f.write(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}"
                + f"/{len(train_loader.dataset)} "
                + f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                + f"Loss: {loss.item():.6f}"
            )

    # Update the SWA model/scheduler

    if args.swa:
        if epoch > args.swa_warmup:
            swa_model.update_parameters(model)

            swa_scheduler.step()

        else:
            scheduler.step()


def main_va_epochs(
    ds_name: str, mod_name: str, args: NamedTuple, clean_epochs: int = 0
):
    """

    Main training method, which establishes the model/dataset before

    conducting training and clean testing for each epoch. Then reports final

    training time and conducts robustness tests on the final model.


    :param ds_name: the dataset to use for training

    :param mod_name: the model to use for training
    """

    # Establish the settings
    model_dir = args.model_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Set up file for printing the outut

    filename = f"log/adt-va-{ds_name}-{mod_name}-output.txt"

    with open(filename, "a") as f:
        # Initialize the model based on the specified parameter

        start_tot = time.time()  # start recording training time

        base_model = get_model(mod_name, ds_name, device)

        # Clean training/warmup round, if specified

        if clean_epochs == 0:
            model = copy.deepcopy(base_model).to(device)

        else:
            main_warmup(ds_name, mod_name, clean_epochs)

            base_model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir,
                        f"model-warmup-{ds_name}-{mod_name}-epoch{clean_epochs}.pt",
                    )
                )
            )

            model = copy.deepcopy(base_model).to(device)

        # Set up the optimizer to the arguments

        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.swa:
            swa_model = AveragedModel(model).to(device)

            scheduler = CosineAnnealingLR(optimizer, T_max=100)

            swa_scheduler = SWALR(optimizer, swa_lr=0.05)

        else:
            swa_model = model

            scheduler = optimizer

            swa_scheduler = optimizer

        # Set up the dataloaders

        train_loader, test_loader = load_data(
            ds_name, args, kwargs, coarse=True
        )

        # Save the model and optimizer

        torch.save(
            model.state_dict(),
            os.path.join(
                model_dir, f"model-adt-va-{ds_name}-{mod_name}-start.pt"
            ),
        )

        torch.save(
            optimizer.state_dict(),
            os.path.join(
                model_dir, f"opt-adt-va-{ds_name}-{mod_name}-start.tar"
            ),
        )

        # For each epoch

        for epoch in range(1, args.epochs + 1):
            # Adjust the SGD learning rate based on the epoch

            adjust_learning_rate(optimizer, epoch, args)

            # Define the dictionary of attack types

            attacks = class_define_attacks(ds_name)

            # Calculate the time elapsed

            # Commence training, timing the method and recording the time

            start = time.time()  # start recording training time

            train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                attacks,
                ds_name,
                f,
                swa_model,
                swa_scheduler,
                scheduler,
            )

            end = time.time()  # stop recording training time

            epoch_time = end - start

            # Update bn statistics for the swa_model at the end

            if args.swa:
                torch.optim.swa_utils.update_bn(
                    train_loader, swa_model, device=device
                )

            # evaluation on natural examples (training and testing)

            print(
                "================================================================"
            )

            f.write("Epoch " + str(epoch) + "\n")

            f.write("Current Attack Dictionary: ")

            f.write(str(attacks) + "\n")

            print("Time for Training: " + str(epoch_time))

            f.write("Time for Training: " + str(epoch_time) + "\n")

            if args.swa:
                if epoch > args.swa_warmup:
                    eval_clean(
                        swa_model, device, train_loader, "train", ds_name, f
                    )

                    eval_clean(
                        swa_model, device, test_loader, "test", ds_name, f
                    )

                    robust_eval(swa_model, device, test_loader, ds_name, f)

            else:
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
                        model_dir,
                        f"model-adt-va-{ds_name}-{mod_name}-epoch{epoch}.pt",
                    ),
                )

                torch.save(
                    swa_model.state_dict(),
                    os.path.join(
                        model_dir,
                        f"swa-model-adt-va-{ds_name}-{mod_name}-epoch{epoch}.pt",
                    ),
                )

                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        model_dir,
                        f"opt-adt-va-{ds_name}-{mod_name}-epoch{epoch}.tar",
                    ),
                )

        # Report the time taken to fully train the model

        end_tot = time.time()

        total_time = end_tot - start_tot

        print("Total training time: " + str(total_time))

        f.write("Total training time: " + str(total_time) + "\n")

        # Load the final trained model, make a deepcopy, and evaluate against

        # FGSM and PGD

        trained_model = get_model(mod_name, ds_name, device)

        if args.swa:
            trained_model = AveragedModel(trained_model).to(device)

            trained_model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir,
                        f"swa-model-adt-va-{ds_name}-{mod_name}-epoch{args.epochs}.pt",
                    )
                )
            )

        else:
            trained_model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir,
                        f"model-adt-va-{ds_name}-{mod_name}-epoch{args.epochs}.pt",
                    )
                )
            )

        model_copy = copy.deepcopy(trained_model)

        robust_eval(model_copy, device, test_loader, ds_name, f)


def main_warmup(
    ds_name: str, mod_name: str, epochs: int, args: NamedTuple
) -> None:
    """

    Main training method, which establishes the model/dataset before

    conducting training and clean testing for each epoch. Then reports

    final training time and conducts robustness tests on the final model.


    :param ds_name: dataset to use for training

    :param mod_name: model to use for training

    :param epochs: number of epochs to train for
    """

    device = "cuda" if torch.cuda.is_available else "cpu"
    model_dir = args.model_dir

    # Create file to print training progress

    filename = f"log/clean-{ds_name}-{mod_name}-output.txt"

    kwargs = (
        {"num_workers": 1, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )

    with open(filename, "a") as f:
        # Initialize the model based on the specified architecture

        start_tot = time.time()  # start recording training time

        model = get_model(mod_name, ds_name, device)

        # Set up the optimizer with the arguments

        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        # Set up the dataloaders

        train_loader, test_loader = load_data(
            ds_name, args, kwargs, coarse=True
        )

        # Begin model training for the specified number of epochs

        for epoch in range(1, epochs + 1):
            # Adjust the SGD learning rate based on the epoch

            adjust_learning_rate(optimizer, epoch, args)

            start = time.time()  # start recording training time

            train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                ds_name,
            )

            end = time.time()  # stop recording training time

            epoch_time = end - start

            # Evaluation on natural and adversarial examples

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
                        model_dir,
                        f"model-warmup-{ds_name}-{mod_name}-epoch{epoch}.pt",
                    ),
                )

                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        model_dir,
                        f"opt-warmup-{ds_name}-{mod_name}-epoch{epoch}.tar",
                    ),
                )

        # Report full training time

        end_tot = time.time()

        total_time = end_tot - start_tot

        f.write("Total training time: " + str(total_time) + "\n")


def adjust_learning_rate(
    optimizer: Optimizer, epoch: int, args: NamedTuple
) -> None:
    """

    Sets the learning rate of the optimizer based on the current epoch


    :param optimizer: optimizer with learning rate being set

    :param epoch: current epoch

    :param args: program arguments
    """
    lr = args.lr

    if args.lr_schedule == "decay":
        if epoch >= 50:
            lr = args.lr * 0.1

        if epoch >= 75:
            lr = args.lr * 0.01

    elif args.lr_schedule == "scheduled":
        if epoch >= 24:
            lr = args.lr * 0.1

        if epoch >= 26:
            lr = args.lr

        if epoch >= 64:
            lr = args.lr * 0.1

        if epoch >= 66:
            lr = args.lr

        if epoch >= 104:
            lr = args.lr * 0.1

        if epoch >= 106:
            lr = args.lr

        if epoch >= 137:
            lr = args.lr * 0.1

        if epoch >= 139:
            lr = args.lr * 0.01

    elif args.lr_schedule == "cosine":
        lr = 0.2

        lr = lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))

    elif args.lr_schedule == "cyclic-10":
        if epoch % 10 < 2 or epoch % 10 == 9:
            lr = args.lr

        elif epoch % 10 < 4 or epoch % 10 > 6:
            lr = args.lr * 0.1

        elif epoch % 10 == 5:
            lr = args.lr * 0.001

        else:
            lr = args.lr * 0.01

    elif args.lr_schedule == "cyclic-5":
        if epoch % 5 == 0:
            lr = args.lr

        elif epoch % 5 == 1 or epoch % 5 == 4:
            lr = args.lr * 0.1

        else:
            lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
