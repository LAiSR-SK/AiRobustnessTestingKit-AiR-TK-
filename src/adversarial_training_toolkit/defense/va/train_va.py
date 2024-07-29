# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import copy
import os
import time
from collections import namedtuple
from typing import Dict, NamedTuple

import numpy as np
import torch
from autoattack import AutoAttack
from torch import Tensor, nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from adversarial_training_toolkit.attack import create_attack
from adversarial_training_toolkit.helper_functions import (
    cw_whitebox,
    eval_clean,
    get_model,
    load_data,
    mim_whitebox,
    robust_eval,
)
from adversarial_training_toolkit.loss import accuracy, standard_loss


class VaTraining:
    """Class which provides a simple interface for the VA training method.
    """
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
        save_freq: int = 1,
        swa_warmup: int = 50,
        swa: bool = True,
        lr_schedule: str = "decay",
    ) -> None:
        """Creates a new `VA` instance.

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
                "save_freq",
                "swa_warmup",
                "swa",
                "lr_schedule",
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
            save_freq,
            swa_warmup,
            swa,
            lr_schedule,
        )
        self._dataset = dataset
        self._model = model

    def __call__(self) -> None:
        """Trains the model defined by __init__.
        """
        main(self._dataset, self._model, 0, self._args)


def train(
    args: NamedTuple,
    model: Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    attacks: Dict[int, str],
    ds_name: str,
):
    """
    Trains one epoch by calculating the loss for each batch based on the given
    set of attacks and updating the model

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

    # For each batch of images in the train loader
    for batch_idx, sample in enumerate(train_loader):
        # Extract the image and classification details
        if ds_name == "cifar100":
            data, target, coarse = sample
        else:
            data, target = sample

        # Set the data/target to the current device
        data, target = (
            data.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )

        # Determine the 'class' of the batch by looking at the most common
        # class
        if ds_name == "cifar100":
            avg_cls = int(torch.mode(coarse)[0])
        else:
            avg_cls = int(torch.mode(target)[0])

        # Determine the attack based on the most common class
        attack = attacks[avg_cls]

        optimizer.zero_grad()  # zero optimizer gradients

        # Calculate loss using the specified attack
        loss, batch_metrics = va_loss(
            model, data, target, optimizer, attack, device, ds_name
        )

        # Update model parameters
        loss.backward()
        optimizer.step()

        # Print training progress
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/"
                + f"{len(train_loader.dataset)} ("
                + f"{100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: "
                + f"{loss.item():.6f}"
            )


def va_loss(
    model: Module,
    x: Tensor,
    y: Tensor,
    optimizer: Optimizer,
    att: str,
    device: torch.device,
    ds: str,
    keep_clean: bool = False,
):
    """
    Adversarial training loss function for later various atts epoch, which
    simply calculates the loss based on the given attack.

    :param model: trained image classifier
    :param x: batch of clean images
    :param y: labels associated with images in x
    :param optimizer: optimizer to be updated
    :param att: attack to be used
    :param device: current device
    :param ds: name of dataset
    :param keep_clean: True if clean examples are kept, False default

    :return loss to the classifier
    :return metrics of the batch
    """

    # Set the criterion to cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Create the adversary for AutoAttack
    adversary = AutoAttack(
        model, norm="Linf", eps=0.031, version="custom", verbose=False
    )

    # Generate adversarial example based on specified attack
    if att == "none":
        return standard_loss(model, x, y, optimizer)
    elif att == "linf-pgd-40":
        attack = create_attack(model, criterion, "linf-pgd", 0.03, 40, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "linf-pgd-20":
        attack = create_attack(model, criterion, "linf-pgd", 0.03, 20, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "linf-pgd-7":
        attack = create_attack(model, criterion, "linf-pgd", 0.03, 7, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "linf-pgd-3":
        attack = create_attack(model, criterion, "linf-pgd", 0.03, 3, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "fgsm-40":
        attack = create_attack(model, criterion, "fgsm", 0.03, 40, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "fgsm-20":
        attack = create_attack(model, criterion, "fgsm", 0.03, 20, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "fgsm-7":
        attack = create_attack(model, criterion, "fgsm", 0.03, 7, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "cw":
        x_adv = cw_whitebox(model, x, y, device, ds)
    elif att == "mim":
        x_adv = mim_whitebox(model, x, y, device)
    elif att == "l2-pgd-40":
        attack = create_attack(model, criterion, "l2-pgd", 0.03, 40, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "l2-pgd-20":
        attack = create_attack(model, criterion, "l2-pgd", 0.03, 20, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "l2-pgd-7":
        attack = create_attack(model, criterion, "l2-pgd", 0.03, 7, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == "autoattack":
        aa_attack = AutoAttack(
            model, norm="Linf", eps=0.031, version="standard", verbose=False
        )
        x_adv = aa_attack.run_standard_evaluation(x, y)
    elif att == "apgd-ce":
        adversary.attacks_to_run = ["apgd-ce"]
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == "apgd-dlr":
        adversary.attacks_to_run = ["apgd-dlr"]
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == "apgd-t":
        adversary.attacks_to_run = ["apgd-t"]
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == "fab":
        adversary.attacks_to_run = ["fab"]
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == "fab-t":
        adversary.attacks_to_run = ["fab-t"]
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == "square":
        adversary.attacks_to_run = ["square"]
        x_adv = adversary.run_standard_evaluation(x, y)
    else:
        print(att)
        raise NotImplementedError

    optimizer.zero_grad()  # zero out the gradients

    # Correctly format x_adv and y_adv
    if keep_clean:
        x_adv = torch.cat((x, x_adv), dim=0)
        y_adv = torch.cat((y, y), dim=0)
    else:
        y_adv = y

    # Calculates the adversarial output and CE loss
    out = model(x_adv)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y_adv)

    preds = out.detach()  # correctly format adversarial output

    # Calculate batch metrics to be returned
    batch_metrics = {"loss": loss.item()}
    if keep_clean:
        preds_clean, preds_adv = preds[: len(x)], preds[len(x) :]
        batch_metrics.update(
            {
                "clean_acc": accuracy(y, preds_clean),
                "adversarial_acc": accuracy(y, preds_adv),
            }
        )
    else:
        batch_metrics.update({"adversarial_acc": accuracy(y, preds)})

    return loss, batch_metrics


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


def epochs_define_attacks(epoch: int, dataset: str) -> Dict[int, str]:
    """
    Defines a dictionary of attacks based on the current epoch.

    :param epoch: current epoch
    :param dataset: name of dataset

    :return attacks: dictionary of attacks
    """

    attacks = {}
    if dataset == "cifar10":
        if epoch <= 5:
            for i in range(4):
                attacks[i] = "l2-pgd-7"
            for i in range(4, 7):
                attacks[i] = "l2-pgd-40"
            for i in range(7, 10):
                attacks[i] = "l2-pgd-20"
        elif epoch <= 10:
            for i in range(4):
                attacks[i] = "l2-pgd-20"
            for i in range(4, 7):
                attacks[i] = "l2-pgd-7"
            for i in range(7, 10):
                attacks[i] = "l2-pgd-40"
        elif epoch <= 15:
            for i in range(4):
                attacks[i] = "l2-pgd-40"
            for i in range(4, 7):
                attacks[i] = "l2-pgd-20"
            for i in range(7, 10):
                attacks[i] = "l2-pgd-7"
        elif epoch <= 25:
            for i in range(10):
                attacks[i] = "linf-pgd-3"
        elif epoch <= 30:
            for i in range(3):
                attacks[i] = "linf-pgd-7"
            for i in range(3, 6):
                attacks[i] = "linf-pgd-20"
            for i in range(6, 8):
                attacks[i] = "linf-pgd-40"
            for i in range(8, 10):
                attacks[i] = "mim"
        elif epoch <= 35:
            for i in range(3):
                attacks[i] = "linf-pgd-20"
            for i in range(3, 6):
                attacks[i] = "linf-pgd-7"
            for i in range(6, 8):
                attacks[i] = "mim"
            for i in range(8, 10):
                attacks[i] = "linf-pgd-40"
        elif epoch <= 40:
            for i in range(3):
                attacks[i] = "linf-pgd-40"
            for i in range(3, 6):
                attacks[i] = "mim"
            for i in range(6, 8):
                attacks[i] = "linf-pgd-7"
            for i in range(8, 10):
                attacks[i] = "linf-pgd-20"
        elif epoch <= 45:
            for i in range(3):
                attacks[i] = "mim"
            for i in range(3, 6):
                attacks[i] = "linf-pgd-40"
            for i in range(6, 8):
                attacks[i] = "linf-pgd-20"
            for i in range(8, 10):
                attacks[i] = "linf-pgd-7"
        elif epoch <= 53:
            for i in range(2):
                attacks[i] = "apgd-t"
            for i in range(2, 5):
                attacks[i] = "cw"
            for i in range(5, 8):
                attacks[i] = "apgd-dlr"
            for i in range(8, 10):
                attacks[i] = "apgd-ce"
        elif epoch <= 61:
            for i in range(2):
                attacks[i] = "apgd-ce"
            for i in range(2, 5):
                attacks[i] = "apgd-t"
            for i in range(5, 8):
                attacks[i] = "cw"
            for i in range(8, 10):
                attacks[i] = "apgd-dlr"
        elif epoch <= 68:
            for i in range(2):
                attacks[i] = "apgd-dlr"
            for i in range(2, 5):
                attacks[i] = "apgd-ce"
            for i in range(5, 8):
                attacks[i] = "apgd-t"
            for i in range(8, 10):
                attacks[i] = "cw"
        elif epoch <= 75:
            for i in range(2):
                attacks[i] = "cw"
            for i in range(2, 5):
                attacks[i] = "apgd-dlr"
            for i in range(5, 8):
                attacks[i] = "apgd-ce"
            for i in range(8, 10):
                attacks[i] = "apgd-t"
        else:
            for i in range(10):
                attacks[i] = "autoattack"
    elif dataset == "cifar100":
        if epoch <= 25:
            for i in range(20):
                attacks[i] = "linf-pgd-3"
        elif epoch <= 45:
            for i in range(10):
                attacks[i] = "linf-pgd-7"
            for i in range(10, 20):
                attacks[i] = "mim"
        elif epoch <= 65:
            for i in range(10):
                attacks[i] = "mim"
            for i in range(10, 20):
                attacks[i] = "linf-pgd-7"
        elif epoch <= 85:
            for i in range(10):
                attacks[i] = "linf-pgd-20"
            for i in range(10, 20):
                attacks[i] = "linf-pgd-40"
        elif epoch <= 105:
            for i in range(10):
                attacks[i] = "linf-pgd-40"
            for i in range(10, 20):
                attacks[i] = "linf-pgd-20"
        elif epoch <= 140:
            for i in range(20):
                attacks[i] = "cw"
        else:
            for i in range(20):
                attacks[i] = "autoattack"
    else:
        raise NotImplementedError

    return attacks


def main_warmup(
    ds_name: str, mod_name: str, epochs: int, args: NamedTuple
) -> None:
    """
    Main training method, which establishes the model/dataset before
    conducting training and clean testing for each epoch. Then reports final
    training time and conducts robustness tests on the final model.

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
            train(args, model, device, train_loader, optimizer, epoch, ds_name)
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

    f.close()  # close output file


def main(ds_name, mod_name, clean_epochs, args) -> None:
    """
    Main training method, which establishes the model/dataset before
    conducting training and clean testing for each epoch. Then reports final
    training time and conducts robustness tests on the final model.

    :param ds_name: dataset (cifar10 or cifar100)
    :param mod_name: training model architecture
    :param clean_pochs: number of epochs in warmup round, default 0
    """

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Create file to print training progress
    filename = f"log/va-epochs-{ds_name}-{mod_name}-output.txt"
    with open(filename, "a") as f:
        # Initialize the model based on the specified architecture
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

        # Save the untrained model and optimizer
        torch.save(
            model.state_dict(),
            os.path.join(
                model_dir,
                f"model-va-epochs-{ds_name}-{mod_name}-start.pt",
            ),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(
                model_dir,
                f"opt-va-epochs-{ds_name}-{mod_name}-start.tar",
            ),
        )

        # Begin model training
        for epoch in range(1, args.epochs + 1):
            # Adjust the SGD learning rate based on the epoch
            adjust_learning_rate(optimizer, epoch, args)

            # Define the dictionary of attack types for the current epoch
            attacks = epochs_define_attacks(epoch, ds_name)

            # Calculate the time elapsed
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
            )
            end = time.time()  # stop recording training time
            epoch_time = end - start

            # Evaluation on natural and adversarial examples
            f.write("Epoch " + str(epoch) + "\n")
            f.write("Current Attack Dictionary: ")
            f.write(str(attacks) + "\n")
            f.write("Time for Training: " + str(epoch_time) + "\n")
            eval_clean(model, device, train_loader, "train", ds_name, f)
            eval_clean(model, device, test_loader, "test", ds_name, f)
            robust_eval(model, device, test_loader, ds_name, f)

            # Save the model (if designated)
            if epoch % args.save_freq == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        model_dir,
                        f"model-va-epochs-{ds_name}-{mod_name}-epoch{epoch}.pt",
                    ),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        model_dir,
                        f"opt-va-epochs-{ds_name}-{mod_name}-epoch{epoch}.tar",
                    ),
                )

        # Report full training time
        end_tot = time.time()
        total_time = end_tot - start_tot
        print("Total training time: " + str(total_time))
        f.write("Total training time: " + str(total_time) + "\n")

        # Load the final trained model, make a deepcopy, and evaluate against
        # FGSM and PGD attacks
        trained_model = get_model(mod_name, ds_name, device)

        trained_model.load_state_dict(
            torch.load(
                os.path.join(
                    model_dir,
                    f"model-va-epochs-{ds_name}-{mod_name}-epoch{epoch}.pt",
                ),
            )
        )
        model_copy = copy.deepcopy(trained_model)
        robust_eval(model_copy, device, test_loader, ds_name, f)
