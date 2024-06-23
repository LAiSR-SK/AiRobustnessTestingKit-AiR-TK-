# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from __future__ import print_function

import argparse
import copy
import os
import time

import torch
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from adversarial_training_toolkit.loss import adt_va_loss
from adversarial_training_toolkit.util import get_model

from helper_functions import (
    adjust_learning_rate,
    class_define_attacks,
    eval_clean,
    load_data,
    robust_eval,
)
from warmup_round import main_warmup


parser = argparse.ArgumentParser(
    description="PyTorch CIFAR TRADES Adversarial Training"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for testing (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train",
)
parser.add_argument(
    "--weight-decay", "--wd", default=2e-4, type=float, metavar="W"
)
parser.add_argument(
    "--lr", type=float, default=0.1, metavar="LR", help="learning rate"
)
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--model-dir",
    default="./data/model",
    help="directory of model for saving checkpoint",
)
parser.add_argument(
    "--save-freq",
    "-s",
    default=1,
    type=int,
    metavar="N",
    help="save frequency",
)
parser.add_argument(
    "--swa-warmup", default=50, type=int, metavar="N", help="save frequency"
)
parser.add_argument(
    "--swa", action="store_true", default=True, help="disables CUDA training"
)
parser.add_argument(
    "--lr-schedule",
    default="decay",
    help="schedule for adjusting learning rate",
)
args = parser.parse_args()

# Establish the settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


def train(
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
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            f.write(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    # Update the SWA model/scheduler
    if args.swa:
        if epoch > args.swa_warmup:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()


def main_va_epochs(ds_name, mod_name, clean_epochs=0):
    """
    Main training method, which establishes the model/dataset before conducting training and
    clean testing for each epoch. Then reports final training time and conducts robustness tests
    on the final model.

    :param ds_name: the dataset to use for training
    :param mod_name: the model to use for training
    """

    # Set up file for printing the outut
    filename = "log/adt-va-{}-{}-output.txt".format(ds_name, mod_name)
    f = open(filename, "a")

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
                    "model-warmup-{}-{}-epoch{}.pt".format(
                        ds_name, mod_name, clean_epochs
                    ),
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
    train_loader, test_loader = load_data(ds_name, args, kwargs, coarse=True)

    # Save the model and optimizer
    torch.save(
        model.state_dict(),
        os.path.join(
            model_dir, "model-adt-va-{}-{}-start.pt".format(ds_name, mod_name)
        ),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(
            model_dir, "opt-adt-va-{}-{}-start.tar".format(ds_name, mod_name)
        ),
    )

    # For each epoch
    for epoch in range(1, args.epochs + 1):
        # Adjust the SGD learning rate based on the epoch
        adjust_learning_rate(optimizer, epoch, args)

        # Define the dictionary of attack types
        attacks = class_define_attacks(ds_name)

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
                eval_clean(swa_model, device, test_loader, "test", ds_name, f)
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
                    "model-adt-va-{}-{}-epoch{}.pt".format(
                        ds_name, mod_name, epoch
                    ),
                ),
            )
            torch.save(
                swa_model.state_dict(),
                os.path.join(
                    model_dir,
                    "swa-model-adt-va-{}-{}-epoch{}.pt".format(
                        ds_name, mod_name, epoch
                    ),
                ),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    model_dir,
                    "opt-adt-va-{}-{}-epoch{}.tar".format(
                        ds_name, mod_name, epoch
                    ),
                ),
            )

    # Report the time taken to fully train the model
    end_tot = time.time()
    total_time = end_tot - start_tot
    print("Total training time: " + str(total_time))
    f.write("Total training time: " + str(total_time) + "\n")

    # Load the final trained model, make a deepcopy, and evaluate against FGSM and PGD
    trained_model = get_model(mod_name, ds_name, device)
    if args.swa:
        trained_model = AveragedModel(trained_model).to(device)
        trained_model.load_state_dict(
            torch.load(
                "saved-models/swa-model-adt-va-{}-{}-epoch{}.pt".format(
                    ds_name, mod_name, args.epochs
                )
            )
        )
    else:
        trained_model.load_state_dict(
            torch.load(
                "saved-models/model-adt-va-{}-{}-epoch{}.pt".format(
                    ds_name, mod_name, args.epochs
                )
            )
        )
    model_copy = copy.deepcopy(trained_model)

    robust_eval(model_copy, device, test_loader, ds_name, f)

    f.close()  # close output file


if __name__ == "__main__":
    main_va_epochs("cifar10", "wideres34", 1)
