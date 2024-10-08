# (c) 2024 LAiSR-SK

# This code is licensed under the MIT license (see LICENSE.md).

import argparse
import os
import time

import torch
from airtk.helper_functions import (
    adjust_learning_rate,
    eval_clean,
    load_data,
    robust_eval,
)
from airtk.loss import standard_loss
from airtk.util import get_model
from torch import optim

parser = argparse.ArgumentParser(description="PyTorch VA Adversarial Training")

parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
parser.add_argument(
    "--model-arch", default="wideres34", help="model architecture to train"
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
    default=140,
    metavar="N",
    help="number of epochs to train",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=0,
    metavar="N",
    help="number of epochs to train with clean data before AT",
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
    "--lr-schedule",
    default="decay",
    help="schedule for adjusting learning rate",
)
parser.add_argument(
    "--epsilon", type=float, default=8.0 / 255.0, help="perturbation"
)
parser.add_argument(
    "--num-steps", type=int, default=20, help="perturb number of steps"
)
parser.add_argument(
    "--step-size", type=float, default=2.0 / 255.0, help="perturb step size"
)
parser.add_argument(
    "--random", default=True, help="random initialization for PGD"
)

args = parser.parse_args()
model_dir = args.model_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


def train(args, model, device, train_loader, optimizer, epoch, ds_name):
    """

    Trains one epoch by calculating the loss for each batch and updating the model


    :param args: set of arguments including learning rate, log interval, etc

    :param model: model to train

    :param device: current device

    :param train_loader: data loader containing the training dataset

    :param optimizer: optimizer to train

    :param epoch: current epoch of training
    """

    model.train()  # set the  model to training mode

    for batch_idx, sample in enumerate(train_loader):
        # Extract the image and classification details

        if ds_name == "cifar100":
            data, target, coarse = sample

        else:
            data, target = sample

        data, target = (
            data.to(device),
            target.to(device),
        )  # set data/target to device

        optimizer.zero_grad()  # zero out the gradients

        # Calculate the clean loss

        loss, batch_metrics = standard_loss(model, data, target, optimizer)

        # Update the model based on the loss

        loss.backward()

        optimizer.step()

        # Print training progress

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def main_warmup(ds_name, mod_name, epochs):
    """

    Main training method, which establishes the model/dataset before conducting training and

    clean testing for each epoch. Then reports final training time and conducts robustness tests
    on the final model.


    :param ds_name: dataset to use for training

    :param mod_name: model to use for training

    :param epochs: number of epochs to train for
    """

    # Create file to print training progress

    filename = f"log/clean-{ds_name}-{mod_name}-output.txt"
    f = open(filename, "a")

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

    train_loader, test_loader = load_data(ds_name, args, kwargs, coarse=True)

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

    print("Total training time: " + str(total_time))

    f.write("Total training time: " + str(total_time) + "\n")

    f.close()  # close output file


if __name__ == "__main__":
    main_warmup("cifar100", "wideres34", 1)

    main_warmup("cifar10", "wideres34", 1)
