# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).

import argparse
import copy
import os
import time

import torch
from adversarial_training_toolkit.loss import clean_loss
from adversarial_training_toolkit.model import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    WideResNet,
)
from adversarial_training_toolkit.helper_functions import (
    adjust_learning_rate,
    eval_clean,
    load_data,
    robust_eval,
)
from torch import optim

parser = argparse.ArgumentParser(
    description="PyTorch CIFAR Adversarial Training Framework"
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
    default=100,
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


def train(args, model, device, train_loader, optimizer, ds_name, epoch, f):
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
        # Set the data and target (x and y) to the device
        if ds_name == "cifar100":
            data, target, coarse = sample
        else:
            data, target = sample
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # zero out the gradients

        # Calculate the clean loss
        loss, batch_metrics = clean_loss(model, data, target, optimizer)

        # Update the model based on the loss
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            f.write(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def main_clean(ds_name, mod_name="wideres34"):
    """
    Main training method, which establishes the model/dataset before conducting training and
    clean testing for each epoch. Then reports final training time and conducts robustness tests
    on the final model.

    :param ds_name: the dataset to use for training
    :param mod_name: the model to use for training
    """

    # Set up file for printing the output
    filename = f"log/clean-{ds_name}-{mod_name}-output.txt"
    f = open(filename, "a")

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
    train_loader, test_loader = load_data(ds_name, args, kwargs, coarse=True)

    # Save the model and optimizer
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, f"model-clean-{ds_name}-{mod_name}-start.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(model_dir, f"opt-clean-{ds_name}-{mod_name}-start.tar"),
    )

    # Begin training for the designated number of epochs
    for epoch in range(1, args.epochs + 1):
        # Adjust the learning rate for the SGD
        adjust_learning_rate(optimizer, epoch, args)

        # Commence training, timing the method and recording the time
        start = time.time()  # start timing the method
        train(args, model, device, train_loader, optimizer, ds_name, epoch, f)
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
                    model_dir,
                    f"model-clean-{ds_name}-{mod_name}-epoch{epoch}.pt",
                ),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    model_dir,
                    f"opt-clean-{ds_name}-{mod_name}-epoch{epoch}.tar",
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
            f"data/model/model-clean-{ds_name}-{mod_name}-epoch{args.epochs}.pt"
        )
    )
    model_copy = copy.deepcopy(trained_model)
    robust_eval(model_copy, device, test_loader, ds_name, f)

    f.close()  # close the output file


if __name__ == "__main__":
    main_clean("cifar100", "res18")
