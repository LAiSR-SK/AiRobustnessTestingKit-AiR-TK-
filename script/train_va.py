# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from __future__ import print_function

import copy
import os
import time

from helper_functions import *
from losses import *
from models.resnet import *
from models.wideresnet import *
from warmup_round import main_warmup

"""Trains a model using the Various Attacks method.

Models: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, WideResNet-34

Datasets: CIFAR-10, CIFAR-100"""

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
    default="./saved-models",
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

# Define settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


def train(
    args, model, device, train_loader, optimizer, epoch, attacks, ds_name
):
    """
    Trains one epoch by calculating the loss for each batch based on the given set
    of attacks and updating the model

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

        # Determine the 'class' of the batch by looking at the most common class
        if ds_name == "cifar100":
            avg_cls = int(torch.mode(coarse)[0])  # TODO cifar100 only
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
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def va_loss(model, x, y, optimizer, att, device, ds, keep_clean=False):
    """
    Adversarial training loss function for later various atts epoch, which simply
    calculates the loss based on the given attack.

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


def epochs_define_attacks(epoch, dataset):
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


def main(ds_name, mod_name, clean_epochs):
    """
    Main training method, which establishes the model/dataset before conducting training and
    clean testing for each epoch. Then reports final training time and conducts robustness tests
    on the final model.

    :param ds_name: dataset (cifar10 or cifar100)
    :param mod_name: training model architecture
    :param clean_pochs: number of epochs in warmup round, default 0
    """

    # Create file to print training progress
    filename = "va-epochs-{}-{}-output.txt".format(ds_name, mod_name)
    f = open(filename, "a")

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
                    "model-warmup-{}-{}-epoch{}.pt".format(
                        ds_name, mod_name, clean_epochs
                    ),
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
    train_loader, test_loader = load_data(ds_name, args, kwargs, coarse=True)

    # Save the untrained model and optimizer
    torch.save(
        model.state_dict(),
        os.path.join(
            model_dir,
            "model-va-epochs-{}-{}-start.pt".format(ds_name, mod_name),
        ),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(
            model_dir,
            "opt-va-epochs-{}-{}-start.tar".format(ds_name, mod_name),
        ),
    )

    # Begin model training
    for epoch in range(1, args.epochs + 1):
        # Adjust the SGD learning rate based on the epoch
        adjust_learning_rate(optimizer, epoch, args)

        # Define the dictionary of attack types for the current epoch
        attacks = epochs_define_attacks(epoch, ds_name)

        # Calculate the time elapsed
        curr_time = time.time()
        time_elapsed = curr_time - start_tot

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
        print(
            "================================================================"
        )
        f.write("Epoch " + str(epoch) + "\n")
        f.write("Current Attack Dictionary: ")
        f.write(str(attacks) + "\n")
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
                    "model-va-epochs-{}-{}-epoch{}.pt".format(
                        ds_name, mod_name, epoch
                    ),
                ),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    model_dir,
                    "opt-va-epochs-{}-{}-epoch{}.tar".format(
                        ds_name, mod_name, epoch
                    ),
                ),
            )

    # Report full training time
    end_tot = time.time()
    total_time = end_tot - start_tot
    print("Total training time: " + str(total_time))
    f.write("Total training time: " + str(total_time) + "\n")

    # Load the final trained model, make a deepcopy, and evaluate against FGSM and PGD attacks
    trained_model = get_model(mod_name, ds_name, device)

    trained_model.load_state_dict(
        torch.load(
            "saved-models/model-va-epochs-{}-{}-epoch{}.pt".format(
                ds_name, mod_name, args.epochs
            )
        )
    )
    model_copy = copy.deepcopy(trained_model)
    robust_eval(model_copy, device, test_loader, ds_name, f)

    f.close()  # close output file


if __name__ == "__main__":
    main("cifar10", "wideres34", 20)
