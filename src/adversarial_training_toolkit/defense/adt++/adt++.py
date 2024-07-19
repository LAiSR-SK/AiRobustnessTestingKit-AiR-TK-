# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from typing import Final, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Module, Tensor, Variable, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

CLASS_TO_ATTACK: Final[List[str]] = [
    "mim",
    "linf-pgd-20",
    "linf-pgd-40",
    "cw",
    "apgd-ce",
    "linf-pgd-20",
    "linf-pgd-7",
    "apgd-dlr",
    "apgd-t",
    "linf-pgd-40",
    "mim",
    "linf-pgd-20",
    "linf-pgd-40",
    "cw",
    "apgd-ce",
    "linf-pgd-20",
    "linf-pgd-7",
    "apgd-dlr",
    "apgd-t",
    "linf-pgd-40",
]


def train_adtpp(
    model: Module,
    data: DataLoader,
    optimizer: Optimizer,
    batch_size: int = 128,
    weight_decay: float = 2e-4,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    schedule: str = "decay",
    yield_every: int = 1,
) -> Iterable[Module]:
    device: Final[torch.device] = model.weight.device
    while True:
        model.train()

        for batch in data:
            x: Final[Tensor] = batch[0].to(device, non_blocking=True)
            y: Final[Tensor] = batch[1].to(device, non_blocking=True)

            if len(batch) == 3:
                coarse: Final[Tensor] = batch[2].to(device, non_blocking=True)
                avg_class: Final[int] = int(torch.mode(coarse)[0])
            else:
                avg_class: Final[int] = int(torch.mode(y)[0])

        optimizer.zero_grad()

        loss = add


def adt_loss(
    model,
    x,
    y,
    optimizer,
    learning_rate=1.0,
    epsilon=8.0 / 255.0,
    perturb_steps=10,
    num_samples=10,
    lbd=0.0,
):
    """
    Loss function for the Adversarial Distributional Training method. For each batch, the algorithm
    optimizes a distribution of adversarial examples surrounding the images and calculates the associated
    loss to the model.

    :param model: trained image classifier
    :param x_natural: batch of clean images from the training set
    :param y: set of correct labels for the clean images
    :param optimizer: optimizer for gradient calculation
    :param learning_rate: learning rate for the optimizer for distribution optimization
    :param epsilon: epsilon value for calculating perturbations
    :param perturb_steps: number of steps to use to calculate the distribution
    :param num_samples: number of samples to draw for distribution calculation
    :param lbd: lambda entropy multiplier

    :return cross-entropy loss between adversarial example drawn from distribution and correct labels
    """
    model.eval()  # put the model in evaluation mode

    # set the mean to a set of zeros the same size as the unperturbed image
    mean = Variable(torch.zeros(x.size()).cuda(), requires_grad=True)
    # set the var to a set of zeros the same size as the unperturbed image
    var = Variable(torch.zeros(x.size()).cuda(), requires_grad=True)
    # set the optimizer as an Adam optimizer based on the mean and var as model parameters
    optimizer_adv = optim.Adam([mean, var], lr=learning_rate, betas=(0.0, 0.0))

    # For each perturbation step
    for _ in range(
        perturb_steps
    ):  # perturb steps = number of steps distribution optimizer
        # For each sample to be drawn
        for s in range(
            num_samples
        ):  # num_samples = number of samples drawn from distribution before optimizing
            # set adv_std to the softplus (smooth + function) of the variation
            adv_std = F.softplus(var)
            # set rand-noise to a tensor the same size as x_natural filled randomly with mean 0 and var 1
            rand_noise = torch.randn_like(x)
            # set adv to the hyperbolic tan of the mean plus the random noise times standard deviation
            adv = torch.tanh(
                mean + rand_noise * adv_std
            )  # Monte-Carlo sampling from distribution
            # omit the constants in -logp
            # set negative_logp to the following complex equation, representing the entropy
            negative_logp = (
                (rand_noise**2) / 2.0
                + (adv_std + 1e-8).log()
                + (1 - adv**2 + 1e-8).log()
            )
            entropy = (
                negative_logp.mean()
            )  # entropy is the average of negative_logp
            # set x_adv to the x_natural + epsilon times adv, clamped between 0 and 1
            x_adv = torch.clamp(x + epsilon * adv, 0.0, 1.0)

            # minimize the negative loss
            # calculate the loss as the cross entropy between the classification of x_adv and y
            # minus the entropy times lambda, then step backwards
            with torch.enable_grad():
                loss = -F.cross_entropy(model(x_adv), y) - lbd * entropy
            loss.backward(retain_graph=s != num_samples - 1)

        # Step the distribution optimizer
        optimizer_adv.step()

    # set x_adv to x_natural plus epsilon times the hyperbolic tangent of the mean plus the softplus of the var
    # times random numbers like x_natual, clamped between 0 and 1
    x_adv = torch.clamp(
        x + epsilon * torch.tanh(mean + F.softplus(var) * torch.randn_like(x)),
        0.0,
        1.0,
    )
    model.train()  # set the model to training mode

    # set x_adv to a variable version of x_adv clamped between 0 and 1
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()  # zero the gradient

    # Calculate the robust loss
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss


def adjust_learning_rate(
    optimizer: Optimizer, epoch: int, lr: float, lr_schedule: str
) -> None:
    """
    Sets the learning rate of the optimizer based on the current epoch

    :param optimizer: optimizer with learning rate being set
    :param epoch: current epoch
    :param lr: the current learning rate
    :param lr_schedule: the scheduler to use for the learning rate.
    """
    if lr_schedule == "decay":
        if epoch >= 50:
            lr = lr * 0.1
        if epoch >= 75:
            lr = lr * 0.01
    elif lr_schedule == "scheduled":
        if epoch >= 24:
            lr = lr * 0.1
        if epoch >= 26:
            lr = lr
        if epoch >= 64:
            lr = lr * 0.1
        if epoch >= 66:
            lr = lr
        if epoch >= 104:
            lr = lr * 0.1
        if epoch >= 106:
            lr = lr
        if epoch >= 137:
            lr = lr * 0.1
        if epoch >= 139:
            lr = lr * 0.01
    elif lr_schedule == "cosine":
        lr = 0.2
        lr = lr * 0.5 * (1 + np.cos((epoch - 1) / epoch * np.pi))
    elif lr_schedule == "cyclic-10":
        if epoch % 10 < 2 or epoch % 10 == 9:
            lr = lr
        elif epoch % 10 < 4 or epoch % 10 > 6:
            lr = lr * 0.1
        elif epoch % 10 == 5:
            lr = lr * 0.001
        else:
            lr = lr * 0.01
    elif lr_schedule == "cyclic-5":
        if epoch % 5 == 0:
            lr = lr
        elif epoch % 5 == 1 or epoch % 5 == 4:
            lr = lr * 0.1
        else:
            lr = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
