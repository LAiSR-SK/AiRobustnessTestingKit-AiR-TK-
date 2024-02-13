from __future__ import print_function
#import os
import argparse
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
from torch.autograd import Variable
import torch.optim as optim
#from torchvision import datasets, transforms
#from models.wideresnet import *
#from models.resnet import *
import numpy as np

"""File containing several different attack methods, which take in a clean image and return an adversarial
example, as well as several helper functions"""

parser = argparse.ArgumentParser(description='PyTorch CIFAR Attacks')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=8.0 / 255.0,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2.0 / 255.0,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
args = parser.parse_args()

# Establish the settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

def one_hot_tensor(y_batch_tensor, num_classes, device):
    """
        Converts a batch tensor into a one-hot tensor

        :param y_batch_tensor: batch tensor to be converted
        :param num_classes: number of classes to fill the tensor

        :return one-hot tensor based on the input batch tensor
    """

    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    """Class for the CW loss, including parameters as well as a method to propogate
    the loss forwards"""

    def __init__(self, num_classes, margin=50, reduce=True):
        """
        Set up the CW loss with number of classes, margin of 50, and reduce = True
        """
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        # Convert the target labels to a one-hot tensor
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        # Calculate self-loss (sum of targets/predictions)
        self_loss = torch.sum(onehot_targets * logits, dim=1)

        # Calculate other-loss
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        # Take the loss as the reverse sum of the loss differences plus the margin (clamped)
        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        # If reducing (default Truue), divide the loss by the number of targets
        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

def cw_whitebox(model,
                 X,
                 y,
                 device,
                 dataset,
                 epsilon=args.epsilon,
                 num_steps=args.num_steps,
                 step_size=args.step_size):
    """
        Attacks the specified image X using the CW attack and returns the adversarial example

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the PGD attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps

        :return adversarial example found with the CW attack
    """

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If adding random, create random noice between - and + epsilon and add to X_pgd
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # For each perturbation step:
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the CW loss and step backward
        with torch.enable_grad():
            loss = CWLoss(100 if dataset == 'cifar100' else 10)(model(X_pgd), y)
        loss.backward()

        # Calculate the perturbation eta as the step size in the gradient direction of X_pgd
        eta = step_size * X_pgd.grad.data.sign()

        # Add the perturbation to X_pgd
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Set the perturbation to the difference between X and X_adv, clamped by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd again
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def mim_whitebox(model,
                  X,
                  y,
                  device,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  decay_factor=1.0):
    """
        Attacks the specified image X using the MIM attack and returns the adversarial example

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the PGD attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps
        :param decay_factor: factor of decay for gradients

        :return adversarial example found with the MIM attack
    """

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If adding random, create random noice between - and + epsilon and add to X_pgd
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # Set the previous gradient to a tensor of 0s in the shape of X
    previous_grad = torch.zeros_like(X.data)

    # For each perturbation step:
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross-entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        # Calculate the gradient by dividing it by the average
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1, 2, 3], keepdim=True)

        # Calculate the previous gradient by multiplying it by the decay factor and adding to the grad
        previous_grad = decay_factor * previous_grad + grad

        # Perturb X_pgd in the direction of the previous grad, by the step size
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)

        # Set the perturbation to the difference between X and X_adv, clamped by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd againt
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def pgd_whitebox(model,
                  X,
                  y,
                  device,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    """
        Attacks the specified image X using the PGD attack and returns the adversarial example

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the PGD attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps

        :return adversarial example found with the PGD attack
    """

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If adding random, create random noice between - and + epsilon and add to X_pgd
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # For each perturbation step:
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        # Calculate the perturbation eta as the step size in the gradient direction of X_pgd
        eta = step_size * X_pgd.grad.data.sign()

        # Add the perturbation to X_pgd
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Set the perturbation to the difference between X and X_adv, clamped by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd again
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd