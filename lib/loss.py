# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from __future__ import print_function

from autoattack import AutoAttack

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

from .attack import create_attack


def adt_va_loss(model,
             x_natural,
             y,
             att,
             ds_name,
             optimizer,
             device):
    """
        Loss function for the ADT++ method. For each batch, the algorithm optimizes a distribution
        of adversarial examples surrounding calculated adversarial images and calculates the associated
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
    model.eval() # put the model in evaluation mode

    # Perturb the example using VA
    x_adv = va_get_xadv(model, x_natural, y, att, device, ds_name).to(device)

    # Calculate the ADT loss
    loss = adt_loss(model, x_adv, y, optimizer)

    return loss

def va_get_xadv(model, x, y, att, device, ds):
    """
        Adversarial training loss function for later various atts epoch, which simply
        calculates the loss based on the given attack.

        :param model: trained image classifier
        :param x: batch of clean images
        :param y: labels associated with images in x
        :param optimizer: optimizer to be updated
        :param att: attack to be used
        :param device: current device
        :param keep_clean: True if clean examples are kept

        :return loss to the classifier
    """

    # Set the criterion to cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Create the adversary for AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=.031, version='custom', verbose=False)

    # Based on the attack type, calculates the adversarial example
    if att == 'linf-pgd-40':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 40, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'cw': # TODO(Ezuharad): These should be moved here
        x_adv = cw_whitebox(model, x, y, device, ds)
    elif att == 'mim':
        x_adv = mim_whitebox(model, x, y, device)
    elif att == 'linf-pgd-3':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 3, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'linf-pgd-20':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 20, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'linf-pgd-7':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 7, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'apgd-ce':
        adversary.attacks_to_run = ['apgd-ce']
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == 'apgd-dlr':
        adversary.attacks_to_run = ['apgd-dlr']
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == 'apgd-t':
        adversary.attacks_to_run = ['apgd-t']
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == 'fab':
        adversary.attacks_to_run = ['fab']
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == 'fab-t':
        adversary.attacks_to_run = ['fab-t']
        x_adv = adversary.run_standard_evaluation(x, y)
    elif att == 'square':
        adversary.attacks_to_run = ['square']
        x_adv = adversary.run_standard_evaluation(x, y)
    else:
        print(att)
        raise NotImplementedError

    return x_adv

def adt_loss(model,
             x_natural,
             y,
             optimizer,
             learning_rate=1.0,
             epsilon=8.0 / 255.0,
             perturb_steps=10,
             num_samples=10,
             lbd=0.0):
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
    model.eval() # put the model in evaluation mode

    # set the mean to a set of zeros the same size as the unperturbed image
    mean = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)
    # set the var to a set of zeros the same size as the unperturbed image
    var = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)
    # set the optimizer as an Adam optimizer based on the mean and var as model parameters
    optimizer_adv = optim.Adam([mean, var], lr=learning_rate, betas=(0.0, 0.0))

    # For each perturbation step
    for _ in range(perturb_steps): # perturb steps = number of steps distribution optimizer

        # For each sample to be drawn
        for s in range(num_samples): # num_samples = number of samples drawn from distribution before optimizing
            # set adv_std to the softplus (smooth + function) of the variation
            adv_std = F.softplus(var)
            # set rand-noise to a tensor the same size as x_natural filled randomly with mean 0 and var 1
            rand_noise = torch.randn_like(x_natural)
            # set adv to the hyperbolic tan of the mean plus the random noise times standard deviation
            adv = torch.tanh(mean + rand_noise * adv_std) # Monte-Carlo sampling from distribution
            # omit the constants in -logp
            # set negative_logp to the following complex equation, representing the entropy
            negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
            entropy = negative_logp.mean()  # entropy is the average of negative_logp
            # set x_adv to the x_natural + epsilon times adv, clamped between 0 and 1
            x_adv = torch.clamp(x_natural + epsilon * adv, 0.0, 1.0)

            # minimize the negative loss
            # calculate the loss as the cross entropy between the classification of x_adv and y
            # minus the entropy times lambda, then step backwards
            with torch.enable_grad():
                loss = -F.cross_entropy(model(x_adv), y) - lbd * entropy
            loss.backward(retain_graph=True if s != num_samples - 1 else False)

        # Step the distribution optimizer
        optimizer_adv.step()

    # set x_adv to x_natural plus epsilon times the hyperbolic tangent of the mean plus the softplus of the var
    # times random numbers like x_natual, clamped between 0 and 1
    x_adv = torch.clamp(x_natural + epsilon * torch.tanh(mean + F.softplus(var) * torch.randn_like(x_natural)), 0.0,
                        1.0)
    model.train() # set the model to training mode

    # set x_adv to a variable version of x_adv clamped between 0 and 1
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad() # zero the gradient

    # Calculate the robust loss
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss

def standard_loss(model, x, y, optimizer):
    """
        Standard training loss function with no adversarial attack.

        :param model: image classifier
        :param x: batch of clean images
        :param y: correct labels for the clean images in x
        :param optimizer: optimizer for the model

        :return loss between correct labels and model output for the batch x
    """
    optimizer.zero_grad() # zero out the gradients

    # Calculate cross-entropy loss between model output and correct labels
    out = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y)

    # Format predictions and batch metrics for return
    preds = out.detach()
    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}

    return loss, batch_metrics

def accuracy(true, preds):
    """
        Computes multi-class accuracy.

        :param true: true labels
        :param preds: predicted labels
        :return multi-class accuracy
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()/float(true.size(0))
    return accuracy.item()

def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss
