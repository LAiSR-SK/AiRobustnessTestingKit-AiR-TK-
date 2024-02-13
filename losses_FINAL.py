from __future__ import print_function
from core.attacks import create_attack
from autoattack import AutoAttack

from attacks import *


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
    adversary = AutoAttack(model, norm='Linf', eps=.031, version='custom', verbose=False)

    # Generate adversarial example based on specified attack
    if att == 'none':
        return standard_loss(model, x, y, optimizer)
    elif att == 'linf-pgd-40':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 40, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'linf-pgd-20':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 20, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'linf-pgd-7':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 7, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'linf-pgd-3':
        attack = create_attack(model, criterion, 'linf-pgd', 0.03, 3, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'fgsm-40':
        attack = create_attack(model, criterion, 'fgsm', 0.03, 40, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'fgsm-20':
        attack = create_attack(model, criterion, 'fgsm', 0.03, 20, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'fgsm-7':
        attack = create_attack(model, criterion, 'fgsm', 0.03, 7, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'cw':
        x_adv = cw_whitebox(model, x, y, device, ds)
    elif att == 'mim':
        x_adv = mim_whitebox(model, x, y, device)
    elif att == 'l2-pgd-40':
        attack = create_attack(model, criterion, 'l2-pgd', 0.03, 40, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'l2-pgd-20':
        attack = create_attack(model, criterion, 'l2-pgd', 0.03, 20, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'l2-pgd-7':
        attack = create_attack(model, criterion, 'l2-pgd', 0.03, 7, 0.01)
        x_adv, _ = attack.perturb(x, y)
    elif att == 'autoattack':
        aa_attack = AutoAttack(model, norm='Linf', eps=0.031, version='standard', verbose=False)
        x_adv = aa_attack.run_standard_evaluation(x, y)
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

    optimizer.zero_grad() # zero out the gradients

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

    preds = out.detach() # correctly format adversarial output

    # Calculate batch metrics to be returned
    batch_metrics = {'loss': loss.item()}
    if keep_clean:
        preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
        batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds_adv)})
    else:
        batch_metrics.update({'adversarial_acc': accuracy(y, preds)})

    return loss, batch_metrics

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