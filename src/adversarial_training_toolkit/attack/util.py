# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).

from adversarial_training_toolkit.attack.apgd import (
    L2APGDAttack,
    LinfAPGDAttack,
)
from adversarial_training_toolkit.attack.fgsm import FGMAttack, FGSMAttack
from adversarial_training_toolkit.attack.pgd import L2PGDAttack, LinfPGDAttack


def create_attack(
    model,
    criterion,
    attack_type,
    attack_eps,
    attack_iter,
    attack_step,
    rand_init_type="uniform",
    clip_min=0.0,
    clip_max=1.0,
):
    """
     Initialize adversary.
     Arguments:
         model (nn.Module): forward pass function.
         criterion (nn.Module): loss function.
         attack_type (str): name of the attack.
         attack_eps (float): attack radius.
         attack_iter (int): number of attack iterations.
         attack_step (float): step size for the attack.
         rand_init_type (str): random initialization type for PGD (default: uniform).
         clip_min (float): mininum value per input dimension.
         clip_max (float): maximum value per input dimension.
    Returns:
        Attack
    """

    if attack_type == "fgsm":
        attack = FGSMAttack(
            model,
            criterion,
            eps=attack_eps,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    elif attack_type == "fgm":
        attack = FGMAttack(
            model,
            criterion,
            eps=attack_eps,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    elif attack_type == "linf-pgd":
        attack = LinfPGDAttack(
            model,
            criterion,
            eps=attack_eps,
            nb_iter=attack_iter,
            eps_iter=attack_step,
            rand_init_type=rand_init_type,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    elif attack_type == "l2-pgd":
        attack = L2PGDAttack(
            model,
            criterion,
            eps=attack_eps,
            nb_iter=attack_iter,
            eps_iter=attack_step,
            rand_init_type=rand_init_type,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    elif attack_type == "pgd-targeted":
        attack = LinfPGDAttack(
            model,
            criterion,
            eps=attack_eps,
            nb_iter=attack_iter,
            eps_iter=attack_step,
            rand_init_type=rand_init_type,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=True,
        )
    elif attack_type == "linf-apgd":
        attack = LinfAPGDAttack(
            model, criterion, n_restarts=2, eps=attack_eps, nb_iter=attack_iter
        )
    elif attack_type == "l2-apgd":
        attack = L2APGDAttack(
            model, criterion, n_restarts=2, eps=attack_eps, nb_iter=attack_iter
        )
    else:
        raise NotImplementedError(f"{attack_type} is not yet implemented!")
    return attack
