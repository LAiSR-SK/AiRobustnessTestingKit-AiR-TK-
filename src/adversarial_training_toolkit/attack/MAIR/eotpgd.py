# (c) 2023 Harry24k
# This code is licensed under the MIT license (see LICENSE.md).
import torch
import torch.nn as nn

from adversarial_training_toolkit.attack.MAIR_attack import Attack


class EOTPGD(Attack):
    """Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"
    [https://arxiv.org/abs/1907.00895]
    This implementaion has been taken from https://github.com/Harry24k/MAIR/blob/main/mair/attacks/attacks/eotpgd.py

    Distance Measure : Linf

    :param model: (nn.Module): model to attack.
    :param eps: (float): maximum perturbation. (Default: 8/255)
    :param alpha: (float): step size. (Default: 2/255)
    :param steps: (int): number of steps. (Default: 10)
    :param eot_iter (int) : number of models to estimate the mean gradient. (Default: 2)
        useful when the model includes some form of randomness, such as dropout or data augmentation

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 10,
        eot_iter: int = 2,
        random_start: bool = True,
    ):
        super().__init__("EOTPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.eot_iter = eot_iter
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        """Overridden of the base."""

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )  # nopep8
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)
            adv_images.requires_grad = True

            for j in range(self.eot_iter):
                outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad += torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

            # (grad/self.eot_iter).sign() == grad.sign()
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(
                adv_images - images, min=-self.eps, max=self.eps
            )
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
