# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax
import torch.nn as nn


class FAT_EARLYSTOP:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def earlystop(
        self,
        data,
        target,
        step_size,
        epsilon,
        perturb_steps,
        tau,
        randominit_type,
        loss_fn,
        rand_init=True,
        omega=0,
    ):
        """
        The implematation of early-stopped PGD
        Following the Alg.1 in our FAT paper <https://arxiv.org/abs/2002.11242>
        :param step_size: the PGD step size
        :param epsilon: the perturbation bound
        :param perturb_steps: the maximum PGD step
        :param tau: the step controlling how early we should stop interations when wrong adv data is found
        :param randominit_type: To decide the type of random inirialization (random start for searching adv data)
        :param rand_init: To decide whether to initialize adversarial sample with random noise (random start for searching adv data)
        :param omega: random sample parameter for adv data generation (this is for escaping the local minimum.)
        :return: output_adv (friendly adversarial data) output_target (targets), output_natural (the corresponding natrual data), count (average backword propagations count)

        :note if you use grayscale images then you may need to adjust the shape output_adv and output_natural
        """
        model = self.model
        model.eval()

        K = perturb_steps
        count = 0
        output_target = []
        output_adv = []
        output_natural = []

        control = (torch.ones(len(target)) * tau).to(self.device)

        # Initialize the adversarial data with random noise
        if rand_init:
            if randominit_type == "normal_distribution_randominit":
                iter_adv = (
                    data.detach()
                    + 0.001 * torch.randn(data.shape).to(self.device).detach()
                )
                iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
            if randominit_type == "uniform_randominit":
                iter_adv = data.detach() + torch.from_numpy(
                    np.random.uniform(-epsilon, epsilon, data.shape)
                ).float().to(self.device)
                iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        else:
            iter_adv = data.to(self.device).detach()

        iter_clean_data = data.to(self.device).detach()
        iter_target = target.to(self.device).detach()
        output_iter_clean_data = model(data)

        while K > 0:
            iter_adv.requires_grad_()
            output = model(iter_adv)
            pred = output.max(1, keepdim=True)[1]
            output_index = []
            iter_index = []

            # Calculate the indexes of adversarial data those still needs to be iterated
            for idx, (p, it) in enumerate(zip(pred, iter_target)):
                if p != it:
                    if control[idx] == 0:
                        output_index.append(idx)
                    else:
                        control[idx] -= 1
                        iter_index.append(idx)
                else:
                    iter_index.append(idx)

            # Add adversarial data those do not need any more iteration into set output_adv
            if len(output_index) != 0:
                if len(output_target) == 0:
                    # incorrect adv data should not keep iterated
                    output_adv = (
                        iter_adv[output_index]
                        .reshape(-1, 3, 32, 32)
                        .to(self.device)
                    )
                    output_natural = (
                        iter_clean_data[output_index]
                        .reshape(-1, 3, 32, 32)
                        .to(self.device)
                    )
                    output_target = (
                        iter_target[output_index].reshape(-1).to(self.device)
                    )
                else:
                    # incorrect adv data should not keep iterated
                    output_adv = torch.cat(
                        (
                            output_adv,
                            iter_adv[output_index]
                            .reshape(-1, 3, 32, 32)
                            .to(self.device),
                        ),
                        dim=0,
                    )
                    output_natural = torch.cat(
                        (
                            output_natural,
                            iter_clean_data[output_index]
                            .reshape(-1, 3, 32, 32)
                            .to(self.device),
                        ),
                        dim=0,
                    )
                    output_target = torch.cat(
                        (
                            output_target,
                            iter_target[output_index]
                            .reshape(-1)
                            .to(self.device),
                        ),
                        dim=0,
                    )

            # calculate gradient
            model.zero_grad()
            with torch.enable_grad():
                if loss_fn == "cent":
                    loss_adv = nn.CrossEntropyLoss(reduction="mean")(
                        output, iter_target
                    )
                if loss_fn == "kl":
                    criterion_kl = nn.KLDivLoss(size_average=False).to(
                        self.device
                    )
                    loss_adv = criterion_kl(
                        log_softmax(output, dim=1),
                        softmax(output_iter_clean_data, dim=1),
                    )
            loss_adv.backward(retain_graph=True)
            grad = iter_adv.grad

            # update iter adv
            if len(iter_index) != 0:
                control = control[iter_index]
                iter_adv = iter_adv[iter_index]
                iter_clean_data = iter_clean_data[iter_index]
                iter_target = iter_target[iter_index]
                output_iter_clean_data = output_iter_clean_data[iter_index]
                grad = grad[iter_index]
                eta = step_size * grad.sign()

                iter_adv = (
                    iter_adv.detach()
                    + eta
                    + omega
                    * torch.randn(iter_adv.shape).detach().to(self.device)
                )
                iter_adv = torch.min(
                    torch.max(iter_adv, iter_clean_data - epsilon),
                    iter_clean_data + epsilon,
                )
                iter_adv = torch.clamp(iter_adv, 0, 1)
                count += len(iter_target)
            else:
                output_adv = output_adv.detach()
                return output_adv, output_target, output_natural, count
            K = K - 1

        if len(output_target) == 0:
            output_target = iter_target.reshape(-1).squeeze().to(self.device)
            output_adv = iter_adv.reshape(-1, 3, 32, 32).to(self.device)
            output_natural = iter_clean_data.reshape(-1, 3, 32, 32).to(
                self.device
            )
        else:
            output_adv = torch.cat(
                (output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0
            ).to(self.device)
            output_target = (
                torch.cat((output_target, iter_target.reshape(-1)), dim=0)
                .squeeze()
                .to(self.device)
            )
            output_natural = torch.cat(
                (
                    output_natural,
                    iter_clean_data.reshape(-1, 3, 32, 32).to(self.device),
                ),
                dim=0,
            ).to(self.device)
        output_adv = output_adv.detach()
        return output_adv, output_target, output_natural, count
