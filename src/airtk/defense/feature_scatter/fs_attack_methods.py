# (c) 2024 LAiSR-SK

# This code is licensed under the MIT license (see LICENSE.md).

import pickle


import torch

from airtk.defense.feature_scatter import fs_ot
from airtk.defense.feature_scatter.fs_utils import (
    label_smoothing,
    one_hot_tensor,
    softCrossEntropy,
)
from torch import nn

# from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"


def zero_gradients(x):
    """Manually zeroes the gradients when pytorch is unable"""

    if x.grad is not None:
        x.grad.zero_()


class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super().__init__()

        self.basic_net = basic_net

        self.attack_net = attack_net

        self.rand = config["random_start"]

        self.step_size = config["step_size"]

        self.epsilon = config["epsilon"]

        self.num_steps = config["num_steps"]
        self.train_flag = (
            True if "train" not in config.keys() else config["train"]
        )

        self.box_type = (
            "white" if "box_type" not in config.keys() else config["box_type"]
        )
        self.ls_factor = (
            0.1 if "ls_factor" not in config.keys() else config["ls_factor"]
        )

        print(config)

    def forward(
        self, inputs, targets, attack=True, targeted_label=-1, batch_idx=0
    ):
        if not attack:
            outputs, _ = self.basic_net(inputs, return_feature=True)

            return outputs, None

        if self.box_type == "white":
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        elif self.box_type == "black":
            assert (
                self.attack_net is not None
            ), "should provide an additional net in black-box case"

            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()

        batch_size = inputs.size(0)

        m = batch_size

        n = batch_size

        logits = aux_net(inputs, return_feature=True)[0]

        num_classes = logits.size(1)

        outputs = aux_net(inputs, return_feature=True)[0]

        x = inputs.detach()

        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()

        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs, return_feature=True)

        num_classes = logits_pred_nat.size(1)

        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()

        iter_num = self.num_steps

        for _ in range(iter_num):
            x.requires_grad_()

            zero_gradients(x)

            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x, return_feature=True)

            ot_loss = fs_ot.sinkhorn_loss_joint_IPOT(
                1, 0.00, logits_pred_nat, logits_pred, None, None, 0.01, m, n
            )

            aux_net.zero_grad()

            adv_loss = ot_loss

            adv_loss.backward(retain_graph=True)

            x_adv = x.data + self.step_size * torch.sign(x.grad.data)

            x_adv = torch.min(
                torch.max(x_adv, inputs - self.epsilon), inputs + self.epsilon
            )

            x_adv = torch.clamp(x_adv, -1.0, 1.0)

            x = Variable(x_adv)

            logits_pred, fea = self.basic_net(x, return_feature=True)

            self.basic_net.zero_grad()

            y_sm = label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

            adv_loss = loss_ce(logits_pred, y_sm.detach())

        return logits_pred, adv_loss
