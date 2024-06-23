import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class Hamiltonian(_Loss):
    def __init__(self, layer, reg_cof=1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):
        y = self.layer(x)
        # l2 = cal_l2_norm(self.layer)

        # print(y.shape, p.shape)
        H = torch.sum(y * p)

        # H = H - self.reg_cof * l2
        return H


class CrossEntropyWithWeightPenlty(_Loss):
    def __init__(self, module, DEVICE, reg_cof=1e-4):
        super(CrossEntropyWithWeightPenlty, self).__init__()

        self.reg_cof = reg_cof
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.module = module
        # print(modules, 'dwadaQ!')

    def __call__(self, pred, label):
        cross_loss = self.criterion(pred, label)
        weight_loss = 0
        # for module in self.module:
        #    print(module)
        #    weight_loss = weight_loss + cal_l2_norm(module)

        weight_loss = cal_l2_norm(self.module)

        loss = cross_loss + self.reg_cof * weight_loss
        return loss


def cal_l2_norm(layer: torch.nn.Module):
    loss = 0.0
    for name, param in layer.named_parameters():
        if name == "weight":
            loss = (
                loss
                + 0.5
                * torch.norm(
                    param,
                )
                ** 2
            )

    return loss
