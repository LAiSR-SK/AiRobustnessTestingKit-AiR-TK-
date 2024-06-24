# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
# encoding: utf-8
# encoding: utf-8
import argparse
import copy
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from adversarial_training_toolkit.defense.lasat.las_at_cifar10_strategy import (
    ResNet18_Strategy,
)
from adversarial_training_toolkit.defense.lasat.las_at_cifar10_utils import (
    clamp,
    get_loaders_cifar100,
    lower_limit,
    std,
    upper_limit,
)

# from CIFAR100_models import *
from adversarial_training_toolkit.model import ResNet18, WideResNet
from helper_functions import evaluate_pgd, evaluate_standard
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Categorical

logger = logging.getLogger(__name__)
CUDA_LAUNCH_BLOCKING = 1

def get_args():
    parser = argparse.ArgumentParser("LAS_AT")
    # target model
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--data-dir", default="../cifar-data100", type=str)
    parser.add_argument(
        "--out-dir", default="LAS_AT", type=str, help="Output directory"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--target_model_lr", default=0.1, type=float, help="learning rate"
    )
    parser.add_argument("--epochs", default=110, type=int)
    parser.add_argument(
        "--target_model_lr_scheduler",
        default="multistep",
        type=str,
        choices=["cyclic", "multistep"],
    )
    parser.add_argument("--target_model_lr_min", default=0.0, type=float)
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument(
        "--model", default="WideResNet", type=str, help="model name"
    )

    parser.add_argument(
        "--path", default="LAS_AT", type=str, help="model name"
    )

    ## search
    parser.add_argument(
        "--attack_types",
        type=list,
        default=["IFGSM"],
        help="all searched policies",
    )
    parser.add_argument(
        "--epsilon_types", type=int, nargs="*", default=range(3, 15)
    )
    parser.add_argument(
        "--attack_iters_types", type=int, nargs="*", default=range(3, 12)
    )
    parser.add_argument(
        "--step_size_types", type=int, nargs="*", default=range(1, 5)
    )

    ## policy Hyperparameters
    parser.add_argument("--policy_model_lr", type=float, default=0.0001)
    parser.add_argument(
        "--policy_model_lr_scheduler",
        default="multistep",
        type=str,
        choices=["cyclic", "multistep"],
    )
    parser.add_argument("--policy_model_lr_min", default=0.0, type=float)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )

    parser.add_argument("--interval_num", type=int, default=20)
    parser.add_argument("--exp_iter", type=int, default=1)

    parser.add_argument(
        "--tensor-path", default="runs", type=str, help="tensorboardX name"
    )

    parser.add_argument(
        "--policy_optimizer",
        default="SGD_with_momentum",
        type=str,
        help="policy_optimizer",
    )
    parser.add_argument(
        "--factor", default=0.6, type=float, help="Label Smoothing"
    )

    parser.add_argument("--a", default=1, type=float)
    parser.add_argument("--b", default=1, type=float)
    parser.add_argument("--c", default=1, type=float)

    parser.add_argument("--R2_param", default=4, type=float)
    parser.add_argument("--R3_param", default=2, type=float)
    parser.add_argument("--clip_grad_norm", default=1.0, type=float)

    arguments = parser.parse_args()
    return arguments


args = get_args()

out_dir = os.path.join(args.out_dir, "model_" + args.model)


out_dir = os.path.join(
    out_dir,
    "epsilon_types_"
    + str(min(args.epsilon_types))
    + "_"
    + str(max(args.epsilon_types)),
)
out_dir = os.path.join(
    out_dir,
    "attack_iters_types_"
    + str(min(args.attack_iters_types))
    + "_"
    + str(max(args.attack_iters_types)),
)
out_dir = os.path.join(
    out_dir,
    "step_size_types_"
    + str(min(args.step_size_types))
    + "_"
    + str(max(args.step_size_types)),
)


tensor_path = os.path.join(out_dir, "runs")
# writer = SummaryWriter(tensor_path)

eps = np.finfo(np.float32).eps.item()


def _label_smoothing(label, factor):
    one_hot = np.eye(100)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.0) * (
        (factor - 1) / float(100 - 1)
    )

    return result


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
logfile = os.path.join(out_dir, "output.log")
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format="[%(asctime)s] - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
    filename=logfile,
)
logger.info(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = get_loaders_cifar100(
    args.data_dir, args.batch_size
)
best_acc = 0
best_loss = 0
best_clean_acc = 0
best_clean_loss = 0
start_epoch = 0


def _get_sub_policies(
    attack_id_list,
    espilon_id_list,
    attack_iters_id_list,
    step_size_id_list,
    prob_id_list,
    args,
):
    policies = []
    attack_id_list = attack_id_list[0].cpu().numpy()
    espilon_id_list = espilon_id_list[0].cpu().numpy()
    attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
    step_size_id_list = step_size_id_list[0].cpu().numpy()

    for n in range(args.subpolicy_num):
        sub_policy = {}
        for i in range(args.op_num_pre_subpolicy):
            all_policy = {}
            all_policy["attack"] = args.attack_types[attack_id_list[n + i]]
            all_policy["epsilon"] = args.epsilon_types[espilon_id_list[n + i]]
            all_policy["attack_iters"] = args.attack_iters_types[
                attack_iters_id_list[n + i]
            ]
            all_policy["step_size"] = args.step_size_types[
                step_size_id_list[n + i]
            ]

            sub_policy[i] = all_policy
        policies.append(sub_policy)
    return policies


def _get_all_policies(
    attack_id_list,
    espilon_id_list,
    attack_iters_id_list,
    step_size_id_list,
    args,
):
    policies = []
    attack_id_list = attack_id_list[0].cpu().numpy()
    espilon_id_list = espilon_id_list[0].cpu().numpy()
    attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
    step_size_id_list = step_size_id_list[0].cpu().numpy()
    # prob_id_list=prob_id_list[0].cpu().numpy()
    for n in range(len(attack_id_list)):
        sub_policy = {}

        all_policy = {}
        all_policy["attack"] = args.attack_types[attack_id_list[n]]
        all_policy["epsilon"] = args.epsilon_types[espilon_id_list[n]]

        all_policy["attack_iters"] = args.attack_iters_types[
            attack_iters_id_list[n]
        ]

        all_policy["step_size"] = args.step_size_types[step_size_id_list[n]]
        # all_policy['prob'] = args.prob_types[prob_id_list[n]]
        sub_policy[n] = all_policy
        policies.append(sub_policy)

    return policies


# class Policy(nn.Module):
#     def __init__(self, args):
#         self.args = args
#         super(Policy, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
#         self.maxp1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
#         self.maxp2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
#         self.maxp3 = nn.MaxPool2d(2, 2)
#         self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
#         self.maxp4 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.Attack_method = nn.Linear(128, len(args.attack_types))  # 所用攻击方式的个数 1
#         self.Attack_epsilon = nn.Linear(128, len(args.epsilon_types))  # 攻击强度 11
#         self.Attack_iters = nn.Linear(128, len(args.attack_iters_types))  # 迭代次数 11
#         self.Attack_step_size = nn.Linear(128, len(args.step_size_types))  # 步长
#
#     def forward(self, x):
#         in_size = x.size(0)
#         x = F.relu(self.maxp1(self.conv1(x)))
#         x = F.relu(self.maxp2(self.conv2(x)))
#         x = F.relu(self.maxp3(self.conv3(x)))
#         x = F.relu(self.maxp4(self.conv4(x)))
#         x = x.view(in_size, -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         Attack_method = self.Attack_method(x)
#         Attack_epsilon = self.Attack_epsilon(x)
#         Attack_iters = self.Attack_iters(x)
#         Attack_step_size = self.Attack_step_size(x)
#
#         return Attack_method, Attack_epsilon, Attack_iters, Attack_step_size


def select_action(policy_model, state):
    # policy_model = policy_model.eval()

    outputs = policy_model(state)
    attack_id_list = []
    espilon_id_list = []
    attack_iters_id_list = []
    step_size_id_list = []
    prob_list = []
    action_list = []

    max_attack_id_list = []
    max_espilon_id_list = []
    max_attack_iters_id_list = []
    max_step_size_id_list = []
    # max_prob_list = []
    # max_action_list = []
    temp_saved_log_probs = []
    for id in range(4):
        logits = outputs[id]
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.data.clone()
        m = Categorical(probs)

        prob_list.append(m)
        action = m.sample()

        max_action = max_probs.max(1)[1]
        mode = id % 5
        if mode == 0:
            attack_id_list.append(action)
            max_attack_id_list.append(max_action)
        elif mode == 1:
            espilon_id_list.append(action)
            max_espilon_id_list.append(max_action)
        elif mode == 2:
            attack_iters_id_list.append(action)
            max_attack_iters_id_list.append(max_action)
        elif mode == 3:
            step_size_id_list.append(action)
            max_step_size_id_list.append(max_action)
        temp_saved_log_probs.append(m.log_prob(action))
    # policy_model.saved_log_probs.append(temp_saved_log_probs)
    curpolicy = _get_all_policies(
        attack_id_list,
        espilon_id_list,
        attack_iters_id_list,
        step_size_id_list,
        args,
    )
    max_curpolicy = _get_all_policies(
        max_attack_id_list,
        max_espilon_id_list,
        max_attack_iters_id_list,
        max_step_size_id_list,
        args,
    )
    action_list.append(attack_id_list)
    action_list.append(espilon_id_list)
    action_list.append(attack_iters_id_list)
    action_list.append(step_size_id_list)

    return action_list, curpolicy, prob_list, max_curpolicy


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(
                -epsilon[i][0][0].item(), epsilon[i][0][0].item()
            )
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(
                d,
                lower_limit - X[index[0], :, :, :],
                upper_limit - X[index[0], :, :, :],
            )
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(
            model(X + delta), y, reduction="none"
        ).detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def train_target_model(input_batch, y_batch, copy_target_model):
    X, Y = input_batch.to(device), y_batch.to(device)
    label_smoothing = Variable(
        torch.tensor(_label_smoothing(Y, args.factor)).cuda()
    )
    target_lr = target_model_scheduler.get_lr()[0]
    optimizer = optim.SGD(
        copy_target_model.parameters(),
        lr=target_lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    copy_target_model.train()
    optimizer.zero_grad()
    target_output = copy_target_model(X)
    copy_target_loss = LabelSmoothLoss(target_output, label_smoothing.float())
    copy_target_loss.backward()
    optimizer.step()
    return copy_target_model


def Attack_policy(input_batch, y_batch, target_model, policies):
    criterion = nn.CrossEntropyLoss()
    X, y = input_batch.cuda(), y_batch.cuda()
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    for ii in range(len(policies)):
        epsilon = (policies[ii][ii]["epsilon"] / 255.0) / std
        alpha = (policies[ii][ii]["step_size"] / 255.0) / std

        temp_X = X[ii : ii + 1]
        temp_delta = torch.zeros_like(temp_X).cuda()
        temp_delta.requires_grad = True
        for _ in range(policies[ii][ii]["attack_iters"]):
            output = target_model(temp_X + temp_delta)
            loss = criterion(output, y[ii : ii + 1])
            loss.backward()
            grad = temp_delta.grad.detach()

            temp_delta.data = clamp(
                temp_delta + alpha * torch.sign(grad), -epsilon, epsilon
            )
            temp_delta.data = clamp(
                temp_delta, lower_limit - temp_X, upper_limit - temp_X
            )
            temp_delta.grad.zero_()
        temp_delta = temp_delta.detach()
        delta[ii : ii + 1] = temp_delta
    delta = delta.detach()
    return delta


def Attack_policy_batch(input_batch, y_batch, target_model, policies):
    criterion = nn.CrossEntropyLoss()
    X, y = input_batch.cuda(), y_batch.cuda()
    delta_batch = torch.zeros_like(X).cuda()

    init_epsilon = (8 / 255.0) / std
    for i in range(len(init_epsilon)):
        delta_batch[:, i, :, :].uniform_(
            -init_epsilon[i][0][0].item(), init_epsilon[i][0][0].item()
        )
    delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
    delta_batch.requires_grad = True
    alpha_batch = []
    epsilon_batch = []
    attack_iters_batch = []
    for ii in range(len(policies)):
        epsilon = (policies[ii][ii]["epsilon"] / 255.0) / std
        epsilon_batch.append(epsilon.cpu().numpy())

        alpha = (policies[ii][ii]["step_size"] / 255.0) / std
        alpha_batch.append(alpha.cpu().numpy())
        attack_iters = policies[ii][ii]["attack_iters"]
        temp_batch = torch.randint(attack_iters, attack_iters + 1, (3, 1, 1))
        attack_iters_batch.append(temp_batch.cpu().numpy())
    alpha_batch = torch.from_numpy(np.array(alpha_batch)).cuda()
    epsilon_batch = torch.from_numpy(np.array(epsilon_batch)).cuda()
    attack_iters_batch = torch.from_numpy(
        np.array(attack_iters_batch)
    ).cuda()

    max_attack_iters = torch.max(attack_iters_batch).cpu().numpy()
    for _ in range(max_attack_iters):
        mask_bacth = attack_iters_batch.ge(1).float()
        output = target_model(X + delta_batch)
        loss = criterion(output, y)

        loss.backward()
        grad = delta_batch.grad.detach()
        delta_batch.data = clamp(
            delta_batch + mask_bacth * alpha_batch * torch.sign(grad),
            -epsilon_batch,
            epsilon_batch,
        )
        delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
        attack_iters_batch = attack_iters_batch - 1
        delta_batch.grad.zero_()
    delta_batch = delta_batch.detach()

    return delta_batch


def Get_delta(input_batch, y_batch, target_model, action):
    target_model.eval()
    inputs= input_batch.cuda()
    delta = Attack_policy_batch(input_batch, y_batch, target_model, action)
    return inputs + delta


def Get_reward(input_batch, y_batch, target_model, action):
    target_model.eval()
    criterion = nn.CrossEntropyLoss()
    inputs, targets = input_batch.cuda(), y_batch.cuda()
    delta = Attack_policy_batch(input_batch, y_batch, target_model, action)
    with torch.no_grad():
        output = target_model(inputs + delta)
    # logsoftmax_func = nn.LogSoftmax(dim=1)
    # soft_output = logsoftmax_func(output)
    # y_one_hot = F.one_hot(y_batch, 10).float()
    R1 = criterion(output, targets)  #### R1的奖励函数
    R1 = torch.clamp(R1, 0, 10)

    copy_target_model = copy.deepcopy(target_model)
    copy_target_model.train()
    # train_target_model(input_batch, y_batch, copy_target_model, proxy, args, epoch, lr, proxy_lr)

    copy_target_model = train_target_model(
        inputs + delta, targets, copy_target_model
    )
    epsilon = (8 / 255.0) / std
    alpha = (2 / 255.0) / std
    pgd_delta = attack_pgd(
        copy_target_model, inputs, targets, epsilon, alpha, 10, 2
    )
    copy_target_model.eval()
    with torch.no_grad():
        R2_output = copy_target_model(inputs + pgd_delta)
        clean_output = copy_target_model(inputs)
    # # logsoftmax_func = nn.LogSoftmax(dim=1)
    # # soft_output = logsoftmax_func(output)
    # # y_one_hot = F.one_hot(y_batch, 10).float()
    # R2 = criterion(R2_output, targets) #### R2的奖励函数
    # R3=criterion(clean_output, targets)
    R2 = (R2_output.max(1)[1] == targets).sum().item()
    R3 = (clean_output.max(1)[1] == targets).sum().item()

    test_n = targets.size(0)
    R2 = (R2) / test_n * args.R2_param
    R3 = (R3) / test_n * args.R3_param

    R2 = torch.clamp(torch.tensor(R2), -10, 10)
    R3 = torch.clamp(torch.tensor(R3), -10, 10)
    return (
        (args.a * R1 + args.b * R2 + args.c * R3),
        R1,
        R2,
        R3,
        inputs + delta,
    )


Strategy_model = ResNet18_Strategy(args)
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = range(torch.cuda.device_count())
if len(device_id) > 1:
    Strategy_model = torch.nn.DataParallel(Strategy_model)

Strategy_model.cuda()
Strategy_model.train()
if args.policy_optimizer == "SGD_without_momentum":
    policy_optimizer = optim.SGD(
        Strategy_model.parameters(), lr=args.policy_model_lr
    )
elif args.policy_optimizer == "SGD_with_momentum":
    policy_optimizer = optim.SGD(
        [
            {
                "params": Strategy_model.parameters(),
                "initial_lr": args.policy_model_lr,
            }
        ],
        lr=args.policy_model_lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
elif args.policy_optimizer == "Adam_with_momentum":
    policy_optimizer = torch.optim.Adam(
        Strategy_model.parameters(), lr=args.policy_model_lr, weight_decay=5e-4
    )
# 所用攻击方式的个数  攻击强度 迭代次数 步长

# logger.info('==> Building model..')
if args.model == "VGG":
    # target_model = VGG('VGG19')
    raise NotImplementedError
elif args.model == "ResNet18":
    target_model = ResNet18()
elif args.model == "PreActResNest18":
    # target_model = PreActResNet18()
    raise NotImplementedError
elif args.model == "WideResNet":
    target_model = WideResNet(num_classes=100)

if len(device_id) > 1:
    target_model = torch.nn.DataParallel(target_model)
target_model = target_model.to(device)
criterion = nn.CrossEntropyLoss()
target_model_optimizer = optim.SGD(
    [
        {
            "params": target_model.parameters(),
            "initial_lr": args.target_model_lr,
        }
    ],
    lr=args.target_model_lr,
    momentum=0.9,
    weight_decay=5e-4,
)

target_model_path = os.path.join(out_dir, "target_model_ckpt.t7")
# .info(target_model_path)
policy_model_path = os.path.join(out_dir, "policy_model_ckpt.t7")


# if os.path.exists(target_model_path):
#     #logger.info("resuming............................................")
#     target_model_checkpoint = torch.load(target_model_path)
#     last_epoch=target_model_checkpoint['epoch']
#    # logger.info(last_epoch)
#     start_epoch = last_epoch+1
#     last_epoch=start_epoch*len(train_loader)
#     try:
#         target_model.load_state_dict(target_model_checkpoint['net'])
#     except:
#         new_state_dict = OrderedDict()
#         for k, v in target_model_checkpoint['net'].items():
#             name = k[7:]  # remove `module.`
#             new_state_dict[name] = v
#         target_model.load_state_dict(new_state_dict,False)
#     policy_model_checkpoint=torch.load(policy_model_path)
#     try:
#         Policy_model.load_state_dict( policy_model_checkpoint['net'])
#     except:
#         new_state_dict = OrderedDict()
#         for k, v in  policy_model_checkpoint['net'].items():
#             name = k[7:]  # remove `module.`
#             new_state_dict[name] = v
#         Policy_model.load_state_dict(new_state_dict,False)
lr_steps = args.epochs * len(train_loader)
if args.target_model_lr_scheduler == "cyclic":
    target_model_scheduler = torch.optim.lr_scheduler.CyclicLR(
        target_model_optimizer,
        base_lr=args.target_model_lr_min,
        max_lr=args.target_model_lr,
        step_size_up=lr_steps / 2,
        step_size_down=lr_steps / 2,
    )
elif args.target_model_lr_scheduler == "multistep":
    target_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_model_optimizer,
        milestones=[int(lr_steps * 99 / 110), int(lr_steps * 104 / 110)],
        gamma=0.1,
    )
policy_lr_steps = int(
    args.epochs * len(train_loader) * args.exp_iter / args.interval_num
)
if args.policy_model_lr_scheduler == "cyclic":
    policy_model_scheduler = torch.optim.lr_scheduler.CyclicLR(
        policy_optimizer,
        base_lr=args.policy_model_lr_min,
        max_lr=args.policy_model_lr,
        step_size_up=policy_lr_steps / 2,
        step_size_down=policy_lr_steps / 2,
    )
elif args.policy_model_lr_scheduler == "multistep":
    policy_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        policy_optimizer,
        milestones=[int(lr_steps * 99 / 110), int(lr_steps * 104 / 110)],
        gamma=0.1,
    )

if os.path.exists(target_model_path):
    logger.info("resuming............................................")
    # start_epoch = args.resume
    target_model_path = os.path.join(out_dir, "target_model_ckpt.t7")
    target_model_checkpoint = torch.load(target_model_path)
    start_epoch = target_model_checkpoint["epoch"]
    try:
        target_model.load_state_dict(target_model_checkpoint["net"])
    except Exception:
        new_state_dict = OrderedDict()
        for k, v in target_model_checkpoint["net"].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        target_model.load_state_dict(new_state_dict, False)
    policy_model_path = os.path.join(out_dir, "policy_model_ckpt.t7")
    policy_model_checkpoint = torch.load(policy_model_path)
    try:
        Strategy_model.load_state_dict(policy_model_checkpoint["net"])
    except Exception:
        new_state_dict = OrderedDict()
        for k, v in policy_model_checkpoint["net"].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        Strategy_model.load_state_dict(new_state_dict, False)

    target_model_optimizer_path = os.path.join(
        out_dir, "target_model_optimizer.pth"
    )
    target_model_optimizer.load_state_dict(
        torch.load(target_model_optimizer_path)
    )
    # torch.save(policy_optimizer.state_dict(), os.path.join(out_dir, 'policy_model_optimizer.pth'))
    policy_optimizer_path = os.path.join(out_dir, "policy_model_optimizer.pth")
    policy_optimizer.load_state_dict(torch.load(policy_optimizer_path))

    target_model_scheduler_path = os.path.join(
        out_dir, "target_model_scheduler.pth"
    )
    target_model_scheduler.load_state_dict(
        torch.load(target_model_scheduler_path)
    )

    policy_model_scheduler_path = os.path.join(
        out_dir, "policy_model_scheduler.pth"
    )
    policy_model_scheduler.load_state_dict(
        torch.load(policy_model_scheduler_path)
    )

    # if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
    #     best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
    # if args.val:
    #     best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
    best_target_model_path = os.path.join(out_dir, "best_target_model_ckpt.t7")
    best_target_model_checkpoint = torch.load(best_target_model_path)

    best_acc = best_target_model_checkpoint["best_acc"]
    best_clean_acc = best_target_model_checkpoint["best_clean_acc"]
    logger.info("Test Acc  \t PGD Acc")
    logger.info("%.4f \t  \t %.4f", best_clean_acc, best_acc)
else:
    start_epoch = 0


global curr_step
curr_step = 0

def train(epoch):
    print("\nEpoch: %d" % epoch)
    # logger.info('\nEpoch: %d' % epoch)
    start_epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        global curr_step
        curr_step = curr_step + 1
        if curr_step % args.interval_num == 0:
            # while num_iter < 1:

            # outputs = net(inputs)
            #####训练policy model
            # logger.info("*******************train policy model**********************")
            # epis_rewards = []
            pocliy_inputs = inputs.clone().cuda()

            Strategy_model.train()

            action_list, policy_outputs, policy_prob, max_policy_outputs = (
                select_action(Strategy_model, pocliy_inputs)
            )

            reward, R1, R2, R3, adv_examples = Get_reward(
                pocliy_inputs, targets, target_model, policy_outputs
            )

            # baseline_reword, R1_baseline, R2_baseline, R3_baseline = Get_reward(inputs, targets, target_model,max_policy_outputs)
            # epis_rewards.append(reward)
            log_probs = []
            policy_loss = []
            for j in range(4):
                log_probs.append(policy_prob[j].log_prob(action_list[j][0]))
                policy_loss.append(-log_probs[j] * reward)
            # logger.info(action_list)
            policy_loss = (
                policy_loss[0].mean()
                + policy_loss[1].mean()
                + policy_loss[2].mean()
                + policy_loss[3].mean()
            )

            policy_optimizer.zero_grad()
            # torch.nn.utils.clip_grad_norm_(Policy_model.parameters(), 5.0)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                Strategy_model.parameters(), args.clip_grad_norm
            )
            policy_optimizer.step()
            policy_model_scheduler.step()

        #####训练target model
        # logger.info("*******************train target model**********************")

        # outputs = Policy_model(inputs)
        pocliy_inputs1 = inputs.clone().cuda()
        for _ in range(args.exp_iter):
            Strategy_model.eval()

            action_list, policy_outputs, policy_prob, max_policy_outputs = (
                select_action(Strategy_model, pocliy_inputs1)
            )
            # logger.info(policy_outputs)
            adv_examples = Get_delta(
                pocliy_inputs1, targets, target_model, policy_outputs
            )
            pocliy_inputs1 = adv_examples
        # action_list, cur_policies, policy_prob, max_policy_outputs = select_action(Policy_model, inputs)
        # cur_delta = Attack_policy_batch(inputs, targets, target_model, cur_policies)
        # cur_delta = cur_delta.detach()
        target_model.train()
        target_model.zero_grad()
        label_smoothing = Variable(
            torch.tensor(_label_smoothing(targets, args.factor)).cuda()
        )
        target_output = target_model(adv_examples)
        target_loss = LabelSmoothLoss(target_output, label_smoothing.float())

        target_model_optimizer.zero_grad()

        target_loss.backward()
        target_model_optimizer.step()
        target_model_scheduler.step()

        train_loss += target_loss.item() * targets.size(0)
        train_acc += (target_output.max(1)[1] == targets).sum().item()
        train_n += targets.size(0)

    epoch_time = time.time()

    lr = target_model_scheduler.get_lr()[0]
    logger.info(
        "%d \t %.1f \t \t %.4f \t %.4f \t %.4f",
        epoch,
        epoch_time - start_epoch_time,
        lr,
        train_loss / train_n,
        train_acc / train_n,
    )


pgd_acc_list = []
clean_acc_list = []


def test(epoch):
    global best_acc
    global best_clean_acc
    target_model.eval()
    test_loss = 0
    # pgd_loss, pgd_acc = evaluate_pgd(test_loader, target_model, 20, 1)
    # test_loss, test_acc = evaluate_standard(test_loader, target_model)
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, target_model, 10, 1)
    pgd_acc_list.append(pgd_acc)
    test_loss, test_acc = evaluate_standard(test_loader, target_model)
    clean_acc_list.append(test_acc)

    acc = pgd_acc
    state = {
        "net": target_model.state_dict(),
        "acc": acc,
        "epoch": epoch,
    }

    state1 = {
        "net": Strategy_model.state_dict(),
    }

    target_path = os.path.join(out_dir, "target_model_ckpt.t7")
    policy_path = os.path.join(out_dir, "policy_model_ckpt.t7")
    torch.save(state, target_path)
    torch.save(state1, policy_path)
    torch.save(
        target_model_optimizer.state_dict(),
        os.path.join(out_dir, "target_model_optimizer.pth"),
    )
    torch.save(
        policy_optimizer.state_dict(),
        os.path.join(out_dir, "policy_model_optimizer.pth"),
    )

    torch.save(
        policy_model_scheduler.state_dict(),
        os.path.join(out_dir, "policy_model_scheduler.pth"),
    )
    torch.save(
        target_model_scheduler.state_dict(),
        os.path.join(out_dir, "target_model_scheduler.pth"),
    )

    # Save checkpoint.
    # Save checkpoint.

    # logger.info('Test acc: ', test_acc)
    # logger.info('Val acc: ', acc)
    if acc >= best_acc:
        # logger.info("Saving..")
        state = {
            "net": target_model.state_dict(),
            "best_clean_acc": test_acc,
            "best_acc": acc,
            "epoch": epoch,
        }

        state1 = {
            "net": Strategy_model.state_dict(),
        }

        target_path = os.path.join(out_dir, "best_target_model_ckpt.t7")
        policy_path = os.path.join(out_dir, "best_policy_model_ckpt.t7")
        torch.save(state, target_path)
        torch.save(state1, policy_path)
        best_acc = acc
        best_clean_acc = test_acc

    # logger.info(best_acc)
    # logger.info(test_acc)
    logger.info("Test Loss \t Test Acc \t PGD Loss \t PGD Acc")
    logger.info(
        "%.4f \t \t %.4f \t %.4f \t %.4f",
        test_loss,
        test_acc,
        pgd_loss,
        pgd_acc,
    )
    logger.info("Test Acc  \t PGD Acc")
    logger.info("%.4f \t  \t %.4f", best_clean_acc, best_acc)

    return best_acc


def main_lasat100():
    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        print(
            "*****************************************test*************************"
        )
        # logger.info(("*****************************************test*************************"))
        result_acc = test(epoch)
        print(result_acc)

        model_dir = "../data/model"
        torch.save(
            target_model.state_dict(),
            os.path.join(
                model_dir,
                "model-lasat-{}-{}-epoch{}.pt".format(
                    "cifar100", "wideres34", epoch
                ),
            ),
        )
    logger.info(pgd_acc_list)
    logger.info(clean_acc_list)
    # print(pgd_acc_list)
    # print(clean_acc_list)


if __name__ == "__main__":
    main_lasat100()
