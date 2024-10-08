# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).

import os
import time
from collections import namedtuple
from os import PathLike

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from airtk.defense.tradesawp.trades_awp_utils_awp import (
    TradesAWP,
)
from airtk.defense.tradesawp.trades_awp_utils_eval import (
    accuracy,
)
from airtk.defense.tradesawp.trades_awp_utils_logger import (
    Logger,
)
from airtk.defense.tradesawp.trades_awp_utils_misc import (
    AverageMeter,
)

# from utils import Bar, Logger, AverageMeter, accuracy
# import models
from airtk.model import WideResNet


class TradesawpTraining:
    def __init__(
        self,
        dataset: str,
        model: str,
        batch_size: int = 128,
        epochs: int = 200,
        weight_decay: float = 5e-4,
        lr: float = 0.1,
        momentum: float = 0.9,
        norm: str = "l_inf",
        epsilon: int = 8,
        num_steps: int = 10,
        step_size: int = 2,
        beta: float = 6.0,
        seed: int = 1,
        save_freq: int = 1,
        awp_gamma: float = 0.005,
        awp_warmup: int = 10,
        model_dir: PathLike = "model/data",
    ) -> None:
        ArgsPrototype = namedtuple(
            "ArgsPrototype",
            [
                "arch",
                "batch_size",
                "test_batch_size",
                "epochs",
                "start_epoch",
                "data",
                "data_path",
                "weight_decay",
                "lr",
                "momentum",
                "no_cuda",
                "norm",
                "epsilon",
                "num_steps",
                "step_size",
                "beta",
                "seed",
                "model_dir",
                "resume_model",
                "resume_optim",
                "save_freq",
                "awp_gamma",
                "awp_warmup",
            ],
        )
        self._args = ArgsPrototype(
            model,
            batch_size,
            batch_size,
            epochs,
            1,
            dataset,
            "data",
            weight_decay,
            lr,
            momentum,
            False,
            norm,
            epsilon,
            num_steps,
            step_size,
            beta,
            seed,
            model_dir,
            model_dir,
            model_dir,
            save_freq,
            awp_gamma,
            awp_warmup,
        )

    def __call__(self) -> None:
        torch.manual_seed(self._args.seed)
        main_tradesawp(self._args)


# parser = argparse.ArgumentParser(
# description="PyTorch CIFAR TRADES Adversarial Training"
# )
# parser.add_argument("--arch", type=str, default="WideResNet34")
# parser.add_argument(
# "--batch-size",
# type=int,
# default=128,
# metavar="N",
# help="input batch size for training (default: 128)",
# )
# parser.add_argument(
# "--test-batch-size",
# type=int,
# default=128,
# metavar="N",
# help="input batch size for testing (default: 128)",
# )
# parser.add_argument(
# "--epochs",
# type=int,
# default=200,
# metavar="N",
# help="number of epochs to train",
# )
# parser.add_argument(
# "--start_epoch",
# type=int,
# default=1,
# metavar="N",
# help="retrain from which epoch",
# )
# parser.add_argument(
# "--data", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"]
# )
# parser.add_argument(
# "--data-path",
# type=str,
# default="../data",
# help="where is the dataset CIFAR-10",
# )
# parser.add_argument(
# "--weight-decay", "--wd", default=5e-4, type=float, metavar="W"
# )
# parser.add_argument(
# "--lr", type=float, default=0.1, metavar="LR", help="learning rate"
# )
# parser.add_argument(
# "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
# )
# parser.add_argument(
# "--no-cuda",
# action="store_true",
# default=False,
# help="disables CUDA training",
# )
# parser.add_argument(
# "--norm",
# default="l_inf",
# type=str,
# choices=["l_inf", "l_2"],
# help="The threat model",
# )
# parser.add_argument("--epsilon", default=8, type=float, help="perturbation")
# parser.add_argument(
# "--num-steps", default=10, type=int, help="perturb number of steps"
# )
# parser.add_argument(
# "--step-size", default=2, type=float, help="perturb step size"
# )
# parser.add_argument(
# "--beta",
# default=6.0,
# type=float,
# help="regularization, i.e., 1/lambda in TRADES",
# )
# parser.add_argument(
# "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
# )
# parser.add_argument(
# "--model-dir",
# default="./data/model",
# help="directory of model for saving checkpoint",
# )
# parser.add_argument(
# "--resume-model",
# default="",
# type=str,
# help="directory of model for retraining",
# )
# parser.add_argument(
# "--resume-optim",
# default="",
# type=str,
# help="directory of optimizer for retraining",
# )
# parser.add_argument(
# "--save-freq",
# "-s",
# default=1,
# type=int,
# metavar="N",
# help="save frequency",
# )

# parser.add_argument(
# "--awp-gamma",
# default=0.005,
# type=float,
# help="whether or not to add parametric noise",
# )
# parser.add_argument(
# "--awp-warmup",
# default=10,
# type=int,
# help="We could apply AWP after some epochs for accelerating.",
# )
# args = parser.parse_args()


def setup(data, args):
    args.data = data
    epsilon = args.epsilon / 255
    step_size = args.step_size / 255
    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty
    NUM_CLASSES = 100 if args.data == "CIFAR100" else 10
    return args, epsilon, step_size, NUM_CLASSES


# settings
# model_dir = args.model_dir
# if not os.path.exists(model_dir):
#    os.makedirs(model_dir)
# torch.manual_seed(args.seed)
# device = torch.device("cuda" if use_cuda else "cpu")
# kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}

# setup data loader
# transform_train = transforms.Compose(
# [
# transforms.RandomCrop(32, padding=4),
# transforms.RandomHorizontalFlip(),
# transforms.ToTensor(),
# ]
# )
# transform_test = transforms.Compose(
# [
# transforms.ToTensor(),
# ]
# )
# trainset = getattr(datasets, args.data)(
# root=args.data_path, train=True, download=True, transform=transform_train
# )
# testset = getattr(datasets, args.data)(
# root=args.data_path, train=False, download=True, transform=transform_test
# )
# train_loader = torch.utils.data.DataLoader(
# trainset, batch_size=args.batch_size, shuffle=True, **kwargs
# )
# test_loader = torch.utils.data.DataLoader(
# testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
# )


def perturb_input(
    model,
    x_natural,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance="l_inf",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    batch_size = len(x_natural)
    if distance == "l_inf":
        x_adv = (
            x_natural.detach()
            + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        )
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                    reduction="sum",
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(
                    F.log_softmax(model(adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                    reduction="sum",
                )
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = (
            x_natural.detach()
            + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        )
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def train(model, train_loader, optimizer, epoch, awp_adversary, data):
    args, epsilon, step_size, NUM_CLASSES = setup(data)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    print(f"epoch: {epoch}")
    # bar = Bar('Processing', max=len(train_loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch_idx, (data, target) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)

        # craft adversarial examples
        x_adv = perturb_input(
            model=model,
            x_natural=x_natural,
            step_size=step_size,
            epsilon=epsilon,
            perturb_steps=args.num_steps,
            distance=args.norm,
        )

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(
                inputs_adv=x_adv,
                inputs_clean=x_natural,
                targets=target,
                beta=args.beta,
            )
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(model(x_natural), dim=1),
            reduction="batchmean",
        )
        # calculate natural loss and backprop
        logits = model(x_natural)
        loss_natural = F.cross_entropy(logits, target)
        loss = loss_natural + args.beta * loss_robust

        prec1, prec5 = accuracy(logits_adv, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        """bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()"""
    # bar.finish()
    return losses.avg, top1.avg


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(test_loader))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            """bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()"""
    # bar.finish()
    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def main(args):
    kwargs = (
        {"num_workers": 2, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )

    # setup data loader
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    trainset = getattr(datasets, args.data)(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = getattr(datasets, args.data)(
        root=args.data_path,
        train=False,
        download=True,
        transform=transform_test,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )
    data = args.data
    args, epsilon, step_size, NUM_CLASSES = setup(data)
    # init model, ResNet18() can be also used here for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model= ResNet18(num_classes=NUM_CLASSES).to(device)
    model = WideResNet(
        depth=34, num_classes=100 if data == "cifar100" else 10
    ).to(device)
    # model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # We use a proxy model to calculate AWP, which does not affect the statistics of BN.
    # proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    # proxy = ResNet18(num_classes=NUM_CLASSES)
    proxy = WideResNet(
        depth=34, num_classes=100 if data == "cifar100" else 10
    ).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(
        model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma
    )

    criterion = nn.CrossEntropyLoss()

    logger = Logger(os.path.join(args.model_dir, "log.txt"), title=args.arch)
    logger.set_names(
        [
            "Learning Rate",
            "Adv Train Loss",
            "Nat Train Loss",
            "Nat Val Loss",
            "Adv Train Acc.",
            "Nat Train Acc.",
            "Nat Val Acc.",
        ]
    )

    if args.resume_model:
        model.load_state_dict(
            torch.load(args.resume_model, map_location=device)
        )
    if args.resume_optim:
        optimizer.load_state_dict(
            torch.load(args.resume_optim, map_location=device)
        )

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch, args)

        # adversarial training
        adv_loss, adv_acc = train(
            model, train_loader, optimizer, epoch, awp_adversary, data
        )

        # evaluation on natural examples
        print(
            "================================================================"
        )
        train_loss, train_acc = test(model, train_loader, criterion)
        val_loss, val_acc = test(model, test_loader, criterion)
        print(
            "================================================================"
        )

        logger.append(
            [lr, adv_loss, train_loss, val_loss, adv_acc, train_acc, val_acc]
        )

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.model_dir, f"ours-model-epoch{epoch}.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    args.model_dir, f"ours-opt-checkpoint_epoch{epoch}.tar"
                ),
            )
    return model


def main_trades_awp_10(args):
    model = main(args)

    md = "./data/model"
    torch.save(
        model.state_dict(),
        os.path.join(
            md, "model-trades-awp-{}-{}.pt".format("cifar10", "wideres34")
        ),
    )

    # target_model = ResNet18(num_classes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_model = WideResNet(depth=34, num_classes=10).to(device)
    from collections import OrderedDict

    checkpoint = torch.load(
        os.path.join(
            md, "model-trades-awp-{}-{}.pt".format("cifar10", "wideres34")
        )
    )
    try:
        target_model.load_state_dict(checkpoint)
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        target_model.load_state_dict(new_state_dict, False)

    target_model = torch.nn.DataParallel(target_model).cuda()

    torch.save(
        target_model.module.state_dict(),
        os.path.join(
            args.model_dir,
            "model-trades-awp-{}-{}.pt".format("cifar10", "wideres34"),
        ),
    )


def main_trades_awp_100(args):
    model = main(args)

    md = "./data/model"
    torch.save(
        model.state_dict(),
        os.path.join(
            md, "model-trades-awp-{}-{}.pt".format("cifar100", "wideres34")
        ),
    )

    # target_model = ResNet18(num_classes=100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_model = WideResNet(depth=34, num_classes=100).to(device)

    from collections import OrderedDict

    checkpoint = torch.load(
        os.path.join(
            md, "model-trades-awp-{}-{}.pt".format("cifar100", "wideres34")
        )
    )
    try:
        target_model.load_state_dict(checkpoint)
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        target_model.load_state_dict(new_state_dict, False)

    target_model = torch.nn.DataParallel(target_model).cuda()

    torch.save(
        target_model.module.state_dict(),
        os.path.join(
            args.model_dir,
            "model-trades-awp-{}-{}.pt".format("cifar100", "wideres34"),
        ),
    )


def main_tradesawp(args):
    if args.data == "cifar10":
        main_trades_awp_10(args)
    elif args.data == "cifar100":
        main_trades_awp_100(args)
    else:
        raise NotImplementedError
