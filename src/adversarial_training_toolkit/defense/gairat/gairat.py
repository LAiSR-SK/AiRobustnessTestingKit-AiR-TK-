# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import adversarial_training_toolkit.defense.gairat.gairat_attack_generator as attack
from adversarial_training_toolkit.model import ResNet18, WideResNet


class GairatTraining:
    def __init__(
        self,
        dataset: str = "cifar10",
        net: str = "resnet18",
        epochs: int = 120,
        weight_decay: float = 2e-4,
        momentum: float = 0.9,
        epsilon: float = 0.031,
        num_steps: int = 10,
        step_size=0.007,
        seed: int = 1,
        log_dir: str = "log/gairat_out.tx",
        lr_schedule: str = "piecewise",
        lr_max: float = 0.1,
        lr_one_drop: float = 0.1,
        lr_drop_epoch: int = 100,
        Lambda: int = -1,
        begin_epoch: int = 60,
    ) -> None:
        ArgPrototype = namedtuple(
            "ArgPrototype",
            [
                "epochs",
                "weight_decay",
                "momentum",
                "epsilon",
                "num_steps",
                "step_size",
                "seed",
                "net",
                "dataset",
                "random",
                "depth",
                "width_factor",
                "drop_rate",
                "resume",
                "out_dir",
                "lr_schedule",
                "lr_max",
                "lr_one_drop",
                "lr_drop_epoch",
                "Lambda",
                "Lambda_max",
                "Lambda_schedule",
                "Weight_assignment_function",
                "begin_epoch",
            ],
        )
        self._args = ArgPrototype(
            epochs,
            weight_decay,
            momentum,
            epsilon,
            num_steps,
            step_size,
            seed,
            net,
            dataset,
            True,
            34,
            10,
            0.0,
            None,
            log_dir,
            lr_schedule,
            lr_max,
            lr_one_drop,
            lr_drop_epoch,
            Lambda,
            float("inf"),
            "fixed",
            "tanh",
            begin_epoch,
        )

    def __call__(
        self,
    ) -> None:
        torch.manual_seed(self._args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        main_gairat(self._args)


# # Training settings
# seed = args.seed
# momentum = args.momentum
# weight_decay = args.weight_decay
# depth = args.depth
# width_factor = args.width_factor
# drop_rate = args.drop_rate
# resume = args.resume
# out_dir = args.out_dir


# Save checkpoint
def save_checkpoint(
    state, checkpoint, filename="checkpoint.pth.tar"
):  # TODO: use args
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


# From GAIR.py file
def GAIR(num_steps, Kappa, Lambda, func):
    # Weight assign
    if func == "Tanh":
        reweight = (
            (
                Lambda
                + (int(num_steps / 2) - Kappa) * 5 / (int(num_steps / 2))
            ).tanh()
            + 1
        ) / 2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Sigmoid":
        reweight = (
            Lambda + (int(num_steps / 2) - Kappa) * 5 / (int(num_steps / 2))
        ).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Discrete":
        reweight = ((num_steps + 1) - Kappa) / (num_steps + 1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()

    return normalized_reweight


# Get adversarially robust network
def train(args, epoch, model, train_loader, optimizer, Lambda, lr_schedule):
    lr = 0
    num_data = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        loss = 0
        data, target = data.cuda(), target.cuda()

        # Get adversarial data and geometry value
        x_adv, Kappa = attack.GA_PGD(
            model,
            data,
            target,
            args.epsilon,
            args.step_size,
            args.num_steps,
            loss_fn="cent",
            category="Madry",
            rand_init=True,
        )

        model.train()
        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()

        logit = model(x_adv)

        if (epoch + 1) >= args.begin_epoch:
            Kappa = Kappa.cuda()
            loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
            # Calculate weight assignment according to geometry value
            normalized_reweight = GAIR(
                args.num_steps, Kappa, Lambda, args.weight_assignment_function
            )
            loss = loss.mul(normalized_reweight).mean()
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)

        train_robust_loss += loss.item() * len(x_adv)

        loss.backward()
        optimizer.step()

        num_data += len(data)

    train_robust_loss = train_robust_loss / num_data

    return train_robust_loss, lr


# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch, args):
    Lam = float(args.Lambda)
    if args.epochs >= 110:
        # Train Wide-ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == "linear":
            if epoch >= 60:
                Lambda = args.Lambda_max - (epoch / args.epochs) * (
                    args.Lambda_max - Lam
                )
        elif args.Lambda_schedule == "piecewise":
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam - 1.0
            elif epoch >= 110:
                Lambda = Lam - 1.5
        elif args.Lambda_schedule == "fixed":
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == "linear":
            if epoch >= 30:
                Lambda = args.Lambda_max - (epoch / args.epochs) * (
                    args.Lambda_max - Lam
                )
        elif args.Lambda_schedule == "piecewise":
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam - 2.0
        elif args.Lambda_schedule == "fixed":
            if epoch >= 30:
                Lambda = Lam
    return Lambda


def main_gairat(args):
    # Training settings
    seed = args.seed
    momentum = args.momentum
    weight_decay = args.weight_decay
    depth = args.depth
    width_factor = args.width_factor
    drop_rate = args.drop_rate
    resume = args.resume
    out_dir = args.out_dir

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Models and optimizer
    if args.net == "res18":
        model = ResNet18(
            num_classes=100 if args.dataset == "cifar100" else 10
        ).cuda()
    if args.net == "wideres":
        nc = 100 if args.dataset == "cifar100" else 10
        model = WideResNet(
            depth=depth,
            num_classes=nc,
            widen_factor=width_factor,
            dropRate=drop_rate,
        ).cuda()

    # model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_max,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Learning schedules
    if args.lr_schedule == "superconverge":
        lr_schedule = lambda t: np.interp(
            [t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0]
        )[0]
    elif args.lr_schedule == "piecewise":

        def lr_schedule(t):
            if args.epochs >= 110:
                # Train Wide-ResNet
                if t / args.epochs < 0.5:
                    return args.lr_max
                elif t / args.epochs < 0.75:
                    return args.lr_max / 10.0
                elif t / args.epochs < (11 / 12):
                    return args.lr_max / 100.0
                else:
                    return args.lr_max / 200.0
            else:
                # Train ResNet
                if t / args.epochs < 0.3:
                    return args.lr_max
                elif t / args.epochs < 0.6:
                    return args.lr_max / 10.0
                else:
                    return args.lr_max / 100.0
    elif args.lr_schedule == "linear":
        lr_schedule = lambda t: np.interp(
            [t],
            [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
            [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100],
        )[0]
    elif args.lr_schedule == "onedrop":

        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == "multipledecay":

        def lr_schedule(t):
            return args.lr_max - (t // (args.epochs // 10)) * (
                args.lr_max / 10
            )
    elif args.lr_schedule == "cosine":

        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

    # Store path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Setup data loader
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

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10",
            train=True,
            download=True,
            transform=transform_train,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10",
            train=False,
            download=True,
            transform=transform_test,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2
        )
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="./data/cifar-100",
            train=True,
            download=True,
            transform=transform_train,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data/cifar-100",
            train=False,
            download=True,
            transform=transform_test,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2
        )
    if args.dataset == "svhn":
        trainset = torchvision.datasets.SVHN(
            root="./data/SVHN",
            split="train",
            download=True,
            transform=transform_train,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.SVHN(
            root="./data/SVHN",
            split="test",
            download=True,
            transform=transform_test,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2
        )
    if args.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(
            root="./data/MNIST",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=128,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        testset = torchvision.datasets.MNIST(
            root="./data/MNIST",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=128,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    # Resume
    best_acc = 0
    start_epoch = 0
    if resume:
        # Resume directly point to checkpoint.pth.tar
        print("==> GAIRAT Resuming from checkpoint ..")
        print(resume)
        assert os.path.isfile(resume)
        out_dir = os.path.dirname(resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["test_pgd20_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
    else:
        print("==> GAIRAT")
        # logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
        # logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])

    ## Training get started
    test_nat_acc = 0
    test_pgd20_acc = 0

    for epoch in range(start_epoch, args.epochs):
        # Get lambda
        Lambda = adjust_Lambda(epoch + 1, args)

        # Adversarial training
        train_robust_loss, lr = train(
            args, epoch, model, train_loader, optimizer, Lambda, lr_schedule
        )

        # Evalutions similar to DAT.
        _, test_nat_acc = attack.eval_clean(model, test_loader)
        _, test_pgd20_acc = attack.eval_robust(
            model,
            test_loader,
            perturb_steps=20,
            epsilon=0.031,
            step_size=0.031 / 4,
            loss_fn="cent",
            category="Madry",
            random=True,
        )

        print(
            "Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n"
            % (epoch, args.epochs, lr, test_nat_acc, test_pgd20_acc)
        )

        # logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])

        # Save the best checkpoint
        if test_pgd20_acc > best_acc:
            best_acc = test_pgd20_acc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "test_nat_acc": test_nat_acc,
                    "test_pgd20_acc": test_pgd20_acc,
                    "optimizer": optimizer.state_dict(),
                },
                filename="bestpoint.pth.tar",
                checkpoint="data"
            )

        # Save the last checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "test_nat_acc": test_nat_acc,
                "test_pgd20_acc": test_pgd20_acc,
                "optimizer": optimizer.state_dict(),
            },
            checkpoint="data"
        )

    # logger_test.close()

    # Save the final model
    model_dir = "../data/model"
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, f"model-gairat-{args.dataset}-{args.net}.pt"),
    )
