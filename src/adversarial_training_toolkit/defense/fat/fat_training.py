# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import argparse
import datetime
import os
import ssl

import adversarial_training_toolkit.defense.fat.fat_attack_generator as attack
import numpy as np
import torch
import torchvision
from adversarial_training_toolkit.defense.fat.fat_earlystop import earlystop
from adversarial_training_toolkit.model import ResNet18, WideResNet
from torch import nn, optim
from torchvision import transforms

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(
    description="PyTorch Friendly Adversarial Training"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=120,
    metavar="N",
    help="number of epochs to train",
)
parser.add_argument(
    "--weight_decay", "--wd", default=2e-4, type=float, metavar="W"
)
parser.add_argument(
    "--lr", type=float, default=0.1, metavar="LR", help="learning rate"
)
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
)
parser.add_argument(
    "--epsilon", type=float, default=0.031, help="perturbation bound"
)
parser.add_argument(
    "--num_steps", type=int, default=10, help="maximum perturbation step K"
)
parser.add_argument("--step_size", type=float, default=0.007, help="step size")
parser.add_argument(
    "--seed", type=int, default=7, metavar="S", help="random seed"
)
parser.add_argument(
    "--net",
    type=str,
    default="wideresnet",
    help="decide which network to use,choose from smallcnn,resnet18,WRN",
)
parser.add_argument("--tau", type=int, default=0, help="step tau")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="choose from cifar10,cifar100",
)
parser.add_argument(
    "--rand_init",
    type=bool,
    default=True,
    help="whether to initialize adversarial sample with random noise",
)
parser.add_argument(
    "--omega",
    type=float,
    default=0.001,
    help="random sample parameter for adv data generation",
)
parser.add_argument(
    "--dynamictau", type=bool, default=True, help="whether to use dynamic tau"
)
parser.add_argument("--depth", type=int, default=34, help="WRN depth")
parser.add_argument(
    "--width_factor", type=int, default=10, help="WRN width factor"
)
parser.add_argument(
    "--drop_rate", type=float, default=0.0, help="WRN drop rate"
)
parser.add_argument(
    "--out_dir", type=str, default="./FAT_results_10", help="dir of output"
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="whether to resume training, default: None",
)

args = parser.parse_args()

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def train(model, train_loader, optimizer, tau):
    starttime = datetime.datetime.now()
    loss_sum = 0
    bp_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # Get friendly adversarial training data via early-stopped PGD
        output_adv, output_target, output_natural, count = earlystop(
            model,
            data,
            target,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            tau=tau,
            randominit_type="uniform_randominit",
            loss_fn="cent",
            rand_init=args.rand_init,
            omega=args.omega,
        )
        bp_count += count
        model.train()
        optimizer.zero_grad()
        output = model(output_adv)

        # calculate standard adversarial training loss
        loss = nn.CrossEntropyLoss(reduction="mean")(output, output_target)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    bp_count_avg = bp_count / len(train_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum, bp_count_avg


def adjust_tau(epoch, dynamictau):
    tau = args.tau
    if dynamictau:
        if epoch <= 50:
            tau = 0
        elif epoch <= 90:
            tau = 1
        else:
            tau = 2
    return tau


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(state, checkpoint=out_dir, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def main_fat(ds_name):
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

    # print('==> Load Test Data')
    if ds_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2
        )
    elif ds_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2
        )
    else:
        raise NotImplementedError

    if args.net == "resnet18":
        model = ResNet18().cuda()
    if args.net == "wideresnet":
        # e.g., WRN-34-10
        model = WideResNet(
            depth=args.depth,
            num_classes=100 if args.dataset == "cifar100" else 10,
            widen_factor=args.width_factor,
            dropRate=args.drop_rate,
        ).cuda()

    model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    # Resume
    if args.resume:
        # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
        # print('==> Friendly Adversarial Training Resuming from checkpoint ..')
        # print(args.resume)
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
    # else:
    # print('==> Friendly Adversarial Training')
    # logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title=title)
    # logger_test.set_names(['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc'])

    test_nat_acc = 0
    fgsm_acc = 0
    test_pgd20_acc = 0
    cw_acc = 0
    if ds_name == "cifar10":
        num_classes = 10
    elif ds_name == "cifar100":
        num_classes = 100
    else:
        raise NotImplementedError
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch + 1)
        train_time, train_loss, bp_count_avg = train(
            model,
            train_loader,
            optimizer,
            adjust_tau(epoch + 1, args.dynamictau),
        )

        ## Evalutions the same as DAT.
        loss, test_nat_acc = attack.eval_clean(model, test_loader)
        loss, fgsm_acc = attack.eval_robust(
            model,
            test_loader,
            num_classes,
            perturb_steps=1,
            epsilon=0.031,
            step_size=0.031,
            loss_fn="cent",
            category="Madry",
            rand_init=True,
        )
        loss, test_pgd20_acc = attack.eval_robust(
            model,
            test_loader,
            num_classes,
            perturb_steps=20,
            epsilon=0.031,
            step_size=0.031 / 4,
            loss_fn="cent",
            category="Madry",
            rand_init=True,
        )
        loss, cw_acc = attack.eval_robust(
            model,
            test_loader,
            num_classes,
            perturb_steps=30,
            epsilon=0.031,
            step_size=0.031 / 4,
            loss_fn="cw",
            category="Madry",
            rand_init=True,
        )

        print(
            "Epoch: [%d | %d] | Train Time: %.2f s | BP Average: %.2f | Natural Test Acc %.2f | FGSM Test Acc %.2f | PGD20 Test Acc %.2f | CW Test Acc %.2f |\n"
            % (
                epoch + 1,
                args.epochs,
                train_time,
                bp_count_avg,
                test_nat_acc,
                fgsm_acc,
                test_pgd20_acc,
                cw_acc,
            )
        )

        # logger_test.append([epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc])

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "bp_avg": bp_count_avg,
                "test_nat_acc": test_nat_acc,
                "test_pgd20_acc": test_pgd20_acc,
                "optimizer": optimizer.state_dict(),
            }
        )


if __name__ == "__main__":
    main_fat("cifar10")
