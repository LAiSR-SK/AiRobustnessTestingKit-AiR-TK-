import os

import torch

# from tensorboardX import SummaryWriter
import torch.optim as optim
from adversarial_training_toolkit.defense.yopo.config import args, config
from adversarial_training_toolkit.defense.yopo.dataset import (
    create_test_dataset,
    create_train_dataset,
)
from adversarial_training_toolkit.defense.yopo.loss import Hamiltonian
from adversarial_training_toolkit.defense.yopo.training_function import (
    FastGradientLayerOneTrainer,
    train_one_epoch,
)
from adversarial_training_toolkit.defense.yopo.training_train import (
    eval_one_epoch,
)
from adversarial_training_toolkit.defense.yopo.utils_misc import (
    load_checkpoint,
    save_checkpoint,
)

# from network import create_network
from adversarial_training_toolkit.defense.yopo.wide_resnet import WideResNet

DEVICE = torch.device("cuda:{}".format(args.d))
torch.backends.cudnn.benchmark = True

# writer = SummaryWriter(log_dir=config.log_dir)


def main_yopo(ds_name):
    if ds_name == "cifar10":
        num_classes = 10
    elif ds_name == "cifar100":
        num_classes = 100
    else:
        raise NotImplementedError
    net = WideResNet(depth=34, num_classes=num_classes).to(
        DEVICE
    )  # TODO set model/dataset here
    net.to(DEVICE)
    criterion = config.create_loss_function().to(DEVICE)
    # criterion = CrossEntropyWithWeightPenlty(net.other_layers, DEVICE, config.weight_decay)#.to(DEVICE)
    # ce_criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = config.create_optimizer(net.parameters())
    lr_scheduler = config.create_lr_scheduler(optimizer)

    ## Make Layer One trainner  This part of code should be writen in config.py

    Hamiltonian_func = Hamiltonian(net.layer_one, config.weight_decay)
    layer_one_optimizer = optim.SGD(
        net.layer_one.parameters(),
        lr=lr_scheduler.get_lr()[0],
        momentum=0.9,
        weight_decay=5e-4,
    )
    lyaer_one_optimizer_lr_scheduler = optim.lr_scheduler.MultiStepLR(
        layer_one_optimizer, milestones=[30, 34, 36], gamma=0.1
    )
    LayerOneTrainer = FastGradientLayerOneTrainer(
        Hamiltonian_func,
        layer_one_optimizer,
        config.inner_iters,
        config.sigma,
        config.eps,
    )

    ds_train = create_train_dataset(ds_name, args.batch_size)
    ds_val = create_test_dataset(ds_name, args.batch_size)

    # TrainAttack = config.create_attack_method(DEVICE)
    EvalAttack = config.create_evaluation_attack_method(DEVICE)

    now_epoch = 0

    if args.auto_continue:
        args.resume = os.path.join(config.model_dir, "last.checkpoint")
    if args.resume is not None and os.path.isfile(args.resume):
        now_epoch = load_checkpoint(args.resume, net, optimizer, lr_scheduler)

    while True:
        if now_epoch > config.num_epochs:
            break
        now_epoch = now_epoch + 1

        descrip_str = "Training epoch:{}/{} -- lr:{}".format(
            now_epoch, config.num_epochs, lr_scheduler.get_lr()[0]
        )
        acc, yofoacc = train_one_epoch(
            net,
            ds_train,
            optimizer,
            criterion,
            LayerOneTrainer,
            config.K,
            DEVICE,
            descrip_str,
        )
        tb_train_dic = {"Acc": acc, "YofoAcc": yofoacc}
        print(tb_train_dic)
        # writer.add_scalars('Train', tb_train_dic, now_epoch)
        if config.val_interval > 0 and now_epoch % config.val_interval == 0:
            acc, advacc = eval_one_epoch(net, ds_val, DEVICE, EvalAttack)
            tb_val_dic = {"Acc": acc, "AdvAcc": advacc}
            # writer.add_scalars('Val', tb_val_dic, now_epoch)

        lr_scheduler.step()
        lyaer_one_optimizer_lr_scheduler.step()
        save_checkpoint(
            now_epoch,
            net,
            optimizer,
            lr_scheduler,
            file_name=os.path.join(
                config.model_dir, "epoch-{}.checkpoint".format(now_epoch)
            ),
        )


if __name__ == "__main__":
    main_yopo()
