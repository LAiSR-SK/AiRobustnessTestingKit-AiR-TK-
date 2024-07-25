# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import datetime
import os
import ssl

import fat_attack_generator as attack
import numpy as np
import torch
import torchvision
from fat_earlystop import FAT_EARLYSTOP
from torch import nn, optim
from torchvision import transforms

from adversarial_training_toolkit.model import ResNet18
from adversarial_training_toolkit.model import WideResNet

ssl._create_default_https_context = ssl._create_unverified_context



class FAT_TRAINING:
    def __init__(self,epochs = 120,
                  weight_decay = 2e-4,
                    lr = 0.1,
                    momentum = 0.9,
                    epsilon = 0.31,
                    num_seps = 10,
                    step_size = 0.007,
                    seed = 7,
                    net = "wideresnet",
                    tau = 0,
                    dataset = "cifar10",
                    rand_init = True,
                    omega = 0.001,
                    dynamictau = True,
                    depth = 34,
                    width_factor = 10,
                    drop_rate = 0.0,
                    out_dir = "./FAT_results_10",
                    resume = "",
                    device = "cuda"
                  ):
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.lr = lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.num_steps = num_seps
        self.step_size = step_size
        self.seed = seed
        self.net = net
        self.tau = tau
        self.dataset = dataset
        self.rand_init = rand_init
        self.omega = omega
        self.dynamictau = dynamictau
        self.depth = depth
        self.width_factor = width_factor
        self.drop_rate = drop_rate
        self.out_dir = out_dir
        self.resume = resume
        self.device = device
        # training settings
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


    def train(self, model, train_loader, optimizer, tau):
        """ This function trains a model using adversarial training.
        
        :param model: The model to be trained.
        :param train_loader: The DataLoader that provides the training data.
        :param optimizer: The optimizer used for training the model.
        :param tau: The tau parameter used in the earlystop function.
        :return time: The time taken for training.
        :return loss_sum: The total loss during training.

        :return bp_count_avg: The average backpropagation count.
        """

        starttime = datetime.datetime.now()
        loss_sum = 0
        bp_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.cuda(self.device)

            # Get friendly adversarial training data via early-stopped PGD
            output_adv, output_target, output_natural, count = FAT_EARLYSTOP.earlystop(
                model,
                data,
                target,
                step_size=self.step_size,
                epsilon=self.epsilon,
                perturb_steps=self.num_steps,
                tau=self.tau,
                randominit_type="uniform_randominit",
                loss_fn="cent",
                rand_init=self.rand_init,
                omega=self.omega,
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


    def adjust_tau(self, epoch, dynamictau):
        """This function adjusts the value of tau based on the current epoch and whether dynamic tau adjustment is enabled.

        :param epoch: The current training epoch.
        :param dynamictau: A boolean indicating whether dynamic tau adjustment is enabled.

        :return tau: The adjusted tau value.
        """
        tau = self.tau
        if dynamictau:
            if epoch <= 50:
                tau = 0
            elif epoch <= 90:
                tau = 1
            else:
                tau = 2
        return tau


    def adjust_learning_rate(self, optimizer, epoch):
        """
        This function adjusts the learning rate based on the current epoch.

        :param optimizer: The optimizer whose learning rate will be adjusted.
        :param epoch: The current training epoch.
        """
        # Set the initial learning rate
        lr = self.lr

        # If the epoch is greater than or equal to 60, decrease the learning rate by a factor of 10
        if epoch >= 60:
            lr = self.lr * 0.1

        # If the epoch is greater than or equal to 90, decrease the learning rate by a factor of 100
        if epoch >= 90:
            lr = self.lr * 0.01

        # If the epoch is greater than or equal to 110, decrease the learning rate by a factor of 200
        if epoch >= 110:
            lr = self.lr * 0.005

        # Apply the new learning rate to all parameter groups in the optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    def save_checkpoint(self, state, checkpoint="", filename="checkpoint.pth.tar"):
        """This function saves the current state of the model as a checkpoint.

        :param state: The current state of the model.
        :param checkpoint: The directory where the checkpoint will be saved. If not provided, the default output directory is used.
        :param filename: The name of the checkpoint file. If not provided, the default is "checkpoint.pth.tar".
        """
        # If no checkpoint directory is provided, use the default output directory
        if checkpoint == "":
            checkpoint = self.out_dir

        # Create the full file path for the checkpoint
        filepath = os.path.join(checkpoint, filename)

        # Save the model state at the filepath
        torch.save(state, filepath)


    def main_fat(self, ds_name = ""):
        if __name__ == "__main__":
            ds_name ="cifar10"
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

        if self.net == "resnet18":
            model = ResNet18().to(self.device)
        if self.net == "wideresnet":
            # e.g., WRN-34-10
            model = WideResNet(
                depth=self.depth,
                num_classes=100 if self.dataset == "cifar100" else 10,
                widen_factor=self.width_factor,
                dropRate=self.drop_rate,
            ).to(self.device)

        model = torch.nn.DataParallel(model)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        start_epoch = 0
        # Resume
        if self.resume:
            # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
            # print('==> Friendly Adversarial Training Resuming from checkpoint ..')
            # print(args.resume)
            assert os.path.isfile(self.resume)
            checkpoint = torch.load(self.resume)
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
        for epoch in range(start_epoch, self.epochs):
            self.adjust_learning_rate(optimizer, epoch + 1)
            train_time, train_loss, bp_count_avg = self.train(
                model,
                train_loader,
                optimizer,
                self.adjust_tau(epoch + 1, self.dynamictau),
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
                    self.epochs,
                    train_time,
                    bp_count_avg,
                    test_nat_acc,
                    fgsm_acc,
                    test_pgd20_acc,
                    cw_acc,
                )
            )

            # logger_test.append([epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc])

            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "bp_avg": bp_count_avg,
                    "test_nat_acc": test_nat_acc,
                    "test_pgd20_acc": test_pgd20_acc,
                    "optimizer": optimizer.state_dict(),
                }
            )

