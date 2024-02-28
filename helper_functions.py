from __future__ import print_function
import argparse
import ssl

import torchvision
from torchvision import transforms

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from cifar_data import *
from models.resnet import *
from models.wideresnet import *

parser = argparse.ArgumentParser(description='PyTorch VA Adversarial Training')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--model-arch', default='wideres34',
                    help='model architecture to train')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=140, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--warmup', type=int, default=0, metavar='N',
                    help='number of epochs to train with clean data before AT')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./saved-models',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--lr-schedule', default='decay',
                    help='schedule for adjusting learning rate')
parser.add_argument('--epsilon', type=float, default=8.0 / 255.0,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2.0 / 255.0,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
args = parser.parse_args()


def eval_clean(model, device, data_loader, name, ds_name, f):
    """
        Calculates the natural error of the model on either the training
        or test set of data

        :param model: trained model to be evaluated
        :param device: the device the model is set on
        :param data_loader: data loader containing the dataset to test
        :param name: 'train' or 'test', denoting which dataset is being tested
        :param ds_name: name of dataset, cifar10 or cifar100
        :param f: file to print results to
    """

    model.eval()

    # Set the total errors to 0 for all atts but AA
    total_err = 0

    # Run tests for each element in the test set
    for sample in data_loader:
        # Set up the data/X and target/y correctly for evaluation
        if ds_name == 'cifar100':
            data, target, _ = sample
        else:
            data, target = sample
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        # Calculate the natural error for each attack
        err = clean(model, X, y)

        # Add the losses to the total loss for each attack
        total_err += err

    # Write the total losses to the file
    if name == 'test':
        # Convert the clean loss to clean accuracy %
        clean_total = int(total_err)
        clean_acc = (10000 - clean_total) / 100
        print("Clean Accuracy (Test): " + str(clean_acc) + "%")
        f.write("Clean Accuracy (Test): " + str(clean_acc) + "%\n")
    elif name == 'train':
        # Convert the clean loss to clean accuracy %
        clean_total = int(total_err)
        clean_acc = (50000 - clean_total) / 500
        print("Clean Accuracy (Train): " + str(clean_acc) + "%")
        f.write("Clean Accuracy (Train): " + str(clean_acc) + "%\n")
    else:
        raise NotImplementedError


def robust_eval(model, device, test_loader, ds_name, f):
    """
        Calculates the clean loss and robust  loss against PGD and FGSM attacks

        :param model: trained model to be evaluated
        :param device: the device the model is set on
        :param test_loader: data loader containing the testing dataset
        :param f: the file to write results to

        :return pgd_acc: the accuracy against PGD attack
    """

    model.eval()

    # Set the total errors to 0 for all atts but AA
    clean_total = 0
    pgd_robust_total = 0
    fgsm_robust_total = 0

    # Run tests for each element in the test set
    for sample in test_loader:
        # Set up the data/X and target/y correctly for evaluation
        if ds_name == 'cifar100':
            data, target, _ = sample
        else:
            data, target = sample
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        # Calculate the natural and robust error for each attack
        pgd_err_natural, pgd_err_robust = pgd_whitebox_eval(model, X, y, device)
        fgsm_err_natural, fgsm_err_robust = fgsm_whitebox_eval(model, X, y)

        # Add the losses to the total loss for each attack
        clean_total += pgd_err_natural
        pgd_robust_total += pgd_err_robust
        fgsm_robust_total += fgsm_err_robust

    # Convert the clean error to clean accuracy %
    clean_total = int(clean_total)
    clean_acc = (10000 - clean_total) / 100

    # Convert the PGD error to clean accuracy %
    pgd_robust_total = int(pgd_robust_total)
    pgd_acc = (10000 - pgd_robust_total) / 100

    # Convert the FGSM error to clean accuracy %
    fgsm_robust_total = int(fgsm_robust_total)
    fgsm_acc = (10000 - fgsm_robust_total) / 100

    # Print out the loss percents
    print("Clean Accuracy (Test): " + str(clean_acc) + "%")
    print("PGD Robust Accuracy: " + str(pgd_acc) + "%")
    print("FGSM Robust Accuracy: " + str(fgsm_acc) + "%")

    # Write loss percents to file
    f.write("Clean Accuracy (Test): " + str(clean_acc) + "%\n")
    f.write("PGD Robust Accuracy: " + str(pgd_acc) + "%\n")
    f.write("FGSM Robust Accuracy: " + str(fgsm_acc) + "%\n")

    return pgd_acc

def adjust_learning_rate(optimizer, epoch, args):
    """
        Sets the learning rate of the optimizer based on the current epoch

        :param optimizer: optimizer with learning rate being set
        :param epoch: current epoch
        :param args: program arguments
    """

    if args.lr_schedule == "decay":
        if epoch <= 74:
            lr = args.lr
        elif epoch <= 99:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif args.lr_schedule == "scheduled":
        if epoch >= 24:
            lr = args.lr * 0.1
        if epoch >= 26:
            lr = args.lr
        if epoch >= 64:
            lr = args.lr * 0.1
        if epoch >= 66:
            lr = args.lr
        if epoch >= 104:
            lr = args.lr * 0.1
        if epoch >= 106:
            lr = args.lr
        if epoch >= 137:
            lr = args.lr * 0.1
        if epoch >= 139:
            lr = args.lr * 0.01
    elif args.lr_schedule == "cosine":
        lr = 0.2
        lr = lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    elif args.lr_schedule == "cyclic-10":
        if epoch % 10 < 2 or epoch % 10 == 9:
            lr = args.lr
        elif epoch % 10 < 4 or epoch % 10 > 6:
            lr = args.lr * 0.1
        elif epoch % 10 == 5:
            lr = args.lr * 0.001
        else:
            lr = args.lr * 0.01
    elif args.lr_schedule == "cyclic-5":
        if epoch % 5 == 0:
            lr = args.lr
        elif epoch % 5 == 1 or epoch % 5 == 4:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_data(ds_name, args, kwargs, coarse=False):
    """
        Loads the specified dataset into a dataloader and returns the data split into
        training and test loaders

        :param ds_name: name of the dataset to load for training
        :param args: program arguments
        :param kwargs: more program arguments

        :return data_loaders containing the training and testing sets
    """

    # Establish the data loader transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    ssl._create_default_https_context = ssl._create_unverified_context # set the context for working with tensors

    if ds_name == 'cifar10':
        # Load in the CIFAR10 dataloaders
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif ds_name == 'cifar100':
        # Load in the CIFAR100 dataloaders
        if coarse==True:
            trainset = CIFAR100(root='../data', train=True, download=True, transform=transform_train, coarse=True)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            testset = CIFAR100(root='../data', train=False, download=True, transform=transform_test,  coarse=True)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        else:
            trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True,
                                                     transform=transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True,
                                                    transform=transform_test)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError

    return train_loader, test_loader

def epochs_define_attacks(epoch, dataset):
    """
        Defines a dictionary of attacks based on the current epoch.

        :param epoch: current epoch
        :param dataset: name of dataset

        :return attacks: dictionary of attacks
    """

    attacks = {}
    if dataset == 'cifar10':
        if epoch <= 5:
            for i in range(4):
                attacks[i] = 'l2-pgd-7'
            for i in range(4, 7):
                attacks[i] = 'l2-pgd-40'
            for i in range(7, 10):
                attacks[i] = 'l2-pgd-20'
        elif epoch <= 10:
            for i in range(4):
                attacks[i] = 'l2-pgd-20'
            for i in range(4, 7):
                attacks[i] = 'l2-pgd-7'
            for i in range(7, 10):
                attacks[i] = 'l2-pgd-40'
        elif epoch <= 15:
            for i in range(4):
                attacks[i] = 'l2-pgd-40'
            for i in range(4, 7):
                attacks[i] = 'l2-pgd-20'
            for i in range(7, 10):
                attacks[i] = 'l2-pgd-7'
        elif epoch <= 25:
            for i in range(10):
                attacks[i] = 'linf-pgd-3'
        elif epoch <= 30:
            for i in range(3):
                attacks[i] = 'linf-pgd-7'
            for i in range(3, 6):
                attacks[i] = 'linf-pgd-20'
            for i in range(6, 8):
                attacks[i] = 'linf-pgd-40'
            for i in range(8, 10):
                attacks[i] = 'mim'
        elif epoch <= 35:
            for i in range(3):
                attacks[i] = 'linf-pgd-20'
            for i in range(3, 6):
                attacks[i] = 'linf-pgd-7'
            for i in range(6, 8):
                attacks[i] = 'mim'
            for i in range(8, 10):
                attacks[i] = 'linf-pgd-40'
        elif epoch <= 40:
            for i in range(3):
                attacks[i] = 'linf-pgd-40'
            for i in range(3, 6):
                attacks[i] = 'mim'
            for i in range(6, 8):
                attacks[i] = 'linf-pgd-7'
            for i in range(8, 10):
                attacks[i] = 'linf-pgd-20'
        elif epoch <= 45:
            for i in range(3):
                attacks[i] = 'mim'
            for i in range(3, 6):
                attacks[i] = 'linf-pgd-40'
            for i in range(6, 8):
                attacks[i] = 'linf-pgd-20'
            for i in range(8, 10):
                attacks[i] = 'linf-pgd-7'
        elif epoch <= 53:
            for i in range(2):
                attacks[i] = 'apgd-t'
            for i in range(2, 5):
                attacks[i] = 'cw'
            for i in range(5, 8):
                attacks[i] = 'apgd-dlr'
            for i in range(8, 10):
                attacks[i] = 'apgd-ce'
        elif epoch <= 61:
            for i in range(2):
                attacks[i] = 'apgd-ce'
            for i in range(2, 5):
                attacks[i] = 'apgd-t'
            for i in range(5, 8):
                attacks[i] = 'cw'
            for i in range(8, 10):
                attacks[i] = 'apgd-dlr'
        elif epoch <= 68:
            for i in range(2):
                attacks[i] = 'apgd-dlr'
            for i in range(2, 5):
                attacks[i] = 'apgd-ce'
            for i in range(5, 8):
                attacks[i] = 'apgd-t'
            for i in range(8, 10):
                attacks[i] = 'cw'
        elif epoch <= 75:
            for i in range(2):
                attacks[i] = 'cw'
            for i in range(2, 5):
                attacks[i] = 'apgd-dlr'
            for i in range(5, 8):
                attacks[i] = 'apgd-ce'
            for i in range(8, 10):
                attacks[i] = 'apgd-t'
        else:
            for i in range(10):
                attacks[i] = 'autoattack'
    elif dataset == 'cifar100':
        if epoch <= 25:
            for i in range(20):
                attacks[i] = 'linf-pgd-3'
        elif epoch <= 45:
            for i in range(10):
                attacks[i] = 'linf-pgd-7'
            for i in range(10, 20):
                attacks[i] = 'mim'
        elif epoch <= 65:
            for i in range(10):
                attacks[i] = 'mim'
            for i in range(10, 20):
                attacks[i] = 'linf-pgd-7'
        elif epoch <= 85:
            for i in range(10):
                attacks[i] = 'linf-pgd-20'
            for i in range(10, 20):
                attacks[i] = 'linf-pgd-40'
        elif epoch <= 105:
            for i in range(10):
                attacks[i] = 'linf-pgd-40'
            for i in range(10, 20):
                attacks[i] = 'linf-pgd-20'
        elif epoch <= 140:
            for i in range(20):
                attacks[i] = 'cw'
        else:
            for i in range(20):
                attacks[i] = 'autoattack'
    else:
        raise NotImplementedError

    return attacks

def one_hot_tensor(y_batch_tensor, num_classes, device):
    """
        Converts a batch tensor into a one-hot tensor

        :param y_batch_tensor: batch tensor to be converted
        :param num_classes: number of classes to fill the tensor
        :param device: current device

        :return one-hot tensor based on the input batch tensor
    """

    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


class CWLoss(nn.Module):
    """Class for the CW loss, including parameters as well as a method to propogate
    the loss forwards"""

    def __init__(self, num_classes, margin=50, reduce=True):
        """
        Set up the CW loss with number of classes, margin of 50, and reduce = True
        """
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param logits: predictions
        :param targets: target labels
        :return: loss
        """
        # Convert the target labels to a one-hot tensor
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        # Calculate self-loss (sum of targets/predictions)
        self_loss = torch.sum(onehot_targets * logits, dim=1)

        # Calculate other-loss
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        # Take the loss as the reverse sum of the loss differences plus the margin (clamped)
        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        # If reducing (default True), divide the loss by the number of targets
        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


def clean(model, X, y):
    """
        Evaluates a model on a clean sample.

        :param model: classifier to be evaluated
        :param X: image
        :param y: correct classification of image

        :return the error between the model prediction of the image and the correct classification
    """

    out = model(X) # model prediction
    err = (out.data.max(1)[1] != y.data).float().sum() # error between prediction and classification
    return err


def pgd_whitebox_eval(model,
                  X,
                  y,
                  device,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    """
        Evaluates the model by perturbing an image using the PGD attack.

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the PGD attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps

        :return clean error and PGD error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If specified, create random noice between - and + epsilon and add to X_pgd
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # For each perturbation step
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        # Calculate the perturbation eta as the step size in the gradient direction of X_pgd
        eta = step_size * X_pgd.grad.data.sign()

        # Add the perturbation to X_pgd
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Set the perturbation to the difference between X and X_adv, clamped by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd again
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # Calculate the error between the PGD-perturbed prediction and correct classification
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()

    return err, err_pgd

def fgsm_whitebox_eval(model, X, y, epsilon=args.epsilon):
    """
        Evaluates the model by perturbing an image using the FGSM attack.

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param epsilon: epsilon size for the FGSM attack

        :return clean error and FGSM error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_fgsm basis by duplicating X as a variable
    X_fgsm = Variable(X.data, requires_grad=True)

    # Create the SGD optimizer and zero the gradients
    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    # With gradients, set up the cross entropy loss and step backward
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()

    # Discover X_fgsm by adding epsilon in the gradient direction and clamping the sample
    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)

    # Calculates the FGSM error between the prediction and correct classification
    err_fgsm = (model(X_fgsm).data.max(1)[1] != y.data).float().sum()

    return err, err_fgsm


def cw_whitebox_eval(model,
                dataset,
                 X,
                 y,
                 device,
                 epsilon=args.epsilon,
                 num_steps=args.num_steps,
                 step_size=args.step_size):
    """
        Evaluates the model by perturbing an image using the CW attack.

        :param model: model being attacked
        :param dataset: name of dataset (cifar10 or cifar100)
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the CW attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps

        :return clean error and CW error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_cw basis by duplicating X as a variable
    X_cw = Variable(X.data, requires_grad=True)

    # If specified, create random noice between - and + epsilon and add to X_cw
    if args.random:
        random_noise = torch.FloatTensor(*X_cw.shape).uniform_(-epsilon, epsilon).to(device)
        X_cw = Variable(X_cw.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_cw], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the CW loss and step backward
        with torch.enable_grad():
            loss = CWLoss(100 if dataset == 'cifar100' else 10)(model(X_cw), y)
        loss.backward()

        # Calculate the perturbation using the CW loss gradient
        eta = step_size * X_cw.grad.data.sign()
        X_cw = Variable(X_cw.data + eta, requires_grad=True)
        eta = torch.clamp(X_cw.data - X.data, -epsilon, epsilon)

        # Add the perturbation and clamp x_cw
        X_cw = Variable(X.data + eta, requires_grad=True)
        X_cw = Variable(torch.clamp(X_cw, 0, 1.0), requires_grad=True)

    # Calculates the CW error between the prediction and correct classification
    err_cw = (model(X_cw).data.max(1)[1] != y.data).float().sum()

    return err, err_cw


def mim_whitebox_eval(model,
                  X,
                  y,
                  device,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  decay_factor=1.0):

    """
        Evaluates the model by perturbing an image using the MIM attack.

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the MIM attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps
        :param decay_factor: coefficient for previous gradient usage in MIM attack

        :return clean error and MIM error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_mim basis by duplicating X as a variable
    X_mim = Variable(X.data, requires_grad=True)

    # If specified, create random noice between - and + epsilon and add to X_cw
    if args.random:
        random_noise = torch.FloatTensor(*X_mim.shape).uniform_(-epsilon, epsilon).to(device)
        X_mim = Variable(X_mim.data + random_noise, requires_grad=True)

    # Set up tensor to hold previous gradients
    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_mim], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_mim), y)
        loss.backward()

        # Calculate the perturbation using the current and previous gradient
        grad = X_mim.grad.data / torch.mean(torch.abs(X_mim.grad.data), [1, 2, 3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_mim = Variable(X_mim.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_mim.data - X.data, -epsilon, epsilon)

        # Add the perturbation and clamp x_mim
        X_mim = Variable(X.data + eta, requires_grad=True)
        X_mim = Variable(torch.clamp(X_mim, 0, 1.0), requires_grad=True)

    # Calculates the MIM error between the prediction and correct classification
    err_mim = (model(X_mim).data.max(1)[1] != y.data).float().sum()

    return err, err_mim


def cw_whitebox(model,
                 X,
                 y,
                 device,
                 dataset,
                 epsilon=args.epsilon,
                 num_steps=args.num_steps,
                 step_size=args.step_size):
    """
        Attacks the specified image X using the CW attack and returns the adversarial example

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the PGD attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps

        :return adversarial example found with the CW attack
    """

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If adding random, create random noice between - and + epsilon and add to X_pgd
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # For each perturbation step:
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the CW loss and step backward
        with torch.enable_grad():
            loss = CWLoss(100 if dataset == 'cifar100' else 10)(model(X_pgd), y)
        loss.backward()

        # Calculate the perturbation eta as the step size in the gradient direction of X_pgd
        eta = step_size * X_pgd.grad.data.sign()

        # Add the perturbation to X_pgd
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Set the perturbation to the difference between X and X_adv, clamped by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd again
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def mim_whitebox(model,
                  X,
                  y,
                  device,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  decay_factor=1.0):
    """
        Attacks the specified image X using the MIM attack and returns the adversarial example

        :param model: model being attacked
        :param X: image being attacked
        :param y: correct label of the image being attacked
        :param device: current device
        :param epsilon: epsilon size for the PGD attack
        :param num_steps: number of perturbation steps
        :param step_size: step size of perturbation steps
        :param decay_factor: factor of decay for gradients

        :return adversarial example found with the MIM attack
    """

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If adding random, create random noice between - and + epsilon and add to X_pgd
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # Set the previous gradient to a tensor of 0s in the shape of X
    previous_grad = torch.zeros_like(X.data)

    # For each perturbation step:
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross-entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        # Calculate the gradient by dividing it by the average
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1, 2, 3], keepdim=True)

        # Calculate the previous gradient by multiplying it by the decay factor and adding to the grad
        previous_grad = decay_factor * previous_grad + grad

        # Perturb X_pgd in the direction of the previous grad, by the step size
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)

        # Set the perturbation to the difference between X and X_adv, clamped by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd againt
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def get_model(mod_name, ds_name, device):
    if mod_name == 'res18':
        model = ResNet18(num_classes=100 if ds_name == 'cifar100' else 10).to(device)
    elif mod_name == 'res34':
        model = ResNet34(num_classes=100 if ds_name == 'cifar100' else 10).to(device)
    elif mod_name == 'res50':
        model = ResNet50(num_classes=100 if ds_name == 'cifar100' else 10).to(device)
    elif mod_name == 'res101':
        model = ResNet101(num_classes=100 if ds_name == 'cifar100' else 10).to(device)
    elif mod_name == 'res152':
        model = ResNet152(num_classes=100 if ds_name == 'cifar100' else 10).to(device)
    elif mod_name == 'wideres34':
        model = WideResNet(depth=34, num_classes=100 if ds_name == 'cifar100' else 10).to(device)
    else:
        raise NotImplementedError

    return model