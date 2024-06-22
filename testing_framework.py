# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import argparse
import copy

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.swa_utils import AveragedModel

from autoattack import AutoAttack

from lib.attack import create_attack
from lib.model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, WideResNet
from helper_functions import cw_whitebox_eval, fgsm_whitebox_eval, mim_whitebox_eval, load_data

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
parser = argparse.ArgumentParser(description='Framework for Adversarial Testing')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_eval(model, dataset, device, test_loader, filename, model_type):
    """
        Main function to run evaluation on a given model.

        :param model: trained image classifier to be tested
        :param dataset: dataset the model is trained on
        :param device: current device
        :param test_loader: data loader containing test set for evaluation
        :param filename: name of file to print results to
        :param model_type: base model of the trained classifier
    """

    # Open an output file and put the model in eval mode
    f = open(filename, "a")
    model.eval()

    # Set the total errors to 0 for all atts but AA
    clean_total = 0
    cw_robust_total = 0
    mim_robust_total = 0
    fgsm_robust_total = 0

    linfpgd7_robust_total = 0
    linfpgd20_robust_total = 0
    linfpgd40_robust_total = 0
    l2pgd7_robust_total = 0
    l2pgd20_robust_total = 0
    l2pgd40_robust_total = 0

    aa_robust_total = 0

    # Set the criterion to cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Run tests for each element in the test set
    for batch_idx, (data, target) in enumerate(test_loader):
        # Set up the data/X and target/y correctly for evaluation
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, requires_grad=True), Variable(target)

        # Calculate the natural and robust error for each attack
        cw_err_natural, cw_err_robust = cw_whitebox_eval(model, dataset, data, target, device)
        mim_err_natural, mim_err_robust = mim_whitebox_eval(model, data, target, device)
        fgsm_err_natural, fgsm_err_robust = fgsm_whitebox_eval(model, data, target)

        # Calculate the PGD examples and corresponding robust errors
        #l2pgd7
        attack_l2pgd7 = create_attack(model, criterion, 'l2-pgd', 0.03, 7, 0.01)
        x_l2pgd7, _ = attack_l2pgd7.perturb(data, target)
        err_l2pgd7 = (model(x_l2pgd7).data.max(1)[1] != target.data).float().sum()

        # l2pgd20
        attack_l2pgd20 = create_attack(model, criterion, 'l2-pgd', 0.03, 20, 0.01)
        x_l2pgd20, _ = attack_l2pgd20.perturb(data, target)
        err_l2pgd20 = (model(x_l2pgd20).data.max(1)[1] != target.data).float().sum()

        # l2pgd40
        attack_l2pgd40 = create_attack(model, criterion, 'l2-pgd', 0.03, 40, 0.01)
        x_l2pgd40, _ = attack_l2pgd40.perturb(data, target)
        err_l2pgd40 = (model(x_l2pgd40).data.max(1)[1] != target.data).float().sum()

        # linfpgd7
        attack_linfpgd7 = create_attack(model, criterion, 'linf-pgd', 0.03, 7, 0.01)
        x_linfpgd7, _ = attack_linfpgd7.perturb(data, target)
        err_linfpgd7 = (model(x_linfpgd7).data.max(1)[1] != target.data).float().sum()

        # linfpgd20
        attack_linfpgd20 = create_attack(model, criterion, 'linf-pgd', 0.03, 20, 0.01)
        x_linfpgd20, _ = attack_linfpgd20.perturb(data, target)
        err_linfpgd20 = (model(x_linfpgd20).data.max(1)[1] != target.data).float().sum()

        # linfpgd40
        attack_linfpgd40 = create_attack(model, criterion, 'linf-pgd', 0.03, 40, 0.01)
        x_linfpgd40, _ = attack_linfpgd40.perturb(data, target)
        err_linfpgd40 = (model(x_linfpgd40).data.max(1)[1] != target.data).float().sum()

        # Calculate the AA examples and corresponding robust errors
        # aa
        adversary_aa = AutoAttack(model, norm='Linf', eps=.031, version='standard', verbose=False)
        x_aa = adversary_aa.run_standard_evaluation(data, target)
        err_aa = (model(x_aa).data.max(1)[1] != target.data).float().sum()

        # Add the losses to the total loss for each attack
        clean_total += cw_err_natural
        cw_robust_total += cw_err_robust
        mim_robust_total += mim_err_robust
        fgsm_robust_total += fgsm_err_robust

        # Add the pgd losses to the total losses
        linfpgd7_robust_total += err_linfpgd7
        linfpgd20_robust_total += err_linfpgd20
        linfpgd40_robust_total += err_linfpgd40
        l2pgd7_robust_total += err_l2pgd7
        l2pgd20_robust_total += err_l2pgd20
        l2pgd40_robust_total += err_l2pgd40

        # Add the AA losses to the total losses
        aa_robust_total += err_aa

    # Convert the clean loss tensor to accuracy %
    clean_total = int(clean_total)
    clean_acc = (10000 - clean_total) / 100

    # Convert the CW loss tensor to accuracy %
    cw_robust_total = int(cw_robust_total)
    cw_acc = (10000 - cw_robust_total) / 100

    # Convert the MIM loss tensor to accuracy %
    mim_robust_total = int(mim_robust_total)
    mim_acc = (10000 - mim_robust_total) / 100

    # Convert the FGSM loss tensor to accuracy %
    fgsm_robust_total = int(fgsm_robust_total)
    fgsm_acc = (10000 - fgsm_robust_total) / 100

    # Convert the linfpgd7 loss tensor to accuracy %
    linfpgd7_robust_total = int(linfpgd7_robust_total)
    linfpgd7_acc = (10000 - linfpgd7_robust_total) / 100

    # Convert the linfpgd20 loss tensor to accuracy %
    linfpgd20_robust_total = int(linfpgd20_robust_total)
    linfpgd20_acc = (10000 - linfpgd20_robust_total) / 100

    # Convert the linfpgd40 loss tensor to accuracy %
    linfpgd40_robust_total = int(linfpgd40_robust_total)
    linfpgd40_acc = (10000 - linfpgd40_robust_total) / 100

    # Convert the l2pgd7 loss tensor to accuracy %
    l2pgd7_robust_total = int(l2pgd7_robust_total)
    l2pgd7_acc = (10000 - l2pgd7_robust_total) / 100

    # Convert the l2pgd20 loss tensor to accuracy %
    l2pgd20_robust_total = int(l2pgd20_robust_total)
    l2pgd20_acc = (10000 - l2pgd20_robust_total) / 100

    # Convert the l2pgd40 loss tensor to accuracy %
    l2pgd40_robust_total = int(l2pgd40_robust_total)
    l2pgd40_acc = (10000 - l2pgd40_robust_total) / 100

    # Convert the aa loss tensor to accuracy %
    aa_robust_total = int(aa_robust_total)
    aa_acc = (10000 - aa_robust_total) / 100

    # Write the total losses to the file in % format
    f.write("Clean Accuracy: " + str(clean_acc) + "%\n")
    f.write("C&W Accuracy: " + str(cw_acc) + "%\n")
    f.write("MIM Accuracy: " + str(mim_acc) + "%\n")
    f.write("FGSM Accuracy: " + str(fgsm_acc) + "%\n")

    # Add the pgd losses to the total losses
    f.write("linf-pgd-7 Accuracy: " + str(linfpgd7_acc) + "%\n")
    f.write("linf-pgd-20 Accuracy: " + str(linfpgd20_acc) + "%\n")
    f.write("linf-pgd-40 Accuracy: " + str(linfpgd40_acc) + "%\n")
    f.write("l2-pgd-7 Accuracy: " + str(l2pgd7_acc) + "%\n")
    f.write("l2-pgd-20 Accuracy: " + str(l2pgd20_acc) + "%\n")
    f.write("l2-pgd-40 Accuracy: " + str(l2pgd40_acc) + "%\n")

    # Add the AA losses to the total losses
    f.write("AutoAttack Accuracy: " + str(aa_acc) + "%\n")

    f.close() # close the file

def main_testing(dataset, base_model, filename, model_string):
    """
        Main function that sets up the testing process.

        :param dataset: dataset the tested classifier was trained on
        :param base_model: model the tested classifier was built on
        :param filename: name of file to write results to
        :param model_string: location of model to be tested
    """

    # Set up the dataset and base model
    train_loader, test_loader = load_data(dataset, args, kwargs)
    if base_model == 'res18':
        model = ResNet18(num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'res34':
        model = ResNet34(num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'res50':
        model = ResNet50(num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'res101':
        model = ResNet101(num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'res152':
        model = ResNet152(num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'wideres28':
        model = WideResNet(depth=28, num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'wideres32':
        model = WideResNet(depth=32, num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'wideres34':
        model = WideResNet(depth=34, num_classes=100 if dataset == 'cifar100' else 10).to(device)
    elif base_model == 'wideres34swa':
        model_base = WideResNet(depth=34, num_classes=100 if dataset == 'cifar100' else 10).to(device)
        model = AveragedModel(model_base).to(device)
    else:
        raise NotImplementedError

    # Load in the desired model and make a deepcopy
    model.load_state_dict(torch.load(model_string))
    model_copy = copy.deepcopy(model)

    # Run the desired evaluation on the model copy and test set
    run_eval(model_copy, dataset, device, test_loader, filename, base_model)


if __name__ == "__main__":
    main_testing('cifar10', 'wideres34swa',
                 'tests/adt-va-cifar10-every9-epoch82-full-results.txt',
                 'saved-models/adt-va-every9/swa-model-adt-va-cifar10-wideres34-epoch82.pt')