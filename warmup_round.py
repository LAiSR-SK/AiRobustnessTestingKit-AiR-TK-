from __future__ import print_function
import time

from losses import *
from helper_functions import *
from models.resnet import *
from models.wideresnet import *
from helper_functions import args

# Define settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def train(args, model, device, train_loader, optimizer, epoch, ds_name):
    """
        Trains one epoch by calculating the loss for each batch and updating the model

        :param args: set of arguments including learning rate, log interval, etc
        :param model: model to train
        :param device: current device
        :param train_loader: data loader containing the training dataset
        :param optimizer: optimizer to train
        :param epoch: current epoch of training
    """

    model.train() # set the  model to training mode

    for batch_idx, sample in enumerate(train_loader):
        # Extract the image and classification details
        if ds_name == "cifar100":
            data, target, coarse = sample
        else:
            data, target = sample

        data, target = data.to(device), target.to(device) # set data/target to device
        optimizer.zero_grad() # zero out the gradients

        # Calculate the clean loss
        loss, batch_metrics = standard_loss(model, data, target, optimizer)

        # Update the model based on the loss
        loss.backward()
        optimizer.step()

        # Print training progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main_warmup(ds_name, mod_name, epochs):
    """
        Main training method, which establishes the model/dataset before conducting training and
        clean testing for each epoch. Then reports final training time and conducts robustness tests
        on the final model.

        :param ds_name: dataset to use for training
        :param mod_name: model to use for training
        :param epochs: number of epochs to train for
    """

    # Create file to print training progress
    filename = 'clean-{}-{}-output.txt'.format(ds_name, mod_name)
    f = open(filename, "a")

    # Initialize the model based on the specified architecture
    start_tot = time.time() # start recording training time
    model = get_model(mod_name, ds_name, device)

    # Set up the optimizer with the arguments
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Set up the dataloaders
    train_loader, test_loader = load_data(ds_name, args, kwargs)

    # Begin model training for the specified number of epochs
    for epoch in range(1, epochs + 1):
        # Adjust the SGD learning rate based on the epoch
        adjust_learning_rate(optimizer, epoch, args)

        start = time.time() # start recording training time
        train(args, model, device, train_loader, optimizer, epoch, ds_name)
        end = time.time() # stop recording training time
        epoch_time = end - start

        # Evaluation on natural and adversarial examples
        print('================================================================')
        f.write("Epoch " + str(epoch) + "\n")
        print("Time for Training: " + str(epoch_time))
        f.write("Time for Training: " + str(epoch_time) + "\n")
        eval_clean(model, device, train_loader, 'train', ds_name, f)
        eval_clean(model, device, test_loader, 'test', ds_name, f)
        robust_eval(model, device, test_loader, ds_name, f)
        print('================================================================')

        # Save the model (if designated)
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-warmup-{}-{}-epoch{}.pt'.format(ds_name, mod_name, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-warmup-{}-{}-epoch{}.tar'.format(ds_name, mod_name, epoch)))

    # Report full training time
    end_tot = time.time()
    total_time = end_tot - start_tot
    print("Total training time: " + str(total_time))
    f.write("Total training time: " + str(total_time) + "\n")

    f.close()  # close output file


if __name__ == '__main__':
    main_warmup('cifar100', 'wideres34', 25)