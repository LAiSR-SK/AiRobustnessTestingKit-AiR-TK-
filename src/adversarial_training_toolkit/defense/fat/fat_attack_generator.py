# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class FAT_ATTACK_GENERATOR:
    
    def __init__(self,model,device):
        self.model = model.to(device)
        self.device = device

    def cwloss(self, output, target, num_classes,confidence=50):
        """
        This function computes the Carlini-Wagner loss between the output of a model and the target labels.
        The CW loss is used for generating adversarial examples.
        The implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT

        :param output: The output tensor from the model. It should have shape (batch_size, num_classes).
        :param target: The true labels for the data. It should be a 1D tensor of shape (batch_size,).
        :param num_classes: The number of classes in the classification task.
        :param confidence: The confidence margin, which is added to increase the confidence of the adversarial example 
        :param device: to set all the tensors to the same device.

        :return: The computed CW loss.
        """
        device = self.device
        target = target.to(device)
        output = output.to(device)
        # Convert the target to its one-hot representation
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.to(device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        # Create a variable for the target
        target_var = Variable(target_onehot, requires_grad=False)

        # Compute the real class probabilities
        real = (target_var * output).sum(1)

        # Compute the maximum other class probabilities
        other = ((1.0 - target_var) * output - target_var * 10000.0).max(1)[0]

        # Compute the CW loss
        loss = -torch.clamp(real - other + confidence, min=0.0)  # equiv to max(..., 0.)
        loss = torch.sum(loss)

        return loss

    def pgd(
        self,
        data,
        target,
        epsilon,
        step_size,
        num_steps,
        num_classes,
        loss_fn,
        category,
        rand_init,
    ):
        """
        This function implements the Projected Gradient Descent (PGD) attack.

        :param model: The model to attack.
        :param data: The input data.
        :param target: The target labels.
        :param epsilon: The maximum perturbation for each pixel.
        :param step_size: The step size for each update.
        :param num_steps: The number of steps for the attack.
        :param num_classes: The number of classes in the classification task.
        :param loss_fn: The loss function to use ('cent' for CrossEntropyLoss, 'cw' for CW loss).
        :param category: The category of the attack ('trades' or 'Madry').
        :param rand_init: Whether to initialize the perturbation with random noise.
        :param device: The device to perform computations on.

        :return: The adversarial examples.
        """
        # Move the model and data to the specified device
        device = self.device
        model = self.model
        data = data.to(device)
        target = target.to(device)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the adversarial examples
        if category == "trades":
            x_adv = (
                data.detach() + 0.001 * torch.randn(data.shape).to(device).detach()
                if rand_init
                else data.detach()
            )
        elif category == "Madry":
            x_adv = (
                data.detach()
                + torch.from_numpy(
                    np.random.uniform(-epsilon, epsilon, data.shape)
                )
                .float()
                .to(device)
                if rand_init
                else data.detach()
            )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Perform the attack
        for _ in range(num_steps):
            x_adv.requires_grad_()
            output = model(x_adv)
            model.zero_grad()
            with torch.enable_grad():
                if loss_fn == "cent":
                    loss_adv = nn.CrossEntropyLoss(reduction="mean")(
                        output, target
                    )
                if loss_fn == "cw":
                    loss_adv = self.cwloss(output, target, num_classes)
            loss_adv.backward()
            eta = step_size * x_adv.grad.sign() #(Pylance) not being able to infer the type of x_adv
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    def eval_clean(self, test_loader):
        """
        This function evaluates a model on clean (non-adversarial) data.

        :param model: The model to evaluate.
        :param test_loader: The DataLoader for the test data.
        :param device: The device to perform computations on.
        
        :return: The average test loss and test accuracy.
        """
        device = self.device
        # Move the model to the specified device
        model = self.model
        model = model.to(device)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the test loss and the number of correct predictions
        test_loss = 0
        correct = 0

        # Disable gradient computations
        with torch.no_grad():
            # Iterate over the test data
            for data, target in test_loader:
                # Move the data and target to the specified device
                data, target = data.to(device), target.to(device)

                # Get the output of the model
                output = model(data)

                # Compute the CrossEntropyLoss and add it to the total test loss
                test_loss += nn.CrossEntropyLoss(reduction="mean")(
                    output, target
                ).item()

                # Get the predicted class for each sample
                pred = output.max(1, keepdim=True)[1]

                # Count the number of correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute the average test loss
        test_loss /= len(test_loader.dataset)

        # Compute the test accuracy
        test_accuracy = correct / len(test_loader.dataset)

        return test_loss, test_accuracy


    def eval_robust(
        self,
        test_loader,
        num_classes,
        perturb_steps,
        epsilon,
        step_size,
        loss_fn,
        category,
        rand_init,
    ):
        """
        This function evaluates the robustness of a model against adversarial attacks.

        :param model: The model to evaluate.
        :param test_loader: The DataLoader for the test data.
        :param num_classes: The number of classes in the classification task.
        :param perturb_steps: The number of steps for the attack.
        :param epsilon: The maximum perturbation for each pixel.
        :param step_size: The step size for each update.
        :param loss_fn: The loss function to use ('cent' for CrossEntropyLoss, 'cw' for CW loss).
        :param category: The category of the attack ('trades' or 'Madry').
        :param rand_init: Whether to initialize the perturbation with random noise.
        :param device: The device to perform computations on.
        :return: The average test loss and test accuracy.
        """
        device = self.device
        # Move the model to the specified device
        model = self.model

        # Set the model to evaluation mode
        model.eval()

        # Initialize the test loss and the number of correct predictions
        test_loss = 0
        correct = 0

        # Enable gradient computations
        with torch.enable_grad():
            # Iterate over the test data
            for data, target in test_loader:
                # Move the data and target to the specified device
                data, target = data.to(device), target.to(device)

                # Generate adversarial examples using the PGD attack
                x_adv = self.pgd(
                    data,
                    target,
                    epsilon,
                    step_size,
                    perturb_steps,
                    num_classes,
                    loss_fn,
                    category,
                    rand_init=rand_init,
                )

                # Get the output of the model on the adversarial examples
                output = model(x_adv)

                # Compute the CrossEntropyLoss and add it to the total test loss
                test_loss += nn.CrossEntropyLoss(reduction="mean")(
                    output, target
                ).item()

                # Get the predicted class for each sample
                pred = output.max(1, keepdim=True)[1]

                # Count the number of correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute the average test loss
        test_loss /= len(test_loader.dataset)

        # Compute the test accuracy
        test_accuracy = correct / len(test_loader.dataset)

        return test_loss, test_accuracy