# (c) 2023 Harry24k
# This code is licensed under the MIT license (see LICENSE.md).
"""
This file has been taken from https://github.com/Harry24k/MAIR/blob/main/mair/attacks/attack.py#L18
"""

import time
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset


def wrapper_method(func):
    # allows you to add extra functionality to an existing function or method
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get("_attacks").values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result

    return wrapper_func


class Attack:
    """Base class for MAIR's attacks.

    .. note::
        This class automatically sets the device to the device where the given model is.
        It changes the training mode to eval during the attack process.
        To change this, please see `set_model_training_mode`.
    """

    def __init__(self, name: str, model: torch.nn.Module):
        """Initializes internal attack state.

        :param name: Name of the attack.
        :param model: Model to attack.
        """

        self.attack: str = name
        self._attacks: OrderedDict = OrderedDict()
        self.set_model(model)

        try:
            self.device: torch.device = next(model.parameters()).device
        except Exception:
            self.device: torch.device = None
            print(
                "Failed to set device automatically, please try set_device() manual."
            )

        # Controls attack mode.
        self.attack_mode: str = "default"
        self.supported_mode: list[str] = ["default"]
        self.targeted: bool = False
        self._target_map_function: callable = None

        # Controls when normalization is used.
        self.normalization_used: torch.Tensor = None
        self._normalization_applied: bool = None
        if self.model.__class__.__name__ == "RobModel":
            self._set_rmodel_normalization_used(model)

        # Controls model mode during attack.
        self._model_training: bool = False
        self._batchnorm_training: bool = False
        self._dropout_training: bool = False

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    @wrapper_method
    def set_model(self, model: torch.nn.Module):
        """Sets the model for the attack."""

        self.model: torch.nn.Module = model
        self.model_name: str = model.__class__.__name__

    def get_logits(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """Gets the logits from the model."""

        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits: torch.Tensor = self.model(inputs)
        return logits

    @wrapper_method
    def _set_normalization_applied(self, flag: bool):
        """Sets the normalization applied flag."""

        self._normalization_applied: bool = flag

    @wrapper_method
    def set_device(self, device: torch.device):
        """Sets the device for the attack."""

        self.device: torch.device = device

    @wrapper_method
    def _set_rmodel_normalization_used(self, model: torch.nn.Module):
        """Sets attack normalization for MAIR [https://github.com/Harry24k/MAIR]."""

        mean = getattr(model, "mean", None)
        std = getattr(model, "std", None)
        if (mean is not None) and (std is not None):
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()
            if isinstance(std, torch.Tensor):
                std = std.cpu().numpy()
            if (mean != 0).all() or (std != 1).all():
                self.set_normalization_used(mean, std)

    @wrapper_method
    def set_normalization_used(self, mean, std):
        """This function sets the normalization parameters (mean and standard deviation) to be used for normalizing inputs.
        The parameters are stored in a dictionary and reshaped to match the input tensor's shape for broadcasting.


        :param mean: The mean values for each channel.
        :param std: The standard deviation values for each channel.
        """
        self.normalization_used = {}
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used["mean"] = mean
        self.normalization_used["std"] = std
        self._set_normalization_applied(True)

    def normalize(self, inputs):
        """This function normalizes the input tensor using the previously set mean and standard deviation.
        The normalization is performed channel-wise.

        :param inputs: The input tensor to be normalized.
        :param torch.Tensor: The normalized input tensor.
        """
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        """This function performs the inverse operation of the normalize function.
        It takes a normalized tensor and transforms it back to its original state using the set mean and standard deviation.

        :param inputs: The input tensor to be inverse normalized.

        :return torch.Tensor: The inverse normalized input tensor.
        """
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return inputs * std + mean

    def get_mode(self):
        """Get attack mode."""

        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        """Set attack mode as default mode."""
        self.attack_mode = "default"
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode, quiet):
        """This function sets the attack mode to 'targeted' if it is supported.
        If the targeted mode is not supported, it raises a ValueError.
        It also sets the 'targeted' attribute to True.

        :param mode: The attack mode to be set.
        :param quiet: If False, the function will print a message indicating the change in attack mode.

        :raise ValueError: If the targeted mode is not supported.

        """
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        if not quiet:
            print("Attack mode is changed to '%s'." % mode)

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function, quiet=False):
        """Set attack mode as targeted.

        :param target_map_function: Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)
        :param quiet: Display information message or not. (Default: False)
        """
        self._set_mode_targeted("targeted(custom)", quiet)
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self, quiet=False):
        """Set attack mode as targeted with random labels.

        :param quiet: Display information message or not. (Default: False)
        """

        self._set_mode_targeted("targeted(random)", quiet)
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1, quiet=False):
        """Set attack mode as targeted with least likely labels.

        :param kth_min: label with the k-th smallest probability used as target labels. (Default: 1)
        :param num_classses: number of classes. (Default: False)
        """
        self._set_mode_targeted("targeted(least-likely)", quiet)
        assert kth_min > 0
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_mode_targeted_by_label(self, quiet=False):
        """Set attack mode as targeted.

        :param quiet: Display information message or not. (Default: False)

        .. note::
            Use user-supplied labels as target labels.
        """
        self._set_mode_targeted("targeted(label)", quiet)
        self._target_map_function = "function is a string"

    @wrapper_method
    def set_model_training_mode(
        self,
        model_training=False,
        batchnorm_training=False,
        dropout_training=False,
    ):
        """Set training mode during attack process.

        :param model_training: True for using training mode for the entire model during attack process.
        :param batchnorm_training: True for using training mode for batchnorms during attack process.
        :param dropout_training: True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        """This function changes the mode of the model based on the 'given_training' flag.
        If 'given_training' is True, it sets the model to training mode.
        Additionally, it iterates over all modules in the model and sets BatchNorm and Dropout layers to evaluation mode
        if '_batchnorm_training' and '_dropout_training' flags are False, respectively.
        If 'given_training' is False, it sets the model to evaluation mode.

        :PARAM given_training (bool): Flag indicating whether the model should be in training mode or not.
        """
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if "BatchNorm" in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if "Dropout" in m.__class__.__name__:
                        m = m.eval()
            else:
                self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        """This function recovers the mode of the model based on the 'given_training' flag.
        If 'given_training' is True, it sets the model back to training mode.
        This function is typically used to revert the model back to its original mode after certain operations.

        :param given_training (bool): Flag indicating whether the model should be in training mode or not.
        """
        if given_training:
            self.model.train()

    def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        """Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        :param save_path: save_path.
        :param data_loader: data loader.
        :param verbose: True for displaying detailed information. (Default: True)
        :param return_verbose: True for returning detailed information. (Default: False)
        :param save_predictions: True for saving predicted labels (Default: False)
        :param save_clean_inputs: True for saving clean inputs (Default: False)
        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    @staticmethod
    def to_type(inputs, type):
        """Return inputs as int if float is given."""
        if type == "int":
            if isinstance(inputs, torch.FloatTensor) or isinstance(
                inputs, torch.cuda.FloatTensor
            ):
                return (inputs * 255).type(torch.uint8)
        elif type == "float":
            if isinstance(inputs, torch.ByteTensor) or isinstance(
                inputs, torch.cuda.ByteTensor
            ):
                return inputs.float() / 255
        else:
            raise ValueError(
                type + " is not a valid type. [Options: float, int]"
            )
        return inputs

    @staticmethod
    def _save_print(progress, rob_acc, l2, elapsed_time, end):
        """This function prints the progress of a certain operation, robust accuracy, L2 norm, and elapsed time per iteration.
        It's typically used for logging purposes during training or testing of models.

        :param progress: The progress of the operation in percentage.
        :param rob_acc: The robust accuracy of the model in percentage.
        :param l2: The L2 norm of the model parameters or gradients.
        :param elapsed_time: The time elapsed per iteration in seconds.
        :param end: The end character to be used in the print function. Typically '\n' for a new line or '\r' to overwrite the current line.
        """
        print(
            "- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t"
            % (progress, rob_acc, l2, elapsed_time),
            end=end,
        )

    @staticmethod
    def load(
        load_path,
        batch_size=128,
        shuffle=False,
        normalize=None,
        load_predictions=False,
        load_clean_inputs=False,
    ):
        save_dict = torch.load(load_path)
        keys = ["adv_inputs", "labels"]

        if load_predictions:
            keys.append("preds")
        if load_clean_inputs:
            keys.append("clean_inputs")

        if save_dict["save_type"] == "int":
            save_dict["adv_inputs"] = save_dict["adv_inputs"].float() / 255
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"].float() / 255
                )  # nopep8

        if normalize is not None:
            n_channels = len(normalize["mean"])
            mean = torch.tensor(normalize["mean"]).reshape(1, n_channels, 1, 1)
            std = torch.tensor(normalize["std"]).reshape(1, n_channels, 1, 1)
            save_dict["adv_inputs"] = (save_dict["adv_inputs"] - mean) / std
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"] - mean
                ) / std  # nopep8

        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(
            adv_data, batch_size=batch_size, shuffle=shuffle
        )
        print(
            "Data is loaded in the following order: [%s]" % (", ".join(keys))
        )  # nopep8
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        """This function loads a saved dataset from a given path and returns a DataLoader object.
        The function also supports optional normalization and loading of additional data such as predictions and clean inputs.

        :param load_path: The path to the saved dataset.
        :param batch_size: The batch size for the DataLoader. Defaults to 128.
        :param shuffle: Whether to shuffle the data. Defaults to False.
        :param normalize: A dictionary containing 'mean' and 'std' for normalization. Defaults to None.
        :param load_predictions: Whether to load predictions. Defaults to False.
        :param load_clean_inputs: Whether to load clean inputs. Defaults to False.

        :return DataLoader: A DataLoader object containing the loaded data.
        """
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(inputs)
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, inputs, labels=None):
        """Function for changing the attack mode.

        :return  input labels.
        """
        if self._target_map_function is None:
            raise ValueError(
                "target_map_function is not initialized by set_mode_targeted."
            )
        if self.attack_mode == "targeted(label)":
            target_labels = labels
        else:
            target_labels = self._target_map_function(inputs, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        """This function computes the least likely label for each input sample based on the model's output.
        If labels are not provided, it uses the model's predictions as labels.
        It's typically used in adversarial attack scenarios where the least likely label is targeted.

        :param inputs: The input samples.
        :param labels: The true labels of the input samples. Defaults to None.

        :return torch.Tensor: The least likely labels for each input sample.
        """
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        """
        This function computes a random target label for each input sample that is different from its true label.
        If labels are not provided, it uses the model's predictions as labels.
        It's typically used in adversarial attack scenarios where a random label is targeted.

        :param inputs: The input samples.
        :param labels: The true labels of the input samples. Defaults to None.

        :return torch.Tensor: The random target labels for each input sample.
        """
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l) * torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def __call__(self, inputs, labels=None, *args, **kwargs):
        """This function is the main entry point for this class. It's called when an instance of the class is invoked like a function.
        It performs the following steps:
        1. Saves the current training mode of the model.
        2. Changes the model's mode based on the saved training mode.
        3. If normalization has been applied, it inverses the normalization on the inputs and sets the '_normalization_applied' flag to False.
        4. Calls the 'forward' method to perform the main operation of the class (e.g., generating adversarial examples).
        5. If normalization has been applied, it normalizes the adversarial inputs and sets the '_normalization_applied' flag back to True.
        6. Recovers the model's mode to its original state.
        7. Returns the adversarial inputs.

        :param inputs: The input samples.
        :param labels: The true labels of the input samples. Defaults to None.
        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.

        :return torch.Tensor: The adversarial inputs.
        """
        given_training = self.model.training
        self._change_model_mode(given_training)

        if self._normalization_applied is True:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            # adv_inputs = self.to_type(adv_inputs, self.return_type)

            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            # adv_inputs = self.to_type(adv_inputs, self.return_type)

        self._recover_model_mode(given_training)

        return adv_inputs

    def __repr__(self):
        """This function provides a string representation of the object.
        It's typically used for debugging and logging purposes.
        The function copies the object's attributes dictionary, removes certain keys,
        and then formats the remaining items into a string.

        :return A string representation of the object.
        """
        info = self.__dict__.copy()

        del_keys = ["model", "attack", "supported_mode"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self.attack_mode
        info["normalization_used"] = (
            True if self.normalization_used is not None else False
        )

        return (
            self.attack
            + "("
            + ", ".join(f"{key}={val}" for key, val in info.items())
            + ")"
        )

    def __setattr__(self, name, value):
        """This function overrides the default behavior of the 'setattr' function.
        It's called when an attribute value is set.
        Besides setting the attribute value, it also updates the '_attacks' dictionary
        if the value is an instance of the 'Attack' class or contains instances of the 'Attack' class.
        :param name: The name of the attribute.
        :param value: The value of the attribute.
        """
        object.__setattr__(self, name, value)

        attacks = self.__dict__.get("_attacks")

        # Define a helper function to get all values in iterable items
        def get_all_values(items, stack=[]):
            if items not in stack:
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = list(items.keys()) + list(items.values())
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items

        # For each 'Attack' instance in the value, add it to the '_attacks' dictionary
        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get("_attacks").items():
                attacks[name + "." + subname] = subvalue
