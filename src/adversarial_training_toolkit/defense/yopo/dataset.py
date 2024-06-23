import torch
import torchvision
import torchvision.transforms as transforms


def create_train_dataset(ds_name, batch_size=128, root="../data"):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    if ds_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )
    elif ds_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train
        )
    # trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return trainloader


def create_test_dataset(ds_name, batch_size=128, root="../data"):
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    if ds_name == "cifar10":
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
    elif ds_name == "cifar100":
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test
        )
    # testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )
    return testloader


if __name__ == "__main__":
    print(create_train_dataset("cifar10"))
    print(create_test_dataset("cifar10"))
