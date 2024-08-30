import torch


from airtk.model import WideResNet

if __name__ == "__main__":
    model = WideResNet(num_classes=100)
    model.load_state_dict(torch.load("data/model/cifar100/clean-cifar100-wideres34.pt"))
    print(model.forward(torch.zeros([1, 3, 32, 32])))
