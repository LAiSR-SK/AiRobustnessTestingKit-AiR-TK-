from adversarial_training_toolkit.data import CIFAR10
from adversarial_training_toolkit.interface.attack.pgd import pgd
from adversarial_training_toolkit.model import ResNet34
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor

if __name__ == "__main__":
    model: ResNet34 = ResNet34(10)
    data: DataLoader = DataLoader(
        CIFAR10("data/download", transform=ToTensor()), batch_size=64
    )

    for batch in data:
        print(type(batch))
        x, _ = batch

        x = pgd(model, x, 6.0, 40, 2)
        print(x.shape)

        ToPILImage()(x[0]).show()
