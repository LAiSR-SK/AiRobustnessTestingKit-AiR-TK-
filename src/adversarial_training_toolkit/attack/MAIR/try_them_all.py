# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from attacks.attacks.apgd import APGD
from attacks.attacks.apgdt import APGDT
from attacks.attacks.autoattack import AutoAttack
from attacks.attacks.bim import BIM
from attacks.attacks.difgsm import DIFGSM
from attacks.attacks.eaden import EADEN
from attacks.attacks.eadl1 import EADL1
from attacks.attacks.eotpgd import EOTPGD
from attacks.attacks.fab import FAB
from attacks.attacks.ffgsm import FFGSM
from attacks.attacks.gn import GN
from attacks.attacks.jitter import Jitter
from attacks.attacks.jsma import JSMA
from attacks.attacks.mifgsm import MIFGSM
from attacks.attacks.nifgsm import NIFGSM
from attacks.attacks.pgdl2 import PGDL2
from attacks.attacks.pgdrs import PGDRS
from attacks.attacks.pgdrsl2 import PGDRSL2
from attacks.attacks.pifgsm import PIFGSM
from attacks.attacks.pifgsmpp import PIFGSMPP
from attacks.attacks.pixle import Pixle
from attacks.attacks.rfgsm import RFGSM
from attacks.attacks.sinifgsm import SINIFGSM
from attacks.attacks.sparsefool import SparseFool
from attacks.attacks.spsa import SPSA
from attacks.attacks.square import Square
from attacks.attacks.tifgsm import TIFGSM
from attacks.attacks.tpgd import TPGD
from attacks.attacks.upgd import UPGD
from attacks.attacks.vanila import VANILA
from attacks.attacks.vmifgsm import VMIFGSM
from attacks.attacks.vnifgsm import VNIFGSM
from attacks.attacks.cw import CW
from attacks.attacks.deepfool import DeepFool
from attacks.attacks.fgsm import FGSM
from attacks.attacks.onepixel import OnePixel
from attacks.attacks.pgd import PGD


# Define a simple CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()

# Define the attacks
attacks = [VNIFGSM,VMIFGSM,VANILA,UPGD,TPGD,Square,SPSA,SparseFool,SINIFGSM,RFGSM,Pixle,
           PGDRSL2,PGDRS,PGDL2,NIFGSM,MIFGSM,JSMA,Jitter,GN,FFGSM,FAB,EOTPGD,EADL1,EADEN,DIFGSM,BIM,AutoAttack,
           APGDT,APGD,FGSM, PGD, CW, DeepFool, OnePixel]
attacks2 = [TIFGSM,PIFGSMPP,PIFGSM,]
# Load the MNIST dataset
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_loader = DataLoader(mnist, batch_size=1, shuffle=True)

# Get a single image from the MNIST dataset
dataiter = iter(mnist_loader)
images, labels = next(dataiter)

adversarial_examples = {}

for attack in attacks:
    attacker = attack(model)
    adversarial_example = attacker(images,labels)
    adversarial_examples[attack.__name__] = adversarial_example



fig, axes = plt.subplots(1, len(adversarial_examples))

for i, (attack_name, adversarial_example) in enumerate(adversarial_examples.items()):
    adversarial_example_np = adversarial_example.detach().cpu().numpy()

    # If the image has 4 dimensions, take the first dimension (assuming it's a single image)
    if len(adversarial_example_np.shape) == 4:
        adversarial_example_np = adversarial_example_np[0, 0]

    # Plot the adversarial example in the current subplot
    axes[i].imshow(adversarial_example_np,cmap="gray")
    axes[i].axis('off')

# Show the plot
plt.show()