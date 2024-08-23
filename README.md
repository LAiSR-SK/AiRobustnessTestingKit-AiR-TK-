<!-- Logo needs to be touched up; add our name-->
![Logo](asset/repo/image/Logo.webp)
<!--The badges will not work until our repo is public-->
<!-- We should add badges for Huggingface, PyPI, and Conda -->
![License](https://img.shields.io/github/license/LAiSR-SK/ImagePatriot) ![Code Style](https://img.shields.io/badge/code_style-Ruff-orange)

**AI Robustness Testing Kit (AiR-TK)** is an AI testing framework built upon PyTorch that enables the AI security community to evaluate the AI models against adversarial attacks easily and comprehensively. Furthermore, Air-TK supports adversarial training, the de-facto technique to improve the robustness of AI models against adversarial attacks. Having easy access to state-of-the-art adversarial attacks and the baseline adversarial training method in one place will help the AI security community to replicate, re-use, and improve the upcoming attacks and defense methods.   

## Motivation & Objectives
- Provide a one-stop framework that covers the most recent adversarial attacks and adversarial training methods for easy evaluation and comparison.  
- Promote and fasten the process of AI security research by providing pre-trained models of the base-line adversarial training methods.
- Provide a _Model-ZOO_ that contains pre-trained models for the most recent adversarial training (AT) method to promote reproducibility and fasten the comparison process with other AT methods.

## Installation
<!-- We will want a PyPI or conda package in the future; this is a very temporary solution -->
Our work is available via this repository. In order to install from here, you will need:
- The [Conda](https://www.anaconda.com/) environment manager.
- The [Git](https://www.git-scm.com/) version control system.

### Repository
First our repository will need to be cloned:
```bash
git clone https://github.com/LAiSR-SK/ImagePatriot.git
```

<!-- Do we want to provide a Linux environment?-->
Then create and activate the conda environment:
```bash
conda env create -p .conda

conda activate ./.conda
```

### Pretrained Models
<!-- We need to add our huggingface models -->
In order to expedite progress in the field of secure AI, we have provided the weights of our trained models on [huggingface](). More information on our achieved benchmarks is available [below](#benchmarks).

## Examples
<!-- we should finalize the interface before we keep these. The current one needs to be redone -->
### Attacks
You can import and use our attacks as shown:
```python
from torch import nn

from lib.attack import FGSMAttack
from lib.data import CIFAR100
from lib.model import Resnet34


if __name__ == "__main__":
  model = Resnet34()
  model.train()

  dataset = CIFAR100(root="data/download")

  attack = FGSMAttack(model.forward, nn.CrossEntropyLoss)

  attack.perturb(dataset)
```

<!-- We should add sections for defenses, models, etc. -->

## Benchmarks
<!-- What tables do we use? Original tables?-->


## Cite Us
See [CITATION.cff](CITATION.cff) for details on how to cite our work.

## Contributors
- Samer Khamaiseh, PhD.: Director of LAiSR Research Group
- Steven Chiacchira: Research Assistant at LAiSR
- Aibak Aljadayah: Research Assistant at LAiSR
- Dierdre Jost: Research Assistant at LAiSR
