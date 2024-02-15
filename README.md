# Various Attacks
This repository contains code for the Various Attacks method introduced in the paper 
"Powerful & Generalizable, Why not Both? VA: Various Attacks Framework for Robust Adversarial Training".
The VA framework trains image classification neural networks using a variety of adversarial 
attacks in order to significantly improve the generalization robustness.

# Prerequisites
- python = 3.9.18
- pytorch = 2.1.1
- pytorch-cuda = 11.8
- torchvision = 0.16.1
- AutoAttack

To install:
```
git clone https://github.com/jostdb/various-attacks
cd various-attacks
conda env create -f environment.yml
```

## Manifest
```
Various Attacks
├── attacks
│   └── (miscellaneous attack files)
├── cifar_data.py
├── environment.yml
├── helper_functions.py
├── losses.py
├── models
│   └── (resnet and wideresnet model architectures)
├── README.md (*)
├── train_va.py
└── warmup_round.py
```
To train a WideResNet-34-10 model using the Various Attacks method on the CIFAR-10 dataset:

```python train_va.py --dataset cifar10 --model-arch wideres34 --warmup 15 --epochs 75```

To use the Various Attacks method on the CIFAR-100 dataset:

```python train_va.py --dataset cifar100 --model-arch wideres34 --warmup 25 --epochs 140```

## Reference Code
1. ADT: https://github.com/dongyp13/Adversarial-Distributional-Training
2. CIFAR-100 Dataloader: https://github.com/xiaodongww/pytorch/tree/master
3. AdverTorch: https://github.com/BorealisAI/advertorch

## Citing Us
Add a bibtex citation for your paper.  ::TODO
