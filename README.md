# Dynamics-Aware Loss for Learning with Label Noise

This repository is the official implementation of [Dynamics-Aware Loss for Learning with Label Noise].

## Usage

(1) *CIFAR*

Download and unzip CIFAR-10, CIFAR-100 to `data/cifar10-data` and `data/cifar100-data` folder.

```(bash)
python cifar.py\
    -dataset {cifar10 or cifar100}\
    -mode {label noise mode: symmetric, asymmetric or instance}\ 
    -noise {label noise rate: 20 or 40 for any noise, and 60 or 80 for only symmetric noise}\ 
    -loss {loss functions: ce, gce, sce, nlnl, btl, nce_and_rce, tce, nce_and_agce, sr, js or dal}\
```

Download Mini-WebVision to `data/mini-webvision-data/train` and `data/mini-webvision-data/val` folder.

