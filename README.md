# Dynamics-Aware Loss for Learning with Label Noise

This repository is the official implementation of [Dynamics-Aware Loss for Learning with Label Noise].

## Usage

(1) *CIFAR*

Download and unzip CIFAR-10, CIFAR-100 to `data/cifar10-data` and `data/cifar100-data` folder.

```(bash)
python cifar.py\
    --dataset {cifar10, cifar100}\
    --mode {label noise mode: symmetric, asymmetric, instance}\ 
    --noise {label noise rate: 20 or 40 for any noise, and 60 or 80 for only symmetric noise}\ 
    --loss {loss functions: ce, gce, sce, nlnl, btl, nce_and_rce, tce, nce_and_agce, sr, js, dal}\
    --q {hyper-parameter of gce and nce_and_agce}
    --alpha {hyper-parameter of sce, nce_and_rce, and nce_and_agce}
    --beta {hyper-parameter of sce, nce_and_rce, and nce_and_agce}
    --N {hyper-parameter of nlnl}
    --t1 {hyper-parameter of btl}
    --t2 {hyper-parameter of btl}
    --t {hyper-parameter of tce}
    --a {hyper-parameter of nce_and_agce}
    --tau {hyper-parameter of sr}
    --p {hyper-parameter of sr}
    --lamb {hyper-parameter of sr}
    --rho {hyper-parameter of sr}
    --pi {hyper-parameter of js}
    --qs {hyper-parameter of dal}
    --qe {hyper-parameter of dal}
```

(2) *Mini-WebVision*
Download Mini-WebVision to `data/mini-webvision-data/train` and `data/mini-webvision-data/val` folder.

```(bash)
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 webvision.py\
    --loss {loss functions: ce, sce, nce_and_rce, agce, sr, js, dal}\
    --q {hyper-parameter of agce}
    --alpha {hyper-parameter of sce, and nce_and_rce}
    --beta {hyper-parameter of sce, and nce_and_rce}
    --a {hyper-parameter of agce}
    --tau {hyper-parameter of sr}
    --p {hyper-parameter of sr}
    --lamb {hyper-parameter of sr}
    --rho {hyper-parameter of sr}
    --pi {hyper-parameter of js}
    --qs {hyper-parameter of dal}
    --qe {hyper-parameter of dal}
```

## Requirements
- python 3.8.13
- numpy 1.23.5
- pytorch 1.8.0
- torchvision 0.9.0
