import argparse
from tqdm import tqdm
import csv
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from resnet import ResNet18
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import utils.noisy_loader as noisy_loader
from utils.logger import get_logger
import utils.losses as losses


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='cifar100')
parser.add_argument("--mode", type=str, default='instance')
parser.add_argument("--noise", type=int, default=40)
parser.add_argument("--loss", type=str, default='dal')
parser.add_argument("--q", type=float, default=0.7)
parser.add_argument("--alpha", type=float, default=10.0)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--N", type=int, default=110)
parser.add_argument("--t1", type=float, default=0.7)
parser.add_argument("--t2", type=float, default=1.5)
parser.add_argument("--t", type=int, default=18)
parser.add_argument("--a", type=float, default=3.0)
parser.add_argument("--tau", type=float, default=0.5)
parser.add_argument("--p", type=float, default=0.01)
parser.add_argument("--lamb", type=float, default=8.0)
parser.add_argument("--rho", type=float, default=1.02)
parser.add_argument("--pi", type=float, default=0.5)
parser.add_argument("--qs", type=float, default=0.6)
parser.add_argument("--qe", type=float, default=1.5)
args = parser.parse_args()

LOGGER = get_logger(__name__, level="DEBUG")

LOGGER.info(f"Load Data {args.dataset}")

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = noisy_loader.NoisyCIFAR10(root='data/cifar10-data', noise=args.noise, mode=args.mode,
                                              train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='data/cifar10-data',
                                    train=False, transform=transform_test, download=True)
    num_categories = 10
elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = noisy_loader.NoisyCIFAR100(root='data/cifar100-data', noise=args.noise, mode=args.mode,
                                               train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR100(root='data/cifar100-data',
                                     train=False, transform=transform_test, download=True)
    num_categories = 100
else:
    raise ValueError

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

LOGGER.info("Load Model")
model = ResNet18(num_categories).cuda()

LOGGER.info("Train")
epochs = 150
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
avg_loss = losses.AverageMeter()

if args.loss == 'ce':
    criterions = [nn.CrossEntropyLoss() for _ in range(epochs)]
elif args.loss == 'gce':
    criterions = [losses.GCE(num_classes=num_categories, q=args.q) for _ in range(epochs)]
elif args.loss == 'sce':
    criterions = [losses.SCE(num_classes=num_categories, alpha=args.alpha, beta=args.beta) for _ in range(epochs)]
elif args.loss == 'nlnl':
    criterions = [losses.NLNL(trainloader, num_classes=num_categories, ln_neg=args.N) for _ in range(epochs)]
elif args.loss == 'btl':
    criterions = [losses.BTL(num_classes=num_categories, t1=args.t1, t2=args.t2) for _ in range(epochs)]
elif args.loss == 'nce_and_rce':
    criterions = [losses.NCEandRCE(num_classes=num_categories, alpha=args.alpha, beta=args.beta) for _ in range(epochs)]
elif args.loss == 'tce':
    criterions = [losses.TCE(num_classes=num_categories, order=args.t) for _ in range(epochs)]
elif args.loss == 'nce_and_agce':
    criterions = [losses.NCEandAGCE(num_classes=num_categories, a=args.a, q=args.q, alpha=args.alpha, beta=args.beta) for _ in range(epochs)]
elif args.loss == 'sr':
    criterions = [losses.SR(lamb=args.lamb * (args.rho ** i), tau=args.tau, p=args.p) for i in range(epochs)]
elif args.loss == 'js':
    criterions = [losses.JS(num_classes=num_categories, pi=args.pi) for _ in range(epochs)]
elif args.loss == 'dal':
    t0 = (1 - args.qs) * epochs / (args.qe - args.qs)
    criterions = [losses.DAL(num_classes=num_categories, q=args.qs + (args.qe - args.qs) * (i / epochs),
                  lamb=max(0, (i - t0) / (epochs - t0))) for i in range(epochs)]
else:
    raise ValueError

for epoch in range(1, epochs+1):
    # train
    model.train()
    loop = tqdm(trainloader)
    avg_loss.reset()
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterions[epoch - 1](logits, target)
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item(), len(data))
        loop.set_description(f"Epoch {epoch}/{epochs} lr {optimizer.param_groups[0]['lr']:05.4e}"
                             f" loss {avg_loss.avg:05.4e}")
        loop.update()
    scheduler.step()
    # test
    correct = 0
    total = 0
    model.eval()
    for data, target in testloader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            logits = model(data)

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()
        total += data.shape[0]

    print(f'-----------------{correct * 100 / total}-----------------')
    if not os.path.exists(f'{args.loss}'):
        os.mkdir(f'{args.loss}')
    if not os.path.exists(f'{args.loss}/{args.mode}'):
        os.mkdir(f'{args.loss}/{args.mode}')
    with open(f'{args.loss}/{args.mode}/{args.dataset}_{args.noise}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([epoch, correct * 100 / total])