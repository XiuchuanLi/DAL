import argparse
from tqdm import tqdm
import csv
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.losses as losses

# new
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--loss", type=str, default='dal')
parser.add_argument("--alpha", type=float, default=50.0)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--a", type=float, default=1e-5)
parser.add_argument("--q", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=0.5)
parser.add_argument("--p", type=float, default=0.01)
parser.add_argument("--lamb", type=float, default=2.0)
parser.add_argument("--rho", type=float, default=1.02)
parser.add_argument("--pi", type=float, default=0.1)
parser.add_argument("--qs", type=float, default=0.4)
parser.add_argument("--qe", type=float, default=1.5)
args = parser.parse_args()
local_rank = args.local_rank

# DDP backend initialization
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

# load data after DDP backend initialization
logging.basicConfig(level=logging.INFO if dist.get_rank() == 0 else logging.WARN)
logging.info(f"Load Mini Webvision")
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_dataset = datasets.ImageFolder(root='data/mini-webvision-data/train', transform=transform_train)
test_dataset = datasets.ImageFolder(root='data/mini-webvision-data/val', transform=transform_test)
num_categories = 50

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, sampler=train_sampler, num_workers=16, pin_memory=True)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)

logging.info("Load Model")
model = models.resnet50(num_classes=num_categories).to(local_rank)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

epochs = 250
optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=3.e-5, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
avg_loss = losses.AverageMeter()

if args.loss == 'ce':
    criterions = [nn.CrossEntropyLoss().to(local_rank) for _ in range(epochs)]
elif args.loss == 'sce':
    criterions = [losses.SCE(num_classes=num_categories, alpha=args.alpha, beta=args.beta).to(local_rank) for _ in range(epochs)]
elif args.loss == 'nce_and_rce':
    criterions = [losses.NCEandRCE(num_classes=num_categories, alpha=args.alpha, beta=args.beta).to(local_rank) for _ in range(epochs)]
elif args.loss == 'agce':
    criterions = [losses.AGCE(num_classes=num_categories, a=args.a, q=args.q).to(local_rank) for _ in range(epochs)]
elif args.loss == 'ce_and_sr':
    criterions = [losses.CEandSR(lamb=args.lamb * (args.rho ** i), tau=args.tau, p=args.p).to(local_rank) for i in range(epochs)]
elif args.loss == 'js':
    criterions = [losses.JS(num_classes=num_categories, pi=args.pi).to(local_rank) for _ in range(epochs)]
elif args.loss == 'dal':
    t0 = (1 - args.qs) * epochs / (args.qe - args.qs)
    criterions = [losses.DAL(num_classes=num_categories, q=args.qs + (args.qe - args.qs) * (i / epochs),
                  lamb=max(0, (i - t0) / (epochs - t0))) for i in range(epochs)]
else:
    raise ValueError

for epoch in range(1, epochs+1):
    # train
    model.train()
    trainloader.sampler.set_epoch(epoch)
    loop = tqdm(trainloader)
    avg_loss.reset()
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(local_rank), target.to(local_rank)
        logits = model(data)
        loss = criterions[epoch-1](logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        avg_loss.update(loss.item(), len(data))
        loop.set_description(f"Epoch {epoch}/{epochs} lr {optimizer.param_groups[0]['lr']:05.4e} loss {avg_loss.avg:05.4e}")
        loop.update()
    scheduler.step()
    # test
    correct, total = 0, 0
    model.eval()
    if dist.get_rank() == 0:
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
        with open(f'{args.loss}/webvision.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([epoch, correct * 100 / total])
