import os
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse

from misc import set_seed
from cfg import *

p = argparse.ArgumentParser()
p.add_argument('--network', choices=["resnet18"], default="resnet18")
p.add_argument('--seed', type=int, default=7)
p.add_argument('--num-workers', type=int, default=2)
args = p.parse_args()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
set_seed(args.seed)
batch_size = 256
data_path = os.path.join(data_path, "CIFAR10")
save_path = os.path.join(results_path, "cifar10-pretrain-resnet18-seed7")
epochs = 100
warmup_epoch = 1
lr = 0.1
weight_decay = 5e-4
momentum = 0.9

# Data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = CIFAR10(root = data_path, train = True, download = False, transform = train_transform)
train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=args.num_workers)
test_data = CIFAR10(root = data_path, train = False, download = False, transform = test_transform)
test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=args.num_workers)

# Network
network = eval(args.network)(num_classes = 10)
network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
network.maxpool = nn.Identity()
network = network.to(device)

# Optimizer
optimizer = optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Make Dir
os.makedirs(save_path, exist_ok=True)
logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

# Train
best_acc = 0. 
scaler = GradScaler()
for epoch in range(epochs):
    network.train()
    total_num = 0
    true_num = 0
    loss_sum = 0
    pbar = tqdm(train_loader, total=len(train_loader),
            desc=f"Train Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():
            fx = network(x)
            loss = F.cross_entropy(fx, y, reduction='mean')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        loss_sum += loss.item() * fx.size(0)
        pbar.set_postfix_str(f"Acc {100*true_num/total_num:.2f}%")
    logger.add_scalar("train/acc", true_num/total_num, epoch)
    logger.add_scalar("train/loss", loss_sum/total_num, epoch)
    scheduler.step()

    # Test
    network.eval()
    fxs = []
    ys = []
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating", ncols=100)
    for x, y in pbar:
        x = x.to(device)
        with torch.no_grad(), autocast():
            fx = network(x)
        fxs.append(fx)
        ys.append(y)
    fxs = torch.cat(fxs).cpu().float()
    ys = torch.cat(ys).cpu()
    acc = torch.argmax(fxs, -1).eq(ys).float().mean()
    logger.add_scalar("test/acc", acc, epoch)

    # Save CKPT
    state_dict = {
        "network_dict": network.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if acc > best_acc:
        best_acc = acc
        state_dict['best_acc'] = acc
        torch.save(state_dict, os.path.join(save_path, 'best.pth'))
    torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))