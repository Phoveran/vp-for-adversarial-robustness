import os
import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tensorboardX import SummaryWriter
import argparse
import math
from tqdm import tqdm
from torch.nn.functional import cross_entropy

from losses import cwloss
from attack import AutoAttack, attack_pgd_restart, ctx_noparamgrad
from model.visual_prompt import VisualPrompt
from misc import set_seed, gen_folder_name, predict_diagonal
from cfg import *


def train_once(prompt, network, train_loader, optimizer, scheduler, epoch, loss_type, τ, β, cw1, cw2, cw3, attacker):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    scaler = GradScaler()
    prompt.train()
    total_num = 0
    clean_true_num = 0
    adv_true_num = 0
    loss_sum = 0
    pbar = tqdm(train_loader, total=len(train_loader),
            desc=f"Train Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
    for x, y in pbar:
        pbar.set_description_str(f"Train Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", refresh=True)
        x, y = x.to(device), y.to(device)
        prompted_xs = prompt(x)
        optimizer.zero_grad()
        fxs = []
        loss = 0
        with autocast():
            for prompted_x in prompted_xs:
                fx_tmp_holder = []
                fx_tmp_holder.append(network(prompted_x).unsqueeze(-2).unsqueeze(0))
                if attacker is not None:
                    prompted_x = attacker(prompted_x, y)
                    fx_tmp_holder.append(network(prompted_x).unsqueeze(-2).unsqueeze(0))
                fx_tmp_holder = torch.cat(fx_tmp_holder, dim=0)
                fxs.append(fx_tmp_holder)
            fxs = torch.cat(fxs, dim=-2)
            if loss_type == 'ce':
                if fxs.size(-2) == 1:
                    loss += cross_entropy(fxs[0].squeeze(), y) # CE
                    if fxs.size(0) == 2:
                        loss += cross_entropy(fxs[1].squeeze(), y) # ACE
                else:
                    for i in range(fxs.size(-2)):
                        loss += cross_entropy(fxs[0, y==i, i, :], y[y==i]) # CE
                        if fxs.size(0) == 2:
                            loss += cross_entropy(fxs[1, y==i, i, :], y[y==i]) # ACE
            elif loss_type == 'ce+cw':
                assert fxs.size(-2) != 1
                for i in range(fxs.size(-2)):
                    loss += cross_entropy(fxs[0, y==i, i, :], y[y==i]) # CE
                    if fxs.size(0) == 2:
                        loss += cross_entropy(fxs[1, y==i, i, :], y[y==i]) # ACE
                loss += β * cwloss(fxs[0], y, τ, cw1, cw2, cw3) # CW
            else:
                raise NotImplementedError("loss not implemented")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if fxs.size(-2) != 1:
            preds = predict_diagonal(fxs)
        else:
            preds = torch.argmax(fxs[:, :, 0, :], dim=-1)
        if preds.size(0) == 2:
            adv_true_num += preds[1].eq(y).float().sum().item()
        clean_true_num += preds[0].eq(y).float().sum().item()
        total_num += y.size(0)
        loss_sum += loss.item() * y.size(0)
        loss = loss_sum/total_num
        clean_acc = clean_true_num/total_num
        adv_acc = adv_true_num/total_num
        pbar.set_postfix_str(f"C-Acc {clean_acc*100:.2f}% Loss {loss:.4f}")
    return loss, clean_acc, adv_acc
    

def test_once(prompt, network, test_loader, attacker=None, classwise=False):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")    
    prompt.eval()
    total_num = 0
    true_num = 0
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Test {'Clean' if attacker is None else 'Adversarial'} Set {'Classwise' if classwise else ''}", ncols=100)
    for x, y in pbar:
        total_num += y.size(0)
        x, y = x.to(device), y.to(device)
        if attacker is not None:
            x = attacker(x, y)
        prompted_xs = prompt(x)
        if classwise:
            for i, prompted_x in enumerate(prompted_xs):
                with torch.no_grad():
                    fx = network(prompted_x)
                true_num += ((y == i).float() * (torch.argmax(fx, dim=-1) == i).float()).sum()
        else:
            fx_p = []
            for prompted_x in prompted_xs:
                with torch.no_grad():
                    fx = network(prompted_x)
                fx_p.append(fx.unsqueeze(-2))
            fx_p = torch.cat(fx_p, -2)
            if fx_p.size(-2) != 1:
                pred = predict_diagonal(fx_p)
            else:
                pred = torch.argmax(fx_p.squeeze(), dim=-1)
            true_num += pred.eq(y).float().sum().item()
        acc = true_num/total_num
        pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
    return acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18"], default="resnet18")
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--num-workers', type=int, default=2)

    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--epoch', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--loss', choices=["ce", "ce+cw"], default="ce+cw")

    p.add_argument('--classwise', action='store_true')
    p.add_argument('--shape', type=str, choices=["pad", "full"], default="pad")
    p.add_argument('--size', type=int, default=8)
    p.add_argument('--np-clip', action='store_false', dest="clip")

    p.add_argument('--attack', choices=['aa', 'pgd', 'none'], default='pgd')
    p.add_argument('--ε', type=float, default=8./255)

    p.add_argument('--no-cw1', action='store_false', dest='cw1')
    p.add_argument('--no-cw2', action='store_false', dest='cw2')
    p.add_argument('--no-cw3', action='store_false', dest='cw3')
    p.add_argument('--τ', type=float, default=0.1)
    p.add_argument('--β', type=float, default=3.)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    data_path = os.path.join(data_path, "CIFAR10")
    save_path = os.path.join(results_path, gen_folder_name(args))

    # Data
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()
    train_data = CIFAR10(root = data_path, train = True, download = False, transform = train_transform)
    train_loader = DataLoader(train_data, args.batch_size, shuffle = True, num_workers=args.num_workers)
    test_data = CIFAR10(root = data_path, train = False, download = False, transform = test_transform)
    test_loader = DataLoader(test_data, args.batch_size, shuffle = False, num_workers=args.num_workers)

    # Network
    if args.network == "resnet18":
        network = resnet18(pretrained=False, num_classes=10)
        network.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        network.maxpool = torch.nn.Identity()
        network.load_state_dict(torch.load(os.path.join(results_path, "cifar10-pretrain-resnet18-seed7", "best.pth"))["network_dict"])
    else:
        raise NotImplementedError("network not implemented")
    network.requires_grad_(False)
    network = network.to(device)
    network.eval()

    # Linf Attacker
    if args.attack == 'aa':
        train_attacker = AutoAttack(network, eps=args.eps, seed=args.seed)
        test_attacker = train_attacker
    elif args.attack == 'pgd':
        def train_attacker(x, y):
            with ctx_noparamgrad(network):
                adv_delta = attack_pgd_restart(
                    model=network,
                    X=x,
                    y=y,
                    eps=args.ε,
                    alpha=args.ε / 4,
                    attack_iters=10,
                    n_restarts=1,
                    rs=True,
                    verbose=False,
                    linf_proj=True,
                    l2_proj=False,
                    l2_grad_update=False,
                    cuda=torch.cuda.is_available()
                )
            return x + adv_delta
        def test_attacker(x, y):
            with ctx_noparamgrad(network):
                adv_delta = attack_pgd_restart(
                    model=network,
                    X=x,
                    y=y,
                    eps=args.ε,
                    alpha=args.ε / 4,
                    attack_iters=10,
                    n_restarts=1,
                    rs=True,
                    verbose=False,
                    linf_proj=True,
                    l2_proj=False,
                    l2_grad_update=False,
                    cuda=torch.cuda.is_available()
                )
            return x + adv_delta
    else:
        train_attacker = None
        def test_attacker(x, y):
            with ctx_noparamgrad(network):
                adv_delta = attack_pgd_restart(
                    model=network,
                    X=x,
                    y=y,
                    eps=args.ε,
                    alpha=args.ε / 4,
                    attack_iters=10,
                    n_restarts=1,
                    rs=True,
                    verbose=False,
                    linf_proj=True,
                    l2_proj=False,
                    l2_grad_update=False,
                    cuda=torch.cuda.is_available()
                )
            return x + adv_delta

    # Visual Prompt
    prompt = VisualPrompt((3,32,32), args.shape, args.size, len(train_data.classes) if args.classwise else 1, args.clip).to(device)

    # Optimizer
    optimizer = optim.SGD(prompt.parameters(), lr=args.lr, momentum=0.9)
    t0 = args.epoch * math.ceil(len(train_data) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t0)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Train
    best_avg_acc = 0.
    for epoch in range(args.epoch):
        train_loss, train_std_acc, train_adv_acc = train_once(prompt, network, train_loader, optimizer, scheduler, epoch, args.loss, args.τ, args.β, args.cw1, args.cw2, args.cw3, train_attacker)
        logger.add_scalar("train/loss", train_loss, epoch)
        logger.add_scalar("train/std_acc", train_std_acc, epoch)
        logger.add_scalar("train/adv_acc", train_adv_acc, epoch)
        std_acc = test_once(prompt, network, test_loader)
        logger.add_scalar("test/std_acc", std_acc, epoch)
        adv_acc = test_once(prompt, network, test_loader, test_attacker)
        logger.add_scalar("test/adv_acc", adv_acc, epoch)
        if args.classwise:
            cls_std_acc = test_once(prompt, network, test_loader, classwise=True)
            logger.add_scalar("test/cls_std_acc", cls_std_acc, epoch)
            cls_adv_acc = test_once(prompt, network, test_loader, test_attacker, classwise=True)
            logger.add_scalar("test/cls_adv_acc", cls_adv_acc, epoch)

        # Save CKPT
        state_dict = {
            "prompt_dict": prompt.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_avg_acc": best_avg_acc,
        }
        avg_acc = (adv_acc + std_acc) / 2
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            state_dict['best_avg_acc'] = best_avg_acc
            state_dict['std_acc'] = std_acc
            state_dict['adv_acc'] = adv_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))