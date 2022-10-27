import os
import torch
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
# from robustbench.utils import load_model
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from attack import AutoAttack, attack_pgd_restart, ctx_noparamgrad
from model.visual_prompt import VisualPrompt
from misc import set_seed
from cfg import *

def predict(prompt, network, test_loader, attacker=None):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")    
    prompt.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=100)
    fx_all = []
    ys = []
    for x, y in pbar:
        ys.append(y)
        x, y = x.to(device), y.to(device)
        if attacker is not None:
            x = attacker(x, y)
        prompted_xs = prompt(x)
        fx_p = []
        for prompted_x in prompted_xs:
            with torch.no_grad():
                fx = network(prompted_x)
            fx_p.append(fx.unsqueeze(-2))
        fx_p = torch.cat(fx_p, -2)
        fx_all.append(fx_p)
    return torch.cat(fx_all), torch.cat(ys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18"], default="resnet18")
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--num-workers', type=int, default=2)

    p.add_argument('--shape', type=str, choices=["pad", "full"], default="pad")
    p.add_argument('--size', type=int, default=8)
    p.add_argument('--np-clip', action='store_false', dest="clip")
    
    p.add_argument('--ckpt-path', type=str)

    p.add_argument('--attack', choices=['aa', 'pgd'], default="pgd")
    p.add_argument('--ε', type=float, default=8./255)

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Data
    data_path = os.path.join(data_path, "CIFAR10")
    test_data = CIFAR10(root = data_path, train = False, download = False, transform = transforms.ToTensor())
    test_loader = DataLoader(test_data, 256, shuffle = False, num_workers=args.num_workers)

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

    # Visual Prompt
    try:
        prompt = VisualPrompt((3, 32, 32), args.shape, args.size, 1, args.clip).to(device)
        if args.ckpt_path is not None:
            prompt.load_state_dict(torch.load(os.path.join(args.ckpt_path, "best.pth"))["prompt_dict"])
    except RuntimeError:
        prompt = VisualPrompt((3, 32, 32), args.shape, args.size, 10, args.clip).to(device)
        if args.ckpt_path is not None:
            prompt.load_state_dict(torch.load(os.path.join(args.ckpt_path, "best.pth"))["prompt_dict"])
    prompt.requires_grad_(False)
    prompt.eval()

    adv_accs = []
    for PGD_step in [0, 1, 10, 20, 50, 100]:
        # Linf Attacker
        if args.attack == 'aa':
            train_attacker = AutoAttack(network, eps=args.eps, seed=args.seed)
            test_attacker = train_attacker
        elif args.attack == 'pgd':
            def attacker(x, y):
                with ctx_noparamgrad(network):
                    adv_delta = attack_pgd_restart(
                        model=network,
                        X=x,
                        y=y,
                        eps=args.ε,
                        alpha=args.ε * 2.5 / (PGD_step+0.0001),
                        attack_iters=PGD_step,
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
            raise NotImplementedError('Attacker not Supported')

        pred_adv, y = predict(prompt, network, test_loader, attacker=attacker)
        if pred_adv.size(-2) == 1:
            pred_adv = torch.softmax(pred_adv.squeeze(), dim=-1)
        else:
            pred_adv = torch.diagonal(torch.softmax(pred_adv, dim=-1), dim1=-2, dim2=-1)
        pred_adv = torch.argmax(pred_adv, -1).cpu()

        adv_acc = pred_adv.eq(y).float().mean().item()
        adv_accs.append(adv_acc)

    print(adv_accs)