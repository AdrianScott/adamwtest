# let's focus on using this for now
import argparse, csv, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import wandb

# Import our custom optimizer
from custom_optimizer import CustomAdamW

# ----------------------------
# Utilities
# ----------------------------
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def param_groups_weight_decay(model: nn.Module, weight_decay: float, exclude_bias_norm: bool = True):
    """Return two param groups: one with WD for weights (ndim>=2), one without for biases/norms."""
    if not exclude_bias_norm:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)       # conv/linear weights
        else:
            no_decay.append(p)    # biases, norm params, etc.
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class SGDW(SGD):
    """Decoupled weight decay for SGD."""
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0, nesterov=False):
        # Use weight_decay=0 in base optimizer; apply decoupled WD manually per group.
        # Preserve per-group intent in a separate key to avoid coupled L2 in base SGD.
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=0.0, nesterov=nesterov)
        self.decoupled_wd = weight_decay  # fallback if groups don't specify
        for g in self.param_groups:
            g["decoupled_weight_decay"] = g.get("weight_decay", self.decoupled_wd)
            g["weight_decay"] = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # decoupled shrink (respect per-group weight_decay)
        for group in self.param_groups:
            wd = group.get("decoupled_weight_decay", self.decoupled_wd)
            if wd != 0.0:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.data.mul_(1 - lr * wd)

        # then standard SGD update (without built-in wd)
        super().step(closure=None)
        return loss


def make_cifar_resnet18(num_classes=10):
    """CIFAR-10 friendly ResNet-18: 3x3 conv, stride=1, no maxpool."""
    m = models.resnet18(weights=None)
    # modify stem for CIFAR
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / B).item())
        return res


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, args):
    model.train()
    loss_meter, acc_meter = 0.0, 0.0
    for i, (images, targets) in enumerate(tqdm(loader, desc=f"Train e{epoch}", leave=False)):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        # Step scheduler first to reflect current LR for this iteration
        if scheduler is not None:
            scheduler.step(epoch - 1 + (i + 1) / len(loader))

        if args.wd_schedule == "cosine":
            # Scale weight decay in lockstep with current LR (after scheduler.step)
            for group in optimizer.param_groups:
                if "initial_wd" in group:
                    base = group["initial_wd"]
                    scale = group["lr"] / group.get("initial_lr", group["lr"])
                    group["weight_decay"] = base * float(scale)

        with torch.no_grad():
            acc1, = accuracy(logits, targets, topk=(1,))
            loss_meter += loss.item() * images.size(0)
            acc_meter += acc1 * images.size(0)

    n = len(loader.dataset)
    return loss_meter / n, acc_meter / n


@torch.no_grad()
def evaluate(model, loader, device, epoch, split="Val"):
    model.eval()
    loss_meter, acc_meter = 0.0, 0.0
    for images, targets in tqdm(loader, desc=f"{split} e{epoch}", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        acc1, = accuracy(logits, targets, topk=(1,))
        loss_meter += loss.item() * images.size(0)
        acc_meter += acc1 * images.size(0)
    n = len(loader.dataset)
    return loss_meter / n, acc_meter / n


# ----------------------------
# Main
# ----------------------------
def train_model():
    # This function will be called by wandb.agent for each sweep run
    # Initialize wandb first before accessing config
    run = wandb.init()

    # Now access parameters via wandb.config
    config = wandb.config

    # Set up arguments in a format compatible with the rest of the code
    class Args:
        pass

    args = Args()

    # Set defaults for any parameters not in the sweep config
    args.data = "./data"
    args.batch_size = 128
    args.momentum = 0.9
    args.betas = "0.9,0.999"
    args.eps = 1e-8
    args.num_workers = 4
    args.wandb = True
    args.wandb_project = "adamw-cifar10"
    args.run_name = ""

    # Apply sweep configuration values
    args.optimizer = config.optimizer
    args.weight_decay = config.weight_decay
    args.seed = config.seed
    args.lr = config.lr
    args.epochs = config.epochs
    args.t0 = config.t0
    args.tmult = config.tmult
    args.wd_schedule = config.wd_schedule
    args.no_restarts = config.get('no_restarts', False)
    args.exclude_bias_norm = config.get('exclude_bias_norm', True)

    # CustomAdamW specific parameters
    args.dynamic_smoothing = config.get('dynamic_smoothing', False)
    args.min_beta1 = config.get('min_beta1', 0.5)
    args.min_beta2 = config.get('min_beta2', 0.9)
    args.global_scaling = config.get('global_scaling', False)
    args.log_betas = config.get('log_betas', False)

    # Set output directory based on sweep parameters
    if args.optimizer == "custom_adamw" and args.dynamic_smoothing:
        # Include dynamic smoothing parameters in the output directory name
        args.output = f"runs/{args.optimizer}_wd{args.weight_decay}_s{args.seed}_minb1{args.min_beta1}_minb2{args.min_beta2}"
        if args.global_scaling:
            args.output += "_global"
    else:
        args.output = f"runs/{args.optimizer}_wd{args.weight_decay}_s{args.seed}"


    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_ds = datasets.CIFAR10(args.data, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(args.data, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    model = make_cifar_resnet18(num_classes=10).to(device)

    # Optimizer & param groups
    groups = param_groups_weight_decay(model, args.weight_decay, exclude_bias_norm=args.exclude_bias_norm)
    betas = tuple(float(x) for x in args.betas.split(","))
    if args.optimizer == "adamw":
        opt = AdamW(groups, lr=args.lr, betas=betas, eps=args.eps)
    elif args.optimizer == "custom_adamw":
        # Use our custom implementation with dynamic smoothing
        opt = CustomAdamW(groups, lr=args.lr, betas=betas, eps=args.eps,
                         weight_decay=args.weight_decay,
                         dynamic_smoothing=args.dynamic_smoothing,
                         min_beta1=args.min_beta1, min_beta2=args.min_beta2,
                         global_scaling=args.global_scaling,
                         log_betas=args.log_betas)
    elif args.optimizer == "adam":
        # Note: this is *coupled* L2 regularization (i.e., classic Adam + L2)
        opt = Adam(groups, lr=args.lr, betas=betas, eps=args.eps)
    elif args.optimizer == "sgdw":
        opt = SGDW(groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.optimizer)

    # Record initial lr/wd for optional cosine WD schedule
    for g in opt.param_groups:
        g["initial_lr"] = args.lr
        g["initial_wd"] = g.get("weight_decay", 0.0)

    # Scheduler: Cosine warm restarts or single long cosine
    if args.no_restarts:
        # emulate "one long cosine": use T_0=epochs and no restarts (T_mult ignored)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.t0, T_mult=args.tmult)

    # Logging
    log_path = outdir / "log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"])

    best_acc = 0.0
    best_path = outdir / "best.pth"

    # Train
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, scheduler, device, epoch, args)
        test_loss, test_acc   = evaluate(model, test_loader, device, epoch, split="Test")

        # Log
        lr_now = opt.param_groups[0]["lr"]
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.3f}", f"{test_loss:.6f}", f"{test_acc:.3f}", f"{lr_now:.6e}"])

        if args.wandb:
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": lr_now,
            }
            wandb.log(log_data, step=epoch)

            # Log beta statistics if we're using CustomAdamW with logging enabled
            if args.optimizer == "custom_adamw" and args.log_betas and hasattr(opt, 'beta_stats'):
                # Access beta_stats as an attribute, not a method
                beta_stats = opt.beta_stats
                if beta_stats:
                    for key, value in beta_stats.items():
                        # If value is a list, log the last value
                        if isinstance(value, list) and value:
                            wandb.log({key: value[-1]}, step=epoch)
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "test_acc": test_acc,
                        "args": vars(args)}, best_path)

        dt = time.time() - t0
        print(f"[e{epoch:04d}/{args.epochs}] "
              f"train {train_loss:.4f}/{train_acc:.2f}% | "
              f"test {test_loss:.4f}/{test_acc:.2f}% | "
              f"lr {lr_now:.3e} | best {best_acc:.2f}% | {dt:.1f}s")

    print(f"Done. Best test acc: {best_acc:.2f}%  (ckpt: {best_path})")
    if args.wandb:
        wandb.run.summary["best_test_acc"] = best_acc

    return best_acc  # Return the best accuracy for the sweep to track

if __name__ == "__main__":
    # Define number of epochs for both training length and scheduler period
    num_epochs = 10

    # Define sweep configuration
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "test_acc",
            "goal": "maximize"
        },
        "parameters": {
            "optimizer": {
                "values": ["custom_adamw", "adamw"]
            },
            "weight_decay": {
                "values": [0.01, 0.025]
            },
            "seed": {
                "values": [0, 1]
            },
            # CustomAdamW specific parameters
            "dynamic_smoothing": {
                "value": True
            },
            "min_beta1": {
                "values": [0.5, 0.7]
            },
            "min_beta2": {
                "value": 0.9
            },
            "global_scaling": {
                "value": False
            },
            "log_betas": {
                "value": True
            },
            # Common parameters
            "lr": {
                "value": 0.001
            },
            "epochs": {
                "value": num_epochs
            },
            "t0": {
                "value": num_epochs
            },
            "tmult": {
                "value": 2
            },
            "wd_schedule": {
                "value": "constant"
            },
            "exclude_bias_norm": {
                "value": True
            },
            "no_restarts": {
                "value": False
            }
        }
    }

    # Initialize sweep
    project_name = "adamw-cifar10"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created sweep with ID: {sweep_id}")

    # Run agent
    # Grid size: 2 (opt) × 2 (wd) × 2 (seed) × 2 (min_beta1) = 16
    wandb.agent(sweep_id, function=train_model, count=16)
