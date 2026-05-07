import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import copy
import random
import contextlib
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------
# GroupNorm utilities
# ----------------------------
def _largest_divisor_leq(n: int, k: int) -> int:
    for d in range(min(k, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def make_gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    g = _largest_divisor_leq(num_channels, max_groups)
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


def replace_bn_with_gn(module: nn.Module, max_groups: int = 32) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            setattr(module, name, make_gn(child.num_features, max_groups=max_groups))
        else:
            replace_bn_with_gn(child, max_groups=max_groups)
    return module


# ----------------------------
# Model: ResNet18 adapted for CIFAR-10 + GN
# ----------------------------
def make_resnet18_cifar10_gn(num_classes=10, gn_groups=32):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    replace_bn_with_gn(m, max_groups=gn_groups)
    return m


# ----------------------------
# Data
# ----------------------------
def make_cifar10_loaders(data_dir, batch_size, num_workers, seed, no_aug=False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if no_aug:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, test_loader


# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def eval_model(model, loader, device, max_batches=None):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0.0, 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += y.numel()

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


# ----------------------------
# Flat-param helpers
# ----------------------------
def select_params_full(model: nn.Module):
    return list(model.parameters())


@torch.no_grad()
def flatten_params(params):
    return torch.cat([p.data.flatten() for p in params], dim=0)


@torch.no_grad()
def assign_params_from_flat(params, flat_vec):
    offset = 0
    for p in params:
        num = p.numel()
        p.data.copy_(flat_vec[offset:offset + num].view_as(p))
        offset += num


def freeze_all_grads(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def clip_update(update_vec, max_norm):
    if max_norm is None or max_norm <= 0:
        return update_vec, False

    n = update_vec.norm() + 1e-12

    if n > max_norm:
        update_vec = update_vec * (max_norm / n)
        return update_vec, True

    return update_vec, False


def _autocast_ctx(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


@torch.no_grad()
def sample_gaussian(d, device, generator=None):
    return torch.randn(d, device=device, generator=generator)


# ----------------------------
# ZO estimator: Gaussian antithetic two-point
# ----------------------------
def zo_two_point_grad_gaussian_unbiased(
    model,
    params,
    theta,
    x,
    y,
    mu,
    q,
    device,
    generator=None,
    amp=False,
):
    model.eval()
    g = torch.zeros_like(theta)
    d = theta.numel()

    for _ in range(q):
        u = sample_gaussian(d, device=device, generator=generator)

        theta_p = theta + mu * u
        assign_params_from_flat(params, theta_p)

        with _autocast_ctx(device, amp):
            loss_p = F.cross_entropy(model(x), y)

        theta_m = theta - mu * u
        assign_params_from_flat(params, theta_m)

        with _autocast_ctx(device, amp):
            loss_m = F.cross_entropy(model(x), y)

        g += ((loss_p.float() - loss_m.float()) / (2.0 * mu)) * u

    assign_params_from_flat(params, theta)

    return g / float(q)


# ----------------------------
# One run: one seed, one algorithm
# ----------------------------
def run_one_seed_one_algo(init_state, seed, algo, device, args):
    set_seed(seed)

    train_loader, test_loader = make_cifar10_loaders(
        args.data,
        args.batch_size,
        args.num_workers,
        seed=seed,
        no_aug=args.no_aug,
    )

    model = make_resnet18_cifar10_gn(
        num_classes=10,
        gn_groups=args.gn_groups,
    ).to(device)

    model.load_state_dict(init_state)
    freeze_all_grads(model)

    params = select_params_full(model)
    theta = flatten_params(params).to(device)

    # momentum state
    m = torch.zeros_like(theta)

    # Same direction RNG stream for fairness
    gen = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    gen.manual_seed(seed + args.dir_seed_offset)

    train_iter = iter(train_loader)

    iters = []

    train_loss_list = []
    test_loss_list = []

    train_acc_list = []
    test_acc_list = []

    clip_hits, clip_cnt = 0, 0
    t0 = time.time()

    for step in range(1, args.steps_total + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Periodic reset only for zo_sgdm_pr
        if algo == "zo_sgdm_pr" and args.reset_every > 0:
            if (step - 1) % args.reset_every == 0:
                m.zero_()

        hit = False
        upd_raw = None

        if algo == "zo_sgd":
            ghat = zo_two_point_grad_gaussian_unbiased(
                model,
                params,
                theta,
                x,
                y,
                mu=args.mu,
                q=args.q,
                device=device,
                generator=gen,
                amp=args.amp,
            )

            upd_raw = args.lr_sgd * ghat
            upd, hit = clip_update(upd_raw, args.max_update_norm)
            theta = theta - upd
            assign_params_from_flat(params, theta)

        elif algo == "zo_sgdm":
            ghat = zo_two_point_grad_gaussian_unbiased(
                model,
                params,
                theta,
                x,
                y,
                mu=args.mu,
                q=args.q,
                device=device,
                generator=gen,
                amp=args.amp,
            )

            m = (1.0 - args.beta_sgdm) * m + args.beta_sgdm * ghat
            upd_raw = args.eta_sgdm * m
            upd, hit = clip_update(upd_raw, args.max_update_norm)
            theta = theta - upd
            assign_params_from_flat(params, theta)

        elif algo == "zo_sgdm_pr":
            ghat = zo_two_point_grad_gaussian_unbiased(
                model,
                params,
                theta,
                x,
                y,
                mu=args.mu,
                q=args.q,
                device=device,
                generator=gen,
                amp=args.amp,
            )

            m = (1.0 - args.beta_pr) * m + args.beta_pr * ghat
            upd_raw = args.eta_pr * m
            upd, hit = clip_update(upd_raw, args.max_update_norm)
            theta = theta - upd
            assign_params_from_flat(params, theta)

        else:
            raise ValueError(f"Unknown algo: {algo}")

        clip_cnt += 1
        clip_hits += int(hit)

        if step % args.eval_every == 0 or step == 1:
            # Evaluate both train and test.
            tr_loss, tr_acc = eval_model(
                model,
                train_loader,
                device,
                max_batches=args.train_eval_batches,
            )

            te_loss, te_acc = eval_model(
                model,
                test_loader,
                device,
                max_batches=args.eval_batches,
            )

            iters.append(step)

            train_loss_list.append(tr_loss)
            test_loss_list.append(te_loss)

            train_acc_list.append(tr_acc)
            test_acc_list.append(te_acc)

            clip_rate = clip_hits / max(1, clip_cnt)
            unorm = float(upd_raw.norm().item()) if upd_raw is not None else float("nan")

            print(
                f"[{algo.upper()}][seed {seed}][step {step:06d}/{args.steps_total}] "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"test loss {te_loss:.4f} acc {te_acc:.4f} | "
                f"||upd_raw|| {unorm:.2e} | clip {clip_rate:.1%} | "
                f"{time.time() - t0:.1f}s"
            )

            clip_hits, clip_cnt = 0, 0

    return {
        "iters": np.array(iters),
        "train_loss": np.array(train_loss_list),
        "test_loss": np.array(test_loss_list),
        "train_acc": np.array(train_acc_list),
        "test_acc": np.array(test_acc_list),
    }


# ----------------------------
# Multi-seed stats + plots
# ----------------------------
def stack_and_stats(curves_list):
    X = np.stack(curves_list, axis=0)
    return X.mean(axis=0), X.std(axis=0)


def plot_mean_std(
    iters,
    curves,
    title,
    ylabel,
    save_path=None,
    xlabel="Iteration",
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    linewidth=2.2,
    marker_size=4,
    marker_every=20,
):
    display_names = {
        "zo_sgd": "ZO-SGD",
        "zo_sgdm": "ZO-SGDM",
        "zo_sgdm_pr": "ZO-SGDM-PR",
    }

    plt.figure(figsize=(7.2, 5.0))

    for name, (mean, std) in curves.items():
        plt.plot(
            iters,
            mean,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            markevery=marker_every,
            label=display_names.get(name, name),
        )
        plt.fill_between(
            iters,
            mean - std,
            mean + std,
            alpha=0.18,
        )

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=legend_fontsize, frameon=True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    ap.add_argument("--steps_total", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=50)

    # test evaluation batches
    ap.add_argument("--eval_batches", type=int, default=50)

    # train evaluation batches
    # 建议不要默认全量 train，否则 CIFAR-10 上会明显变慢
    ap.add_argument("--train_eval_batches", type=int, default=50)

    # GN
    ap.add_argument("--gn_groups", type=int, default=32)

    # Gaussian ZO estimator
    ap.add_argument("--mu", type=float, default=1e-4)
    ap.add_argument("--q", type=int, default=8)

    # clip
    ap.add_argument("--max_update_norm", type=float, default=1.0)

    # direction RNG seed offset
    ap.add_argument("--dir_seed_offset", type=int, default=777)

    # ZO-SGD
    ap.add_argument("--lr_sgd", type=float, default=3e-4)

    # ZO-SGDM
    ap.add_argument("--eta_sgdm", type=float, default=3e-4)
    ap.add_argument("--beta_sgdm", type=float, default=0.1)

    # ZO-SGDM with periodic reset
    ap.add_argument("--eta_pr", type=float, default=3e-4)
    ap.add_argument("--beta_pr", type=float, default=0.1)
    ap.add_argument("--reset_every", type=int, default=1000)

    # save
    ap.add_argument("--save_prefix", type=str, default="cifar10_zo_resnet18")

    args = ap.parse_args()

    if args.eval_batches is not None and args.eval_batches < 0:
        args.eval_batches = None

    if args.train_eval_batches is not None and args.train_eval_batches < 0:
        args.train_eval_batches = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Fixed init across all seeds and algos
    set_seed(0)
    model0 = make_resnet18_cifar10_gn(
        num_classes=10,
        gn_groups=args.gn_groups,
    ).to(device)

    init_state = copy.deepcopy(model0.state_dict())

    algos = ["zo_sgd", "zo_sgdm", "zo_sgdm_pr"]

    all_train_losses = {a: [] for a in algos}
    all_test_losses = {a: [] for a in algos}

    all_train_accs = {a: [] for a in algos}
    all_test_accs = {a: [] for a in algos}

    iters_ref = None

    raw_results = {a: [] for a in algos}

    for s in args.seeds:
        for algo in algos:
            print("\n==============================")
            print(f"Seed {s}: {algo.upper()}")
            print("==============================")

            out = run_one_seed_one_algo(
                init_state=init_state,
                seed=s,
                algo=algo,
                device=device,
                args=args,
            )

            iters = out["iters"]

            if iters_ref is None:
                iters_ref = iters
            else:
                if len(iters_ref) != len(iters) or np.any(iters_ref != iters):
                    raise RuntimeError("Evaluation grid mismatch across runs.")

            all_train_losses[algo].append(out["train_loss"])
            all_test_losses[algo].append(out["test_loss"])

            all_train_accs[algo].append(out["train_acc"])
            all_test_accs[algo].append(out["test_acc"])

            raw_results[algo].append(out)

    curves_train_loss = {}
    curves_test_loss = {}

    curves_train_acc = {}
    curves_test_acc = {}

    for algo in algos:
        mean_tr_l, std_tr_l = stack_and_stats(all_train_losses[algo])
        mean_te_l, std_te_l = stack_and_stats(all_test_losses[algo])

        mean_tr_a, std_tr_a = stack_and_stats(all_train_accs[algo])
        mean_te_a, std_te_a = stack_and_stats(all_test_accs[algo])

        curves_train_loss[algo] = (mean_tr_l, std_tr_l)
        curves_test_loss[algo] = (mean_te_l, std_te_l)

        curves_train_acc[algo] = (mean_tr_a, std_tr_a)
        curves_test_acc[algo] = (mean_te_a, std_te_a)

    # Save raw results
    result_path = f"{args.save_prefix}_results.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(
            {
                "args": vars(args),
                "iters": iters_ref,
                "raw_results": raw_results,
                "curves_train_loss": curves_train_loss,
                "curves_test_loss": curves_test_loss,
                "curves_train_acc": curves_train_acc,
                "curves_test_acc": curves_test_acc,
            },
            f,
        )

    print(f"\nSaved raw results to {result_path}")

    plot_mean_std(
        iters_ref,
        curves_train_loss,
        title="Train Loss",
        ylabel="Cross-Entropy Loss",
        xlabel="Iteration",
        save_path="cifar10_train_loss.png",
        title_fontsize=16,
        label_fontsize=14,
        tick_fontsize=12,
        legend_fontsize=12,
        marker_every=20,
    )

    plot_mean_std(
        iters_ref,
        curves_test_loss,
        title="Test Loss",
        ylabel="Cross-Entropy Loss",
        xlabel="Iteration",
        save_path="cifar10_test_loss.png",
        title_fontsize=16,
        label_fontsize=14,
        tick_fontsize=12,
        legend_fontsize=12,
        marker_every=20,
    )

    plot_mean_std(
        iters_ref,
        curves_train_acc,
        title="Train Accuracy",
        ylabel="Accuracy",
        xlabel="Iteration",
        save_path="cifar10_train_acc.png",
        title_fontsize=16,
        label_fontsize=14,
        tick_fontsize=12,
        legend_fontsize=12,
        marker_every=20,
    )

    plot_mean_std(
        iters_ref,
        curves_test_acc,
        title="Test Accuracy",
        ylabel="Accuracy",
        xlabel="Iteration",
        save_path="cifar10_test_acc.png",
        title_fontsize=16,
        label_fontsize=14,
        tick_fontsize=12,
        legend_fontsize=12,
        marker_every=20,
    )

    plt.show()


if __name__ == "__main__":
    main()