import math
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# =========================================================
# 1. Basic utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_loglog_slope(x, y, tail_ratio=0.3):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 1e-16)

    start = int(len(x) * (1.0 - tail_ratio))
    xx = np.log(x[start:])
    yy = np.log(y[start:])

    A = np.vstack([xx, np.ones_like(xx)]).T
    slope, intercept = np.linalg.lstsq(A, yy, rcond=None)[0]
    alpha = -slope
    return alpha, slope, intercept


def moving_min_gap(values: List[float]) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    best = np.minimum.accumulate(vals)
    return np.maximum(vals - best, 1e-16)


# =========================================================
# 2. Dataset: MNIST binary classification
# =========================================================

def make_mnist_binary(
    class_pos=3,
    class_neg=8,
    train_limit=4000,
    test_limit=1000,
    batch_size=128,
    root="./data",
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    def filter_binary(ds):
        xs = []
        ys = []
        for x, y in ds:
            if y == class_pos:
                xs.append(x)
                ys.append(1.0)
            elif y == class_neg:
                xs.append(x)
                ys.append(-1.0)
        X = torch.stack(xs, dim=0)
        Y = torch.tensor(ys, dtype=torch.float32)
        return X, Y

    Xtr, Ytr = filter_binary(train_set)
    Xte, Yte = filter_binary(test_set)

    if train_limit is not None:
        Xtr = Xtr[:train_limit]
        Ytr = Ytr[:train_limit]

    if test_limit is not None:
        Xte = Xte[:test_limit]
        Yte = Yte[:test_limit]

    train_loader = DataLoader(
        TensorDataset(Xtr, Ytr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        TensorDataset(Xte, Yte),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, Xtr, Ytr, Xte, Yte


# =========================================================
# 3. Model
# =========================================================

class SmallMLP(nn.Module):
    def __init__(self, in_dim=784, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================================================
# 4. Parameter vector helpers
# =========================================================

def get_param_vector(model: nn.Module) -> torch.Tensor:
    return nn.utils.parameters_to_vector([p.detach() for p in model.parameters()])


def set_param_vector_(model: nn.Module, vec: torch.Tensor):
    nn.utils.vector_to_parameters(vec, model.parameters())


def l2_sq(vec: torch.Tensor) -> torch.Tensor:
    return torch.sum(vec * vec)


def l1_norm(vec: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(vec))


def soft_threshold(vec: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.sign(vec) * torch.clamp(torch.abs(vec) - thresh, min=0.0)


# =========================================================
# 5. Loss definitions
# =========================================================

def logistic_loss_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-y * logits)).mean()


def hinge_loss_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0 - y * logits, min=0.0).mean()


def classification_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.where(logits >= 0, 1.0, -1.0)
    return (pred == y).float().mean().item()


# =========================================================
# 6. Full-batch evaluation
# =========================================================

@torch.no_grad()
def evaluate_objective_and_acc(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    objective_type: str,
    reg_strength: float,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    logits = model(X.to(device))
    theta = get_param_vector(model).to(device)

    if objective_type == "smooth":
        loss = logistic_loss_from_logits(logits, Y.to(device)) + 0.5 * reg_strength * l2_sq(theta)
    elif objective_type == "nonsmooth":
        loss = hinge_loss_from_logits(logits, Y.to(device)) + reg_strength * l1_norm(theta)
    else:
        raise ValueError("objective_type must be smooth or nonsmooth")

    acc = classification_accuracy(logits, Y.to(device))
    return loss.item(), acc


def full_batch_smooth_grad_norm_sq(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    mu: float,
    device: torch.device,
) -> float:
    model.zero_grad(set_to_none=True)
    model.train()

    logits = model(X.to(device))
    theta = nn.utils.parameters_to_vector([p for p in model.parameters()])
    loss = logistic_loss_from_logits(logits, Y.to(device)) + 0.5 * mu * l2_sq(theta)
    loss.backward()

    grad_vec = nn.utils.parameters_to_vector(
        [
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ]
    )

    return torch.sum(grad_vec * grad_vec).item()


def full_batch_nonsmooth_prox_grad_mapping_sq(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    lam: float,
    eta: float,
    device: torch.device,
) -> float:
    model.zero_grad(set_to_none=True)
    model.train()

    logits = model(X.to(device))
    loss_fit = hinge_loss_from_logits(logits, Y.to(device))
    loss_fit.backward()

    theta = get_param_vector(model).to(device)
    grad_vec = nn.utils.parameters_to_vector(
        [
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ]
    ).detach()

    prox_arg = theta - eta * grad_vec
    prox_out = soft_threshold(prox_arg, eta * lam)
    gmap = (theta - prox_out) / eta

    return torch.sum(gmap * gmap).item()


# =========================================================
# 7. ZO two-point estimators on parameter vector
# =========================================================

@torch.no_grad()
def set_model_from_vec_and_eval_minibatch(
    model: nn.Module,
    vec: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    objective_type: str,
    reg_strength: float,
    device: torch.device,
) -> float:
    old_vec = get_param_vector(model).detach().clone().to(device)

    set_param_vector_(model, vec)

    logits = model(x.to(device))
    theta = get_param_vector(model).to(device)

    if objective_type == "smooth":
        val = logistic_loss_from_logits(logits, y.to(device)) + 0.5 * reg_strength * l2_sq(theta)
    elif objective_type == "nonsmooth_fit_only":
        val = hinge_loss_from_logits(logits, y.to(device))
    else:
        raise ValueError("unknown objective_type")

    out = val.item()

    set_param_vector_(model, old_vec)

    return out


@torch.no_grad()
def zo_two_point_estimator(
    model: nn.Module,
    w: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    nu: float,
    q: int,
    objective_type: str,
    reg_strength: float,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    d = w.numel()
    g_acc = torch.zeros_like(w)

    for _ in range(q):
        u = torch.randn_like(w)
        u = u / (torch.norm(u) + 1e-12)

        f_plus = set_model_from_vec_and_eval_minibatch(
            model=model,
            vec=w + nu * u,
            x=x,
            y=y,
            objective_type=objective_type,
            reg_strength=reg_strength,
            device=device,
        )

        f_minus = set_model_from_vec_and_eval_minibatch(
            model=model,
            vec=w - nu * u,
            x=x,
            y=y,
            objective_type=objective_type,
            reg_strength=reg_strength,
            device=device,
        )

        g_hat = (d / (2.0 * nu)) * (f_plus - f_minus) * u
        g_acc += g_hat

    return g_acc / q, 2 * q


# =========================================================
# 8. Training configs
# =========================================================

@dataclass
class SmoothConfig:
    beta: float = 0.1
    eta: float = 0.02
    gamma: float = 0.0
    nu: float = 0.01
    K: int = 20
    J: int = 60
    q: int = 5
    mu: float = 1e-4
    eval_every_epoch: bool = True


@dataclass
class NonsmoothConfig:
    beta: float = 0.1
    eta: float = 0.01
    gamma: float = 0.0
    nu: float = 0.01
    K: int = 20
    J: int = 60
    q: int = 5
    lam: float = 1e-5
    eval_every_epoch: bool = True


# =========================================================
# 9. Main training loops
# =========================================================

def run_zo_sgdm_smooth(
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xte: torch.Tensor,
    Yte: torch.Tensor,
    cfg: SmoothConfig,
    device: torch.device,
) -> Dict:
    set_seed(seed)

    model = SmallMLP().to(device)
    model.train()

    train_iter = iter(train_loader)
    w_epoch = get_param_vector(model).detach().clone().to(device)

    total_queries = 0

    queries_log = []
    train_obj_log = []
    test_obj_log = []
    train_acc_log = []
    test_acc_log = []
    stat_log = []

    for j in range(cfg.J):
        w = w_epoch.clone()
        w_prev = w.clone()
        m = torch.zeros_like(w)
        traj = []

        for k in range(cfg.K):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)

            v = w + cfg.gamma * (w - w_prev)

            g_hat, q_used = zo_two_point_estimator(
                model=model,
                w=v,
                x=xb,
                y=yb,
                nu=cfg.nu,
                q=cfg.q,
                objective_type="smooth",
                reg_strength=cfg.mu,
                device=device,
            )

            total_queries += q_used

            m = (1.0 - cfg.beta) * m + cfg.beta * g_hat
            w_next = w - cfg.eta * m

            w_prev = w.clone()
            w = w_next

            traj.append(w.clone())

        w_bar = torch.stack(traj, dim=0).mean(dim=0)
        w_epoch = w_bar.clone()

        set_param_vector_(model, w_epoch)

        tr_obj, tr_acc = evaluate_objective_and_acc(
            model=model,
            X=Xtr,
            Y=Ytr,
            objective_type="smooth",
            reg_strength=cfg.mu,
            device=device,
        )

        te_obj, te_acc = evaluate_objective_and_acc(
            model=model,
            X=Xte,
            Y=Yte,
            objective_type="smooth",
            reg_strength=cfg.mu,
            device=device,
        )

        stat = full_batch_smooth_grad_norm_sq(
            model=model,
            X=Xtr,
            Y=Ytr,
            mu=cfg.mu,
            device=device,
        )

        queries_log.append(total_queries)
        train_obj_log.append(tr_obj)
        test_obj_log.append(te_obj)
        train_acc_log.append(tr_acc)
        test_acc_log.append(te_acc)
        stat_log.append(stat)

        print(
            f"[smooth][seed {seed}] epoch {j + 1:03d}/{cfg.J} | "
            f"queries={total_queries:6d} | "
            f"train_obj={tr_obj:.6f} | "
            f"test_obj={te_obj:.6f} | "
            f"train_acc={100.0 * tr_acc:.2f}% | "
            f"test_acc={100.0 * te_acc:.2f}% | "
            f"stat={stat:.6e}"
        )

    return {
        "queries": np.array(queries_log),
        "train_obj": np.array(train_obj_log),
        "test_obj": np.array(test_obj_log),
        "train_acc": np.array(train_acc_log),
        "test_acc": np.array(test_acc_log),
        "stat": np.array(stat_log),
    }


def run_zo_proxsgdm_nonsmooth(
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xte: torch.Tensor,
    Yte: torch.Tensor,
    cfg: NonsmoothConfig,
    device: torch.device,
) -> Dict:
    set_seed(seed)

    model = SmallMLP().to(device)
    model.train()

    w_epoch = get_param_vector(model).detach().clone().to(device)

    # 保留你原来的 nonsmooth nonzero init
    w_epoch = w_epoch + 0.01 * torch.randn_like(w_epoch)

    set_param_vector_(model, w_epoch)

    train_iter = iter(train_loader)

    total_queries = 0

    queries_log = []
    train_obj_log = []
    test_obj_log = []
    train_acc_log = []
    test_acc_log = []
    stat_log = []
    zero_ratio_log = []

    for j in range(cfg.J):
        w = w_epoch.clone()
        w_prev = w.clone()
        m = torch.zeros_like(w)
        traj = []

        for k in range(cfg.K):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)

            v = w + cfg.gamma * (w - w_prev)

            g_hat, q_used = zo_two_point_estimator(
                model=model,
                w=v,
                x=xb,
                y=yb,
                nu=cfg.nu,
                q=cfg.q,
                objective_type="nonsmooth_fit_only",
                reg_strength=0.0,
                device=device,
            )

            total_queries += q_used

            m = (1.0 - cfg.beta) * m + cfg.beta * g_hat
            u = w - cfg.eta * m
            w_next = soft_threshold(u, cfg.eta * cfg.lam)

            w_prev = w.clone()
            w = w_next

            traj.append(w.clone())

        w_bar = torch.stack(traj, dim=0).mean(dim=0)
        w_epoch = w_bar.clone()

        set_param_vector_(model, w_epoch)

        tr_obj, tr_acc = evaluate_objective_and_acc(
            model=model,
            X=Xtr,
            Y=Ytr,
            objective_type="nonsmooth",
            reg_strength=cfg.lam,
            device=device,
        )

        te_obj, te_acc = evaluate_objective_and_acc(
            model=model,
            X=Xte,
            Y=Yte,
            objective_type="nonsmooth",
            reg_strength=cfg.lam,
            device=device,
        )

        stat = full_batch_nonsmooth_prox_grad_mapping_sq(
            model=model,
            X=Xtr,
            Y=Ytr,
            lam=cfg.lam,
            eta=cfg.eta,
            device=device,
        )

        zero_ratio = (torch.abs(w_epoch) < 1e-12).float().mean().item()

        queries_log.append(total_queries)
        train_obj_log.append(tr_obj)
        test_obj_log.append(te_obj)
        train_acc_log.append(tr_acc)
        test_acc_log.append(te_acc)
        stat_log.append(stat)
        zero_ratio_log.append(zero_ratio)

        print(
            f"[nonsmooth][seed {seed}] epoch {j + 1:03d}/{cfg.J} | "
            f"queries={total_queries:6d} | "
            f"train_obj={tr_obj:.6f} | "
            f"test_obj={te_obj:.6f} | "
            f"train_acc={100.0 * tr_acc:.2f}% | "
            f"test_acc={100.0 * te_acc:.2f}% | "
            f"stat={stat:.6e} | "
            f"zero_ratio={zero_ratio:.3f}"
        )

    return {
        "queries": np.array(queries_log),
        "train_obj": np.array(train_obj_log),
        "test_obj": np.array(test_obj_log),
        "train_acc": np.array(train_acc_log),
        "test_acc": np.array(test_acc_log),
        "stat": np.array(stat_log),
        "zero_ratio": np.array(zero_ratio_log),
    }


# =========================================================
# 10. Aggregate and plotting
# =========================================================

def aggregate_runs(run_list: List[Dict], key: str):
    arr = np.stack([r[key] for r in run_list], axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


# ---------- Font-size settings for figures ----------
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 14
TICK_FONTSIZE = 13


def plot_mean_std_linear(x, mean, std, label):
    lower = mean - std
    upper = mean + std

    plt.plot(x, mean, marker="o", label=label)
    plt.fill_between(x, lower, upper, alpha=0.2)


def plot_mean_std_loglog(x, mean, std, label):
    lower = np.maximum(mean - std, 1e-16)
    upper = np.maximum(mean + std, 1e-16)

    plt.loglog(x, mean, marker="o", label=label)
    plt.fill_between(x, lower, upper, alpha=0.2)


def format_current_axes():
    """
    Only enlarge axis tick labels.
    Other plotting styles are unchanged.
    """
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)


def save_results_pickle(
    smooth_runs,
    nonsmooth_runs,
    smooth_cfg,
    nonsmooth_cfg,
    seeds,
    filename="zo_sgdm_results.pkl",
):
    with open(filename, "wb") as f:
        pickle.dump(
            {
                "smooth_runs": smooth_runs,
                "nonsmooth_runs": nonsmooth_runs,
                "smooth_cfg": smooth_cfg,
                "nonsmooth_cfg": nonsmooth_cfg,
                "seeds": seeds,
            },
            f,
        )

    print(f"\nSaved training results to {filename}")


def plot_final_loss_acc_curves(
    smooth_runs,
    nonsmooth_runs,
    save_fig=True,
    show_fig=True,
):
    q_s = smooth_runs[0]["queries"]
    q_n = nonsmooth_runs[0]["queries"]

    # train loss / objective
    s_train_obj_mean, s_train_obj_std = aggregate_runs(smooth_runs, "train_obj")
    n_train_obj_mean, n_train_obj_std = aggregate_runs(nonsmooth_runs, "train_obj")

    # test loss / objective
    s_test_obj_mean, s_test_obj_std = aggregate_runs(smooth_runs, "test_obj")
    n_test_obj_mean, n_test_obj_std = aggregate_runs(nonsmooth_runs, "test_obj")

    # train acc
    s_train_acc_mean, s_train_acc_std = aggregate_runs(smooth_runs, "train_acc")
    n_train_acc_mean, n_train_acc_std = aggregate_runs(nonsmooth_runs, "train_acc")

    # test acc
    s_test_acc_mean, s_test_acc_std = aggregate_runs(smooth_runs, "test_acc")
    n_test_acc_mean, n_test_acc_std = aggregate_runs(nonsmooth_runs, "test_acc")

    # stationarity
    s_stat_mean, s_stat_std = aggregate_runs(smooth_runs, "stat")
    n_stat_mean, n_stat_std = aggregate_runs(nonsmooth_runs, "stat")

    # slope fitting
    s_gap = moving_min_gap(s_train_obj_mean.tolist())
    n_gap = moving_min_gap(n_train_obj_mean.tolist())

    alpha_s_gap, _, _ = fit_loglog_slope(q_s, s_gap, tail_ratio=0.4)
    alpha_n_gap, _, _ = fit_loglog_slope(q_n, n_gap, tail_ratio=0.4)

    alpha_s_stat, _, _ = fit_loglog_slope(q_s, s_stat_mean, tail_ratio=0.4)
    alpha_n_stat, _, _ = fit_loglog_slope(q_n, n_stat_mean, tail_ratio=0.4)

    print("\n========== tail slope fit ==========")
    print(f"[Smooth]    surrogate gap alpha ≈ {alpha_s_gap:.4f}")
    print(f"[Nonsmooth] surrogate gap alpha ≈ {alpha_n_gap:.4f}")
    print(f"[Smooth]    stationarity alpha ≈ {alpha_s_stat:.4f}")
    print(f"[Nonsmooth] stationarity alpha ≈ {alpha_n_stat:.4f}")

    # -----------------------------
    # Figure 1: train loss
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plot_mean_std_linear(
        q_s,
        s_train_obj_mean,
        s_train_obj_std,
        "smooth: train loss",
    )
    plot_mean_std_linear(
        q_n,
        n_train_obj_mean,
        n_train_obj_std,
        "nonsmooth: train loss",
    )
    plt.xlabel("Number of queries", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Train loss / objective", fontsize=LABEL_FONTSIZE)
    plt.title("Train Loss Curve", fontsize=TITLE_FONTSIZE)
    format_current_axes()
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    if save_fig:
        plt.savefig("train_loss_curve.png", dpi=300)

    # -----------------------------
    # Figure 2: test loss
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plot_mean_std_linear(
        q_s,
        s_test_obj_mean,
        s_test_obj_std,
        "smooth: test loss",
    )
    plot_mean_std_linear(
        q_n,
        n_test_obj_mean,
        n_test_obj_std,
        "nonsmooth: test loss",
    )
    plt.xlabel("Number of queries", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Test loss / objective", fontsize=LABEL_FONTSIZE)
    plt.title("Test Loss Curve", fontsize=TITLE_FONTSIZE)
    format_current_axes()
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    if save_fig:
        plt.savefig("test_loss_curve.png", dpi=300)

    # -----------------------------
    # Figure 3: train accuracy
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plot_mean_std_linear(
        q_s,
        100.0 * s_train_acc_mean,
        100.0 * s_train_acc_std,
        "smooth: train acc",
    )
    plot_mean_std_linear(
        q_n,
        100.0 * n_train_acc_mean,
        100.0 * n_train_acc_std,
        "nonsmooth: train acc",
    )
    plt.xlabel("Number of queries", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Train accuracy (%)", fontsize=LABEL_FONTSIZE)
    plt.title("Train Accuracy Curve", fontsize=TITLE_FONTSIZE)
    format_current_axes()
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    if save_fig:
        plt.savefig("train_acc_curve.png", dpi=300)

    # -----------------------------
    # Figure 4: test accuracy
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plot_mean_std_linear(
        q_s,
        100.0 * s_test_acc_mean,
        100.0 * s_test_acc_std,
        "smooth: test acc",
    )
    plot_mean_std_linear(
        q_n,
        100.0 * n_test_acc_mean,
        100.0 * n_test_acc_std,
        "nonsmooth: test acc",
    )
    plt.xlabel("Number of queries", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Test accuracy (%)", fontsize=LABEL_FONTSIZE)
    plt.title("Test Accuracy Curve", fontsize=TITLE_FONTSIZE)
    format_current_axes()
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    if save_fig:
        plt.savefig("test_acc_curve.png", dpi=300)

    # -----------------------------
    # Figure 5: stationarity
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plot_mean_std_loglog(
        q_s,
        s_stat_mean,
        s_stat_std,
        f"smooth stat, alpha={alpha_s_stat:.3f}",
    )
    plot_mean_std_loglog(
        q_n,
        n_stat_mean,
        n_stat_std,
        f"nonsmooth stat, alpha={alpha_n_stat:.3f}",
    )
    plt.xlabel("Number of queries", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Stationarity surrogate", fontsize=LABEL_FONTSIZE)
    plt.title("Log-log Stationarity Surrogate", fontsize=TITLE_FONTSIZE)
    format_current_axes()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    if save_fig:
        plt.savefig("stationarity_curve.png", dpi=300)

    if show_fig:
        print("\nAll figures are ready. Closing the plot windows will terminate the script.")
        plt.show()


# =========================================================
# 11. Main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # -----------------------------
    # data
    # -----------------------------
    train_loader, test_loader, Xtr, Ytr, Xte, Yte = make_mnist_binary(
        class_pos=3,
        class_neg=8,
        train_limit=4000,
        test_limit=1000,
        batch_size=128,
        root="./data",
    )

    # -----------------------------
    # configs
    # -----------------------------
    smooth_cfg = SmoothConfig(
        beta=0.1,
        eta=0.02,
        gamma=0.0,
        nu=0.01,
        K=20,
        J=50,
        q=1,
        mu=1e-4,
    )

    nonsmooth_cfg = NonsmoothConfig(
        beta=0.1,
        eta=0.01,
        gamma=0.0,
        nu=0.01,
        K=20,
        J=50,
        q=1,
        lam=1e-5,
    )

    seeds = [0, 1, 2]

    smooth_runs = []
    nonsmooth_runs = []

    # -----------------------------
    # run smooth
    # -----------------------------
    for seed in seeds:
        out = run_zo_sgdm_smooth(
            seed=seed,
            train_loader=train_loader,
            test_loader=test_loader,
            Xtr=Xtr,
            Ytr=Ytr,
            Xte=Xte,
            Yte=Yte,
            cfg=smooth_cfg,
            device=device,
        )
        smooth_runs.append(out)

    # -----------------------------
    # run nonsmooth
    # -----------------------------
    for seed in seeds:
        out = run_zo_proxsgdm_nonsmooth(
            seed=seed,
            train_loader=train_loader,
            test_loader=test_loader,
            Xtr=Xtr,
            Ytr=Ytr,
            Xte=Xte,
            Yte=Yte,
            cfg=nonsmooth_cfg,
            device=device,
        )
        nonsmooth_runs.append(out)

    # -----------------------------
    # save results
    # -----------------------------
    save_results_pickle(
        smooth_runs=smooth_runs,
        nonsmooth_runs=nonsmooth_runs,
        smooth_cfg=smooth_cfg,
        nonsmooth_cfg=nonsmooth_cfg,
        seeds=seeds,
        filename="zo_sgdm_results.pkl",
    )

    # -----------------------------
    # plot final figures after all training
    # -----------------------------
    print("\nAll training finished. Plotting final curves...")

    plot_final_loss_acc_curves(
        smooth_runs=smooth_runs,
        nonsmooth_runs=nonsmooth_runs,
        save_fig=True,
        show_fig=True,
    )


if __name__ == "__main__":
    main()