"""Turn the saved logs into the tables and figures for the report.

    uv run python -m training.analysis
Figures go to assets/figures, tables to logs/tables.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.common import ROOT
from training.configs import SWEEPS

FIG = os.path.join(ROOT, "assets", "figures")
TAB = os.path.join(ROOT, "logs", "tables")
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOR = {"dqn": "#2563eb", "reinforce": "#dc2626", "ppo": "#16a34a", "a2c": "#d97706"}


def load_all():
    runs = {a: [] for a in ALGOS}
    for a in ALGOS:
        for f in sorted(glob.glob(os.path.join(ROOT, "logs", a, "*.json"))):
            runs[a].append(json.load(open(f)))
    return runs


def smooth(y, k=30):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    k = min(k, len(y))
    return np.convolve(y, np.ones(k) / k, mode="valid")


def best_run(rs):
    return max(rs, key=lambda r: r["final"]["mean_return"])


def changed(algo, cfg):
    base = SWEEPS[algo][0]
    diff = {k: v for k, v in cfg.items() if base.get(k) != v}
    return ", ".join(f"{k}={v}" for k, v in diff.items()) or "base setting"


def tables(runs):
    os.makedirs(TAB, exist_ok=True)
    for a, rs in runs.items():
        if not rs:
            continue
        lines = ["| Run | Setting changed | Final mean return | Std | Finished house | Weeks | Return on unseen starts |",
                 "|---|---|---|---|---|---|---|"]
        for r in rs:
            f, g = r["final"], r["generalisation"]
            lines.append(f"| {r['name'][3:]} | {changed(a, r['cfg'])} | {f['mean_return']:.1f} | {f['std_return']:.1f} | "
                         f"{f['success_rate'] * 100:.0f}% | {f['mean_weeks']:.0f} | {g['mean_return']:.1f} |")
        open(os.path.join(TAB, f"{a}_table.md"), "w").write("\n".join(lines) + "\n")


def fig_rewards(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, rs in runs.items():
        if not rs:
            continue
        r = best_run(rs)
        ep = np.array(r["episodes"])
        if len(ep):
            y = smooth(ep[:, 1], 30)
            ax[0].plot(ep[len(ep) - len(y):, 0], y, label=LABEL[a], color=COLOR[a])
        ev = np.array(r["evals"])
        if len(ev):
            ax[1].plot(ev[:, 0], ev[:, 1], marker="o", label=LABEL[a], color=COLOR[a])
    ax[0].set_title("Training episode return (moving average of 30)")
    ax[1].set_title("Greedy evaluation return on fixed starts")
    for x in ax:
        x.set_xlabel("environment steps")
        x.set_ylabel("return")
        x.grid(alpha=0.3)
        x.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "comparative_rewards.png"), dpi=150)
    plt.close(fig)


def fig_dqn_loss(runs):
    rs = runs["dqn"]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rs:
        l = np.array(r["loss"])
        if len(l):
            ax.plot(l[:, 0], smooth(l[:, 1], 20).tolist() + [np.nan] * (len(l) - len(smooth(l[:, 1], 20))), alpha=0.7, label=r["name"])
    ax.set_yscale("log")
    ax.set_title("DQN temporal difference loss, all runs")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("loss (log scale)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dqn_loss.png"), dpi=150)
    plt.close(fig)


def fig_entropy(runs):
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in ["reinforce", "ppo", "a2c"]:
        rs = runs[a]
        if not rs:
            continue
        e = np.array(best_run(rs)["entropy"])
        if len(e):
            ax.plot(e[:, 0], smooth(e[:, 1], 5).tolist() + [np.nan] * (len(e) - len(smooth(e[:, 1], 5))), label=LABEL[a], color=COLOR[a])
    ax.axhline(np.log(10), color="gray", linestyle="--", linewidth=1, label="uniform over 10 actions")
    ax.set_title("Policy entropy during training (best run per method)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("entropy (nats)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "policy_entropy.png"), dpi=150)
    plt.close(fig)


def convergence_frames(r):
    ep = np.array(r["episodes"])
    if len(ep) < 40:
        return None
    y = smooth(ep[:, 1], 30)
    x = ep[len(ep) - len(y):, 0]
    target = y.min() + 0.9 * (y.max() - y.min())
    hit = np.where(y >= target)[0]
    return float(x[hit[0]]) if len(hit) else None


def load_generalisation():
    path = os.path.join(ROOT, "logs", "generalisation.json")
    return json.load(open(path)) if os.path.exists(path) else None


SET_LABEL = {
    "fresh_seeds": "new seeds, same kind of start",
    "mild_shift": "prices +5%",
    "stress": "prices +12%, budget -4%",
}


def fig_convergence_generalisation(runs):
    gen = load_generalisation()
    names, conv, bests = [], [], []
    for a, rs in runs.items():
        if not rs:
            continue
        r = best_run(rs)
        names.append(LABEL[a])
        bests.append((a, r))
        conv.append(convergence_frames(r) or np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(names, conv, color=[COLOR[a] for a, _ in bests])
    ax[0].set_title("Steps to reach 90% of the best training level")
    ax[0].set_ylabel("environment steps")

    if gen is not None:
        groups = names + ["Rule owner"]
        x = np.arange(len(groups))
        lines = ["| Agent | " + " | ".join(SET_LABEL.values()) + " |", "|---|---|---|---|"]
        for i, set_name in enumerate(SET_LABEL):
            vals = [gen["models"][f"{a}/{r['name']}"][set_name]["mean_return"] for a, r in bests]
            vals.append(gen["baselines"][set_name]["rule"]["mean_return"])
            ax[1].bar(x + (i - 1) * 0.27, vals, 0.27, label=SET_LABEL[set_name])
        for g, (a, r) in zip(names, bests):
            row = [gen["models"][f"{a}/{r['name']}"][k] for k in SET_LABEL]
            lines.append(f"| {g} ({r['name']}) | " + " | ".join(
                f"{m['mean_return']:.1f} ({m['success_rate'] * 100:.0f}% finished)" for m in row) + " |")
        row = [gen["baselines"][k]["rule"] for k in SET_LABEL]
        lines.append("| Rule owner | " + " | ".join(
            f"{m['mean_return']:.1f} ({m['success_rate'] * 100:.0f}% finished)" for m in row) + " |")
        os.makedirs(TAB, exist_ok=True)
        open(os.path.join(TAB, "generalisation_table.md"), "w").write("\n".join(lines) + "\n")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(groups)
        ax[1].set_title("Generalisation: best run per method on unseen starts")
    else:
        x = np.arange(len(names))
        ax[1].bar(x - 0.2, [r["final"]["mean_return"] for _, r in bests], 0.4, label="seen kind of start")
        ax[1].bar(x + 0.2, [r["generalisation"]["mean_return"] for _, r in bests], 0.4, label="stress start")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(names)
        ax[1].set_title("Generalisation of the best run per method")
    ax[1].set_ylabel("mean return over 30 episodes")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence_generalisation.png"), dpi=150)
    plt.close(fig)


def fig_sweeps(runs):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for x, a in zip(ax.flat, ALGOS):
        rs = runs[a]
        if not rs:
            continue
        x.bar([r["name"][3:] for r in rs], [r["final"]["mean_return"] for r in rs], color=COLOR[a])
        x.set_title(LABEL[a] + ": final return by run")
        x.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sweep_overview.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = load_all()
    tables(runs)
    fig_rewards(runs)
    fig_dqn_loss(runs)
    fig_entropy(runs)
    fig_convergence_generalisation(runs)
    fig_sweeps(runs)
    for a in ALGOS:
        if runs[a]:
            b = best_run(runs[a])
            print(f"{LABEL[a]:<10} runs {len(runs[a]):>2}  best {b['name']}  return {b['final']['mean_return']:.1f}  "
                  f"success {b['final']['success_rate'] * 100:.0f}%  stress {b['generalisation']['mean_return']:.1f}  "
                  f"converged at {convergence_frames(b)}")
    print("figures in", FIG)


if __name__ == "__main__":
    main()
