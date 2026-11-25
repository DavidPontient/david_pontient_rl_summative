"""Test every saved model on three kinds of unseen starts and compare with the rule based owner.

    uv run python -m training.evaluate_models

Writes logs/generalisation.json. Run it after the sweep and before training.analysis.

Test sets (all use seeds that were never used in training):
  fresh_seeds  same kind of start as in training, new random seeds
  mild_shift   prices start 5% higher
  stress       prices start 12% higher and the budget is 4% smaller
"""
import argparse
import glob
import json
import os

import numpy as np

from environment import ConstructionEnv
from environment.baselines import random_policy, rule_policy
from play import load_policy
from training.common import ROOT, evaluate

SETS = {
    "fresh_seeds": dict(seed0=50_000, options=None),
    "mild_shift": dict(seed0=60_000, options={"price_shift": 0.05}),
    "stress": dict(seed0=40_000, options={"price_shift": 0.12, "budget_scale": 0.96}),
}


def evaluate_baseline(policy, episodes, seed0, options):
    env = ConstructionEnv()
    rets, success, weeks = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i, options=options)
        env.action_space.seed(i)
        done = False
        while not done:
            obs, _, te, tr, info = env.step(policy(env, obs))
            done = te or tr
        rets.append(env.total_reward)
        success.append(float(info["success"]))
        weeks.append(info["week"])
    return {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
            "success_rate": float(np.mean(success)), "mean_weeks": float(np.mean(weeks))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    out = {"baselines": {}, "models": {}}
    for set_name, spec in SETS.items():
        out["baselines"][set_name] = {
            "rule": evaluate_baseline(rule_policy, args.episodes, spec["seed0"], spec["options"]),
            "random": evaluate_baseline(random_policy, args.episodes, spec["seed0"], spec["options"]),
        }
    print("baselines done", flush=True)

    for log in sorted(glob.glob(os.path.join(ROOT, "logs", "*", "run*.json"))):
        run = json.load(open(log))
        algo, name = run["algo"], run["name"]
        folder = "dqn" if algo == "dqn" else "pg"
        path = os.path.join(ROOT, "models", folder, f"{algo}_{name}.pt")
        if not os.path.exists(path):
            print("missing model", path)
            continue
        _, policy = load_policy(path)
        res = {s: evaluate(policy, episodes=args.episodes, seed0=spec["seed0"], options=spec["options"])
               for s, spec in SETS.items()}
        out["models"][f"{algo}/{name}"] = res
        print(f"{algo} {name}: " + "  ".join(f"{s} {r['mean_return']:.1f}" for s, r in res.items()), flush=True)

    with open(os.path.join(ROOT, "logs", "generalisation.json"), "w") as f:
        json.dump(out, f)
    print("saved logs/generalisation.json")


if __name__ == "__main__":
    main()
