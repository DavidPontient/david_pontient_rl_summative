"""Watch a trained agent build the house.

    uv run play.py                                  # best model found in logs/
    uv run play.py --model models/pg/ppo_run00.pt   # a specific model
    uv run play.py --episodes 3 --seed 100          # more episodes, other starts
"""
import argparse
import glob
import json
import os

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP, QValueActor
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment import ConstructionEnv
from environment.custom_env import OBS_DIM, ACTION_NAMES
from training.common import ROOT, make_env
from training.pg_training import build_actor


def load_policy(path):
    algo = os.path.basename(path).split("_")[0]
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["cfg"]
    env = make_env()
    if algo == "dqn":
        net = MLP(in_features=OBS_DIM, out_features=len(ACTION_NAMES), num_cells=[cfg["hidden"]] * 2, activation_class=nn.ReLU)
        module = TensorDictModule(net, in_keys=["observation"], out_keys=["action_value"])
        policy = QValueActor(module, spec=env.action_spec)
    else:
        policy = build_actor(env, cfg["hidden"])
    policy.load_state_dict(ckpt["state_dict"])
    return algo, policy


def best_model_path():
    best, best_score = None, -1e9
    for f in glob.glob(os.path.join(ROOT, "logs", "*", "*.json")):
        r = json.load(open(f))
        score = r["final"]["mean_return"]
        if score > best_score:
            folder = "dqn" if r["algo"] == "dqn" else "pg"
            best, best_score = os.path.join(ROOT, "models", folder, f"{r['algo']}_{r['name']}.pt"), score
    if best is None:
        raise SystemExit("No trained models found. Train first, see README.")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    path = args.model or best_model_path()
    algo, policy = load_policy(path)
    print(f"Loaded {algo.upper()} agent from {os.path.relpath(path, ROOT)}")

    env = ConstructionEnv(render_mode=None if args.no_render else "human")
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            td = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[])
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                action = int(policy(td)["action"].argmax(-1))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"week {info['week']:>3} | {info['phase']:<11} | {ACTION_NAMES[action]:<18} | reward {reward:+7.2f} | budget left {info['budget_left'] / 1e6:6.1f}M | progress {info['progress'] * 100:3.0f}%")
        print(f"Episode {ep + 1}: return {env.total_reward:.1f}, finished {info['success']}, weeks {info['week']}, budget left {info['budget_left'] / 1e6:.1f}M RWF")
    env.close()


if __name__ == "__main__":
    main()
