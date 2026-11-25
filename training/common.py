"""Shared pieces for all four algorithms: the environment wrapper, evaluation and logging."""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import Compose, GymWrapper, TransformedEnv
from torchrl.envs.transforms import RewardScaling, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type

try:  # name used by recent TorchRL versions
    from torchrl.collectors import Collector
except ImportError:  # older versions
    from torchrl.collectors import SyncDataCollector as Collector

from environment.custom_env import ConstructionEnv

REWARD_SCALE = 0.05          # the raw return is around 100, this keeps the losses small
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Unseen starts used to test generalisation. Seeds are never used in training and
# the prices start 12% higher with a smaller budget than anything seen in training.
GENERALISATION_OPTIONS = {"price_shift": 0.12, "budget_scale": 0.96}


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env():
    """The same environment for every algorithm, wrapped for TorchRL."""
    base = ConstructionEnv()
    return TransformedEnv(
        GymWrapper(base),
        Compose(RewardScaling(loc=0.0, scale=REWARD_SCALE), RewardSum()),
    )


def env_sizes(env):
    n_obs = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    return int(n_obs), int(n_act)


def finished_episode_returns(batch):
    """Raw (unscaled) returns of the episodes that ended inside this batch."""
    done = batch["next", "done"].reshape(-1)
    ep = batch["next", "episode_reward"].reshape(-1)[done]
    return (ep / REWARD_SCALE).tolist()


@torch.no_grad()
def evaluate(policy, episodes: int = 20, seed0: int = 10_000, options: dict | None = None):
    """Run the policy greedily on fresh environments and report plain numbers."""
    env = ConstructionEnv()
    rets, success, weeks, left, overpaid = [], [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i, options=options)
        done = False
        while not done:
            td = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[])
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td = policy(td)
            action = td["action"]
            action = int(action.argmax(-1)) if action.dim() > 0 else int(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rets.append(env.total_reward)
        success.append(float(info["success"]))
        weeks.append(info["week"])
        left.append(info["budget_left"] / 1e6)
        overpaid.append(info["architect_overpaid"] / 1e6)
    return {
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "success_rate": float(np.mean(success)),
        "mean_weeks": float(np.mean(weeks)),
        "mean_budget_left_m": float(np.mean(left)),
        "mean_architect_overpaid_m": float(np.mean(overpaid)),
    }


def save_run(result: dict, algo: str, name: str):
    os.makedirs(os.path.join(ROOT, "logs", algo), exist_ok=True)
    path = os.path.join(ROOT, "logs", algo, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f)
    return path


def model_path(algo: str, name: str):
    folder = "dqn" if algo == "dqn" else "pg"
    os.makedirs(os.path.join(ROOT, "models", folder), exist_ok=True)
    return os.path.join(ROOT, "models", folder, f"{algo}_{name}.pt")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def seconds(self):
        return time.time() - self.t0
