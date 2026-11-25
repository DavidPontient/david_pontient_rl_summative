"""Entry point. Run with:  uv run main.py

Right now this runs the environment with a simple agent so you can see the site.
Trained agents are loaded the same way once the models exist.
"""
import argparse

import numpy as np

from environment import ConstructionEnv
from environment.baselines import random_policy, rule_policy

POLICIES = {"random": random_policy, "rule": rule_policy}


def main():
    parser = argparse.ArgumentParser(description="Kigali home build environment")
    parser.add_argument("--agent", choices=list(POLICIES), default="rule")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", help="run without the pygame window")
    args = parser.parse_args()

    env = ConstructionEnv(render_mode=None if args.no_render else "human")
    policy = POLICIES[args.agent]
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        env.action_space.seed(args.seed + ep)
        done = False
        while not done:
            action = policy(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"week {info['week']:>3}  phase {info['phase']:<11} action {info['event']:<16} reward {reward:+7.2f}  budget left {info['budget_left'] / 1e6:6.1f}M")
        print(f"episode {ep + 1}: return {env.total_reward:.1f}, success {info['success']}, weeks {info['week']}, budget left {info['budget_left'] / 1e6:.1f}M RWF")
    env.close()


if __name__ == "__main__":
    main()
