"""DQN with TorchRL: replay buffer, target network, epsilon greedy exploration."""
from __future__ import annotations

import argparse

import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import (
    Collector, GENERALISATION_OPTIONS, Timer, env_sizes, evaluate, finished_episode_returns,
    make_env, model_path, save_run, set_all_seeds,
)
from training.configs import get_config


def run_dqn(cfg: dict, name: str, seed: int = 0, eval_every: int = 25_000, verbose: bool = True):
    set_all_seeds(seed)
    timer = Timer()
    env = make_env()
    env.set_seed(seed)
    n_obs, n_act = env_sizes(env)

    qnet = MLP(in_features=n_obs, out_features=n_act, num_cells=[cfg["hidden"]] * 2, activation_class=nn.ReLU)
    qvalue = TensorDictModule(qnet, in_keys=["observation"], out_keys=["action_value"])
    actor = QValueActor(qvalue, spec=env.action_spec)

    egreedy = EGreedyModule(
        spec=env.action_spec, eps_init=1.0, eps_end=cfg["eps_end"], annealing_num_steps=cfg["eps_anneal"],
    )
    explore_policy = TensorDictSequential(actor, egreedy)

    collector = Collector(
        env, explore_policy, frames_per_batch=cfg["frames_per_batch"], total_frames=cfg["total_frames"],
    )
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])

    loss_module = DQNLoss(actor, action_space=env.action_spec, delay_value=True, double_dqn=cfg["double"])
    loss_module.make_value_estimator(gamma=cfg["gamma"])
    if cfg["target"] == "soft":
        updater = SoftUpdate(loss_module, tau=cfg["tau"])
    else:
        updater = HardUpdate(loss_module, value_network_update_interval=cfg["hard_interval"])
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    episodes, losses, evals = [], [], []
    frames, next_eval = 0, eval_every
    for batch in collector:
        n = batch.numel()
        frames += n
        for r in finished_episode_returns(batch):
            episodes.append([frames, r])
        buffer.extend(batch.reshape(-1))

        if frames >= cfg["warmup"] and len(buffer) >= cfg["batch_size"]:
            batch_loss = 0.0
            for _ in range(cfg["updates_per_batch"]):
                sample = buffer.sample()
                loss = loss_module(sample)["loss"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
                optim.step()
                updater.step()
                batch_loss += loss.item()
            losses.append([frames, batch_loss / cfg["updates_per_batch"]])
        egreedy.step(n)

        if frames >= next_eval:
            ev = evaluate(actor, episodes=10, seed0=20_000)
            evals.append([frames, ev["mean_return"], ev["success_rate"]])
            next_eval += eval_every
            if verbose:
                print(f"[dqn {name}] frames {frames:>7} eval return {ev['mean_return']:7.1f} success {ev['success_rate']:.2f}", flush=True)
    collector.shutdown()

    final = evaluate(actor, episodes=30, seed0=30_000)
    general = evaluate(actor, episodes=30, seed0=40_000, options=GENERALISATION_OPTIONS)
    torch.save({"state_dict": actor.state_dict(), "cfg": cfg}, model_path("dqn", name))
    result = {
        "algo": "dqn", "name": name, "seed": seed, "cfg": cfg, "episodes": episodes, "loss": losses,
        "evals": evals, "final": final, "generalisation": general, "seconds": timer.seconds(),
    }
    save_run(result, "dqn", name)
    if verbose:
        print(f"[dqn {name}] done in {timer.seconds():.0f}s, final {final['mean_return']:.1f}, unseen {general['mean_return']:.1f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=0, help="index of the config in training/configs.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="override total frames (for quick tests)")
    args = parser.parse_args()
    config = get_config("dqn", args.run)
    if args.frames:
        config["total_frames"] = args.frames
    run_dqn(config, f"run{args.run:02d}", seed=args.seed)
