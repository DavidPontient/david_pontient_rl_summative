"""Policy gradient methods with TorchRL: REINFORCE, PPO and A2C on the same environment."""
from __future__ import annotations

import argparse

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from training.common import (
    Collector, GENERALISATION_OPTIONS, Timer, env_sizes, evaluate, finished_episode_returns,
    make_env, model_path, save_run, set_all_seeds,
)
from training.configs import get_config


def build_actor(env, hidden):
    n_obs, n_act = env_sizes(env)
    net = MLP(in_features=n_obs, out_features=n_act, num_cells=[hidden] * 2, activation_class=nn.Tanh)
    module = TensorDictModule(net, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        module, in_keys=["logits"], spec=env.action_spec,
        distribution_class=OneHotCategorical, return_log_prob=True,
    )


def build_critic(env, hidden):
    n_obs, _ = env_sizes(env)
    net = MLP(in_features=n_obs, out_features=1, num_cells=[hidden] * 2, activation_class=nn.Tanh)
    return ValueOperator(net, in_keys=["observation"])


def _entropy_of(out, actor, data):
    """TorchRL leaves the entropy out of the loss output when entropy_coeff is 0, so compute it directly then."""
    if "entropy" in out.keys():
        return out["entropy"].item()
    with torch.no_grad():
        return actor.get_dist(data).entropy().mean().item()


def run_pg(algo: str, cfg: dict, name: str, seed: int = 0, eval_every: int = 25_000, verbose: bool = True):
    set_all_seeds(seed)
    timer = Timer()
    env = make_env()
    env.set_seed(seed)
    actor = build_actor(env, cfg["hidden"])
    critic = build_critic(env, cfg["hidden"]) if algo in ("ppo", "a2c") else None
    collector = Collector(env, actor, frames_per_batch=cfg["frames_per_batch"], total_frames=cfg["total_frames"])

    params = list(actor.parameters()) + (list(critic.parameters()) if critic is not None else [])
    optim = torch.optim.Adam(params, lr=cfg["lr"])

    if algo == "ppo":
        gae = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic)
        loss_module = ClipPPOLoss(
            actor, critic, clip_epsilon=cfg["clip_epsilon"], entropy_coeff=cfg["entropy_coeff"],
            critic_coeff=cfg["critic_coeff"], normalize_advantage=True,
        )
    elif algo == "a2c":
        gae = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic)
        loss_module = A2CLoss(
            actor, critic, entropy_coeff=cfg["entropy_coeff"], critic_coeff=cfg["critic_coeff"],
        )

    episodes, losses, entropies, evals = [], [], [], []
    frames, next_eval = 0, eval_every
    for batch in collector:
        n = batch.numel()
        frames += n
        for r in finished_episode_returns(batch):
            episodes.append([frames, r])

        if algo == "reinforce":
            rtg = reward2go(batch["next", "reward"], batch["next", "done"], cfg["gamma"])
            if cfg["normalize_returns"]:
                rtg = (rtg - rtg.mean()) / (rtg.std() + 1e-8)
            dist = actor.get_dist(batch)
            log_prob = dist.log_prob(batch["action"])
            entropy = dist.entropy().mean()
            loss = -(log_prob * rtg.squeeze(-1)).mean() - cfg["entropy_coeff"] * entropy
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            losses.append([frames, loss.item()])
            entropies.append([frames, entropy.item()])

        elif algo == "ppo":
            with torch.no_grad():
                gae(batch)
            flat = batch.reshape(-1)
            epoch_loss, epoch_entropy, count = 0.0, 0.0, 0
            for _ in range(cfg["epochs"]):
                order = torch.randperm(flat.shape[0])
                for start in range(0, flat.shape[0], cfg["minibatch_size"]):
                    sub = flat[order[start:start + cfg["minibatch_size"]]]
                    out = loss_module(sub)
                    loss = out["loss_objective"] + out["loss_critic"]
                    if "loss_entropy" in out.keys():
                        loss = loss + out["loss_entropy"]
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 0.5)
                    optim.step()
                    epoch_loss += loss.item()
                    epoch_entropy += _entropy_of(out, actor, sub)
                    count += 1
            losses.append([frames, epoch_loss / count])
            entropies.append([frames, epoch_entropy / count])

        else:  # a2c
            with torch.no_grad():
                gae(batch)
            flat = batch.reshape(-1)
            out = loss_module(flat)
            loss = out["loss_objective"] + out["loss_critic"]
            if "loss_entropy" in out.keys():
                loss = loss + out["loss_entropy"]
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 0.5)
            optim.step()
            losses.append([frames, loss.item()])
            entropies.append([frames, _entropy_of(out, actor, flat)])

        if frames >= next_eval:
            ev = evaluate(actor, episodes=10, seed0=20_000)
            evals.append([frames, ev["mean_return"], ev["success_rate"]])
            next_eval += eval_every
            if verbose:
                print(f"[{algo} {name}] frames {frames:>7} eval return {ev['mean_return']:7.1f} success {ev['success_rate']:.2f}", flush=True)
    collector.shutdown()

    final = evaluate(actor, episodes=30, seed0=30_000)
    general = evaluate(actor, episodes=30, seed0=40_000, options=GENERALISATION_OPTIONS)
    torch.save({"state_dict": actor.state_dict(), "cfg": cfg}, model_path(algo, name))
    result = {
        "algo": algo, "name": name, "seed": seed, "cfg": cfg, "episodes": episodes, "loss": losses,
        "entropy": entropies, "evals": evals, "final": final, "generalisation": general,
        "seconds": timer.seconds(),
    }
    save_run(result, algo, name)
    if verbose:
        print(f"[{algo} {name}] done in {timer.seconds():.0f}s, final {final['mean_return']:.1f}, unseen {general['mean_return']:.1f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["reinforce", "ppo", "a2c"], required=True)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="override total frames (for quick tests)")
    args = parser.parse_args()
    config = get_config(args.algo, args.run)
    if args.frames:
        config["total_frames"] = args.frames
    run_pg(args.algo, config, f"run{args.run:02d}", seed=args.seed)
