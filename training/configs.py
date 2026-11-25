"""Hyperparameter sweeps. Each list is one table in the report.

Run 0 is the base setting. Every other run changes one thing (or two for the
exploration setting) so the effect can be read from the table.
"""

DQN_BASE = dict(
    lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64,
    eps_end=0.05, eps_anneal=100_000, target="soft", tau=0.01, hard_interval=500,
    hidden=128, frames_per_batch=64, updates_per_batch=16, warmup=2_000,
    double=False, total_frames=300_000,
)

DQN_RUNS = [
    {},
    dict(lr=3e-4),
    dict(lr=3e-3),
    dict(gamma=0.95),
    dict(gamma=0.999),
    dict(buffer_size=10_000),
    dict(buffer_size=150_000),
    dict(batch_size=32),
    dict(batch_size=256),
    dict(eps_end=0.20, eps_anneal=150_000),
    dict(target="hard", hard_interval=500),
    dict(double=True, tau=0.005),
]

REINFORCE_BASE = dict(
    lr=3e-3, gamma=0.99, hidden=128, entropy_coeff=0.01, frames_per_batch=500,
    normalize_returns=True, total_frames=300_000,
)

REINFORCE_RUNS = [
    {},
    dict(lr=1e-3),
    dict(lr=1e-2),
    dict(gamma=0.95),
    dict(gamma=0.999),
    dict(entropy_coeff=0.0),
    dict(entropy_coeff=0.05),
    dict(frames_per_batch=250),
    dict(frames_per_batch=2_000),
    dict(normalize_returns=False),
]

PPO_BASE = dict(
    lr=3e-4, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, entropy_coeff=0.01,
    critic_coeff=0.5, epochs=8, minibatch_size=256, frames_per_batch=2_048,
    hidden=128, total_frames=300_000,
)

PPO_RUNS = [
    {},
    dict(lr=1e-4),
    dict(lr=1e-3),
    dict(clip_epsilon=0.1),
    dict(clip_epsilon=0.3),
    dict(entropy_coeff=0.0),
    dict(entropy_coeff=0.05),
    dict(lmbda=0.9),
    dict(epochs=3),
    dict(frames_per_batch=8_192, minibatch_size=512),
]

A2C_BASE = dict(
    lr=7e-4, gamma=0.99, lmbda=0.95, entropy_coeff=0.01, critic_coeff=0.5,
    frames_per_batch=256, hidden=128, total_frames=300_000,
)

A2C_RUNS = [
    {},
    dict(lr=3e-4),
    dict(lr=3e-3),
    dict(gamma=0.95),
    dict(lmbda=0.8),
    dict(lmbda=1.0),
    dict(entropy_coeff=0.0),
    dict(entropy_coeff=0.05),
    dict(frames_per_batch=64),
    dict(frames_per_batch=1_024),
]

SWEEPS = {
    "dqn": (DQN_BASE, DQN_RUNS),
    "reinforce": (REINFORCE_BASE, REINFORCE_RUNS),
    "ppo": (PPO_BASE, PPO_RUNS),
    "a2c": (A2C_BASE, A2C_RUNS),
}


def get_config(algo: str, run: int) -> dict:
    base, runs = SWEEPS[algo]
    cfg = dict(base)
    cfg.update(runs[run])
    return cfg
