# Kigali Home Build, reinforcement learning

An RL agent that manages the budget of a home build in Kigali, Rwanda. Every step is one week.
The agent buys materials, pays the crew, deals with the architect and tries to finish five
building phases before the money runs out. Reference prices and phase budgets come from the
receipts and the architect's bill of quantities of a real build.

Four agents are trained on the same environment with TorchRL: DQN, REINFORCE, PPO and A2C.

## Setup and run

Everything runs with uv only.

```
uv sync
uv run main.py                          # a rule based owner builds the house in the pygame window
uv run main.py --agent random           # a random agent for comparison
uv run play.py                          # the best trained agent builds the house
uv run play.py --model models/pg/ppo_run00.pt --episodes 3
uv run python -m pytest tests           # environment tests
```

## Training

The full sweep is 12 DQN runs and 10 runs each for REINFORCE, PPO and A2C, 300,000 steps per run.
It is meant to run on Google Colab, see `colab_training.ipynb`.

```
uv run python -m training.run_sweep --algo all --workers 2     # everything, safe to restart
uv run python -m training.run_sweep --algo ppo --runs 0-4      # a part of it
uv run python -m training.dqn_training --run 0                 # one run
uv run python -m training.pg_training --algo ppo --run 0
uv run python -m training.analysis                             # tables and figures
```

Logs are written to `logs/<algo>/runNN.json`, models to `models/dqn` and `models/pg`,
figures to `assets/figures`, tables to `logs/tables`.

## Environment

Actions (discrete, 10): wait, buy cement, buy steel, buy masonry, buy timber, compare suppliers,
pay labour, pay architect, verify architect request, build.

Observation (24 numbers): week, budget left, progress, phase, phase progress, 4 price levels,
4 price trends, 4 stock cover values, labour owed, architect request (pending, size, verified,
weeks waiting), supplier quote, crew can build.

Reward: progress on the house, minus money lost against the reference price, minus any overpaid
architect request, minus the weekly site cost, small penalty for invalid actions, bonus for
finishing and penalty for running out of money.

## Layout

```
main.py                     demo of the environment
play.py                     watch a trained agent
environment/custom_env.py   Gymnasium environment
environment/rendering.py    pygame view
environment/baselines.py    random and rule based reference policies
training/                   TorchRL training scripts, sweep runner, analysis
models/                     saved models
logs/                       training logs
assets/                     reference prices and figures
tests/                      environment tests
```

The original receipts and the architect spreadsheet are not part of the repository because they
contain account numbers and names. `assets/price_reference.json` has only the prices that were used.
