import numpy as np
from gymnasium.utils.env_checker import check_env

from environment import ConstructionEnv
from environment.baselines import rule_policy


def test_gymnasium_api():
    check_env(ConstructionEnv(), skip_render_check=True)


def test_rule_policy_finishes_a_build():
    env = ConstructionEnv()
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        obs, r, te, tr, info = env.step(rule_policy(env, obs))
        done = te or tr
    assert info["success"]


def test_invalid_action_is_penalised():
    env = ConstructionEnv()
    env.reset(seed=0)
    _, r, *_ = env.step(9)  # build with an empty site
    assert r < 0


def test_same_seed_same_prices():
    a, b = ConstructionEnv(), ConstructionEnv()
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    assert np.allclose(oa, ob)
