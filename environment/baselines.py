"""Two simple reference policies. They are used to sanity check the reward and as a baseline in the report."""
import numpy as np

from environment.custom_env import (
    WAIT, BUY_CEMENT, COMPARE, PAY_LABOUR, PAY_ARCHITECT, VERIFY, BUILD,
    PHASE_NEED, PHASE_BUILDS, STOCK_CAP,
)


def random_policy(env, obs):
    return env.action_space.sample()


def rule_policy(env, obs):
    """A careful owner: pays when asked, verifies big requests, buys only what the next builds need."""
    if env.req_pending and not env.req_verified and env.req_fair > 1_000_000:
        return VERIFY
    if env.req_pending and (env.req_verified or env.req_fair <= 1_000_000):
        if env.phase_builds >= max(PHASE_BUILDS[env.phase] // 2 - 1, 0) or env.req_verified:
            return PAY_ARCHITECT
    if env.labour_owed > 1.0 and env.labour_owed >= env._wage() * 0.99:
        return PAY_LABOUR
    if env._can_build():
        return BUILD
    if env.phase < 5:
        need = env._need_per_build()
        cover = np.where(need > 1e-9, env.stock / np.maximum(need, 1e-9), 99.0)
        m = int(np.argmin(cover))
        if cover[m] < 1.0 and env.stock[m] + 1.0 <= STOCK_CAP:
            if env.discount <= 0 and env.budget > 5_000_000:
                return COMPARE
            return BUY_CEMENT + m
    if env.req_pending:
        return PAY_ARCHITECT
    return WAIT
