"""Construction budget environment for a home build in Kigali, Rwanda.

The agent plays the owner of a house build. Every step is one week. The agent
has to buy materials at the right time, pay the crew, deal with the architect's
payment requests and finish all five building phases before the money runs out.

All reference prices and phase budgets come from a real build (receipts and an
architect bill of quantities). Price movement over time is simulated, because
only about eight weeks of receipts exist.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Constants taken from the real build
# ---------------------------------------------------------------------------
MATERIALS = ["cement", "steel", "masonry", "timber"]
PHASES = ["Preliminary", "Foundation", "Elevation", "Roofing", "Finishes"]

# Bill of quantities, phase budgets in RWF (sum is 108,561,774)
PHASE_BUDGET = np.array([4_500_000, 11_982_500, 52_180_000, 3_533_100, 36_366_174], dtype=np.float64)
TOTAL_BUDGET = float(PHASE_BUDGET.sum())
PROGRESS_WEIGHT = PHASE_BUDGET / TOTAL_BUDGET

# Cost of one purchase lot at the reference price level (RWF)
#   cement: 200 bags x 14,800
#   steel : 50 bars of 12mm x 16,000 + 96 bars of 8mm x 8,200
#   masonry: 2 sand loads + 2 stone trucks
#   timber: 1,300 pieces of imbaho x 3,100
LOT_COST = np.array([2_960_000, 1_587_200, 2_660_000, 4_030_000], dtype=np.float64)

# Material needed per phase, measured in lots (rows are phases, columns are MATERIALS).
# This split is a modelling assumption, the receipts do not say which phase used what.
PHASE_NEED = np.array([
    [0.0, 0.0, 0.5, 0.0],   # Preliminary
    [1.5, 1.0, 1.0, 0.0],   # Foundation
    [5.0, 4.0, 4.5, 0.5],   # Elevation
    [0.0, 0.5, 0.0, 0.5],   # Roofing
    [2.5, 1.5, 1.5, 2.0],   # Finishes
])
PHASE_BUILDS = np.array([2, 3, 8, 2, 5])                      # build weeks needed per phase
PHASE_LABOUR = np.array([1.5e6, 2.0e6, 9.0e6, 1.2e6, 8.0e6])  # labour paid in instalments (RWF)
ARCHITECT_SHARE = 0.04                                        # fair request is 4% of the phase budget

# Weekly and per purchase costs
WATCHMAN_WEEK = 6_000 * 7     # watchman, 6,000 RWF per day on the receipts
HANDLING_FEE = 8_000          # unloading and transport per purchase
QUOTE_FEE = 15_000            # cost of getting quotes from several suppliers
DISCOUNT_RANGE = (0.03, 0.10)  # discounts seen between list and paid price on the receipts
STOCK_CAP = 3.0               # lots of one material the site can store
STOCK_DECAY = np.array([0.010, 0.002, 0.002, 0.004])  # weekly loss (cement goes hard in the damp)

# Action ids
WAIT, BUY_CEMENT, BUY_STEEL, BUY_MASONRY, BUY_TIMBER = 0, 1, 2, 3, 4
COMPARE, PAY_LABOUR, PAY_ARCHITECT, VERIFY, BUILD = 5, 6, 7, 8, 9
ACTION_NAMES = [
    "Wait", "Buy cement", "Buy steel", "Buy masonry", "Buy timber",
    "Compare suppliers", "Pay labour", "Pay architect", "Verify request", "Build",
]

# Reward scales
COST_SCALE = 400.0       # reward points per 100% of the total budget saved or lost
PROGRESS_SCALE = 100.0   # reward points for finishing the whole house
SUCCESS_BONUS = 50.0
FAIL_PENALTY = -50.0
INVALID_PENALTY = -0.5

OBS_DIM = 24


class ConstructionEnv(gym.Env):
    """One home build, one action per week."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode: str | None = None, max_weeks: int = 150, animate: bool = True):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.max_weeks = max_weeks
        self.animate = animate
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32)
        self._renderer = None
        self.reset_state_vars()

    # ------------------------------------------------------------------
    def reset_state_vars(self):
        self.week = 0
        self.phase = 0
        self.phase_builds = 0
        self.budget = TOTAL_BUDGET
        self.stock = np.zeros(4)
        self.labour_owed = 0.0
        self.req_pending = False
        self.req_amount = 0.0
        self.req_fair = 0.0
        self.req_verified = False
        self.req_weeks = 0
        self.discount = 0.0
        self.discount_ttl = 0
        self.x = np.zeros(4)
        self.season_offset = 0
        self.price_hist = []
        self.spent = {"materials": 0.0, "labour": 0.0, "architect": 0.0, "overhead": 0.0, "quotes": 0.0}
        self.overpaid_architect = 0.0
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.last_event = "start"
        self.invalid_count = 0

    # ------------------------------------------------------------------
    # Price model
    # ------------------------------------------------------------------
    def _season(self, week: int) -> np.ndarray:
        phase_shift = np.array([0.0, 0.8, 1.6, 2.4])
        return 0.03 * np.sin(2 * np.pi * (week + self.season_offset) / 52.0 + phase_shift)

    def _ratio(self) -> np.ndarray:
        return np.clip(np.exp(self.x + self._season(self.week)), 0.75, 1.7)

    def _step_prices(self):
        rng = self.np_random
        common = rng.normal(0.0, 0.012)
        idio = rng.normal(0.0, 0.012, 4)
        self.x = 0.92 * self.x + 0.0010 + common + idio
        if rng.random() < 0.03:  # supply shock on one material (shortage, import delay)
            m = rng.integers(0, 4)
            self.x[m] += rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.15)

    # ------------------------------------------------------------------
    def _issue_request(self):
        if self.phase >= len(PHASES):
            return
        fair = ARCHITECT_SHARE * PHASE_BUDGET[self.phase]
        inflated = self.np_random.random() < 0.5
        mult = 1.0 + (self.np_random.uniform(0.15, 0.35) if inflated else 0.0)
        self.req_fair = fair
        self.req_amount = fair * mult
        self.req_pending = True
        self.req_verified = False
        self.req_weeks = 0

    def _need_per_build(self) -> np.ndarray:
        if self.phase >= len(PHASES):
            return np.zeros(4)
        return PHASE_NEED[self.phase] / PHASE_BUILDS[self.phase]

    def _wage(self) -> float:
        if self.phase >= len(PHASES):
            return 0.0
        return PHASE_LABOUR[self.phase] / PHASE_BUILDS[self.phase]

    def _hold(self) -> bool:
        """The architect stops work at half way if the request is still unpaid."""
        if self.phase >= len(PHASES) or not self.req_pending:
            return False
        return self.phase_builds >= int(np.ceil(0.5 * PHASE_BUILDS[self.phase]))

    def _can_build(self) -> bool:
        if self.phase >= len(PHASES):
            return False
        if self._hold():
            return False
        if self.labour_owed > self._wage() + 1e-6:
            return False
        return bool(np.all(self.stock + 1e-9 >= self._need_per_build()))

    def progress(self) -> float:
        total = float(PROGRESS_WEIGHT[: min(self.phase, 5)].sum())
        if self.phase < len(PHASES):
            total += PROGRESS_WEIGHT[self.phase] * self.phase_builds / PHASE_BUILDS[self.phase]
        return min(total, 1.0)

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        ratio = self._ratio()
        mom = np.clip((ratio - self.price_hist[0]) / 0.2, -1.0, 1.0)
        need = self._need_per_build()
        cover = np.ones(4)
        for m in range(4):
            if need[m] > 1e-9:
                cover[m] = min(self.stock[m] / need[m], 4.0) / 4.0
        wage = self._wage()
        owed = min(self.labour_owed / wage, 2.0) / 2.0 if wage > 0 else 0.0
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = self.week / self.max_weeks
        obs[1] = np.clip(self.budget / TOTAL_BUDGET, 0.0, 1.2)
        obs[2] = self.progress()
        obs[3] = min(self.phase, 5) / 5.0
        obs[4] = (self.phase_builds / PHASE_BUILDS[self.phase]) if self.phase < 5 else 1.0
        obs[5:9] = np.clip((ratio - 0.7), 0.0, 1.0)
        obs[9:13] = mom
        obs[13:17] = cover
        obs[17] = owed
        obs[18] = 1.0 if self.req_pending else 0.0
        obs[19] = min(self.req_amount / 3_000_000.0, 1.0) if self.req_pending else 0.0
        obs[20] = 1.0 if self.req_verified else 0.0
        obs[21] = min(self.req_weeks / 4.0, 1.0)
        obs[22] = self.discount / DISCOUNT_RANGE[1]
        obs[23] = 1.0 if self._can_build() else 0.0
        return obs

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.reset_state_vars()
        rng = self.np_random
        self.season_offset = int(options.get("season_offset", rng.integers(0, 52)))
        self.x = rng.normal(0.0, 0.05, 4) + options.get("price_shift", 0.0)
        self.budget = TOTAL_BUDGET * options.get("budget_scale", rng.uniform(0.97, 1.03))
        self.price_hist = [self._ratio().copy() for _ in range(4)]
        self._issue_request()
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def _info(self):
        return {
            "week": self.week,
            "phase": PHASES[self.phase] if self.phase < 5 else "Done",
            "progress": self.progress(),
            "budget_left": self.budget,
            "spent": dict(self.spent),
            "architect_overpaid": self.overpaid_architect,
            "invalid_actions": self.invalid_count,
            "price_ratio": self._ratio().copy(),
        }

    # ------------------------------------------------------------------
    def step(self, action):
        a = int(action)
        reward = 0.0
        event = "wait"
        invalid = False
        ratio = self._ratio()

        if a == WAIT:
            event = "wait"

        elif BUY_CEMENT <= a <= BUY_TIMBER:
            m = a - 1
            paid = LOT_COST[m] * ratio[m] * (1.0 - self.discount) + HANDLING_FEE
            if self.stock[m] + 1.0 > STOCK_CAP + 1e-9 or paid > self.budget:
                invalid = True
            else:
                self.budget -= paid
                self.spent["materials"] += paid
                self.stock[m] += 1.0
                reward -= COST_SCALE * (paid - LOT_COST[m]) / TOTAL_BUDGET
                self.discount, self.discount_ttl = 0.0, 0
                event = f"buy:{MATERIALS[m]}"

        elif a == COMPARE:
            if self.discount > 0 or self.budget < QUOTE_FEE:
                invalid = True
            else:
                self.budget -= QUOTE_FEE
                self.spent["quotes"] += QUOTE_FEE
                reward -= COST_SCALE * QUOTE_FEE / TOTAL_BUDGET
                self.discount = float(self.np_random.uniform(*DISCOUNT_RANGE))
                self.discount_ttl = 2
                event = "compare"

        elif a == PAY_LABOUR:
            if self.labour_owed <= 1.0 or self.labour_owed > self.budget:
                invalid = True
            else:
                self.budget -= self.labour_owed
                self.spent["labour"] += self.labour_owed
                self.labour_owed = 0.0
                event = "pay_labour"

        elif a == PAY_ARCHITECT:
            if not self.req_pending or self.req_amount > self.budget:
                invalid = True
            else:
                paid = self.req_amount
                self.budget -= paid
                self.spent["architect"] += paid
                over = paid - self.req_fair
                self.overpaid_architect += over
                reward -= COST_SCALE * over / TOTAL_BUDGET
                self.req_pending, self.req_verified, self.req_amount = False, False, 0.0
                event = "pay_architect"

        elif a == VERIFY:
            if not self.req_pending or self.req_verified:
                invalid = True
            else:
                self.req_amount = self.req_fair
                self.req_verified = True
                event = "verify"

        elif a == BUILD:
            if not self._can_build():
                invalid = True
            else:
                before = self.progress()
                self.stock -= self._need_per_build()
                self.labour_owed += self._wage()
                self.phase_builds += 1
                if self.phase_builds >= PHASE_BUILDS[self.phase]:
                    self.phase += 1
                    self.phase_builds = 0
                    self._issue_request()
                reward += PROGRESS_SCALE * (self.progress() - before)
                event = "build"

        if invalid:
            reward += INVALID_PENALTY
            self.invalid_count += 1
            event = "invalid"

        # end of the week
        self.budget -= WATCHMAN_WEEK
        self.spent["overhead"] += WATCHMAN_WEEK
        reward -= COST_SCALE * WATCHMAN_WEEK / TOTAL_BUDGET
        self.stock = self.stock * (1.0 - STOCK_DECAY)
        if self.req_pending:
            self.req_weeks += 1
        if self.discount_ttl > 0:
            self.discount_ttl -= 1
            if self.discount_ttl == 0:
                self.discount = 0.0
        self.price_hist.pop(0)
        self.price_hist.append(ratio.copy())
        self._step_prices()
        self.week += 1

        terminated = False
        truncated = False
        success = False
        if self.phase >= len(PHASES) and self.labour_owed <= 1.0 and not self.req_pending:
            terminated, success = True, True
            reward += SUCCESS_BONUS
        elif self.budget < 300_000:
            terminated = True
            reward += FAIL_PENALTY
        elif self.week >= self.max_weeks:
            truncated = True

        self.total_reward += reward
        self.last_action, self.last_reward, self.last_event = a, reward, event
        info = self._info()
        info["success"] = success
        info["event"] = event

        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def action_mask(self) -> np.ndarray:
        """True for actions that would not be invalid in the current state."""
        mask = np.ones(len(ACTION_NAMES), dtype=bool)
        ratio = self._ratio()
        for m in range(4):
            paid = LOT_COST[m] * ratio[m] * (1.0 - self.discount) + HANDLING_FEE
            mask[1 + m] = not (self.stock[m] + 1.0 > STOCK_CAP + 1e-9 or paid > self.budget)
        mask[COMPARE] = not (self.discount > 0 or self.budget < QUOTE_FEE)
        mask[PAY_LABOUR] = not (self.labour_owed <= 1.0 or self.labour_owed > self.budget)
        mask[PAY_ARCHITECT] = not (not self.req_pending or self.req_amount > self.budget)
        mask[VERIFY] = not (not self.req_pending or self.req_verified)
        mask[BUILD] = self._can_build()
        return mask

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import ConstructionRenderer
            self._renderer = ConstructionRenderer(self, show_window=(self.render_mode == "human"))
        return self._renderer.render(animate=self.animate and self.render_mode == "human")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
