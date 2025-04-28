#!/usr/bin/env python
"""train_sac.py – Train a Soft-Actor-Critic agent with:
  * automatic train/test split
  * realistic leverage cap
  * spread-based transaction costs
  * carry-cost penalty on large positions
  * Gaussian action-noise decay callback

Outputs:
  - models/sac_microstrat.zip            (trained weights)
  - <csv>_train.csv, <csv>_test.csv      (data splits)
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

###############################################################################
# 1.  Environment with leverage cap, spread cost, and carry penalty
###############################################################################

class EquityDailyEnv(gym.Env):
    """Trading env penalizing large positions and realistic spreads."""
    metadata = {"render.modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 30,
        init_balance: float = 1_000_000,
        carry_cost: float = 0.001,
    ) -> None:
        super().__init__()
        needed = {"RET","BA_SPREAD","ASK","BID"}
        missing = needed.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.window = window
        self.init_balance = init_balance
        self.carry_cost = carry_cost

        # half-spread cost uses BA_SPREAD and mid_price
        self.df["mid_price"] = (self.df["ASK"] + self.df["BID"]) / 2
        feats = ["RET","BA_SPREAD"]
        self.features = self.df[feats].astype(np.float32).values

        obs_dim = window*len(feats) + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-0.3,0.3,(1,), np.float32)

        self.i = 0; self.cash = 0.; self.pos = 0.; self.nav = 0.

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = self.window
        self.cash = float(self.init_balance)
        self.pos = 0.0
        self.nav = float(self.init_balance)
        return self._get_obs(), {}

    def step(self, action):
        # clip to ±30% leverage
        target = float(np.clip(action[0], -0.3, 0.3))
        desired_val = target * self.nav
        cur_val = self.pos * self.nav
        trade = desired_val - cur_val

        # spread-based transaction cost
        spread = float(self.df.loc[self.i, "BA_SPREAD"])
        mid = float(self.df.loc[self.i, "mid_price"])
        cost = abs(trade) * (spread/mid)

        # update cash & position
        self.cash -= trade + cost
        self.pos = target

        # advance time
        self.i += 1
        done = self.i >= len(self.df)-1

        # pnl & carry penalty
        ret = float(self.df.loc[self.i, "RET"])
        pnl = self.pos * ret * self.nav
        carry_pen = self.carry_cost * (self.pos**2)
        self.nav += pnl
        reward = pnl - cost - carry_pen

        info = {"nav": self.nav, "pnl": pnl, "cost": cost, "carry_pen": carry_pen}
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        feats = self.features[self.i-self.window:self.i].reshape(-1)
        return np.concatenate([feats, [self.cash/self.init_balance, self.pos]]).astype(np.float32)

###############################################################################
# 2.  Data split utility
###############################################################################

def load_and_split(csv_path: str|Path, test_size: float):
    p = Path(csv_path)
    df = pd.read_csv(p, parse_dates=["date"]).dropna().reset_index(drop=True)
    split = int(len(df)*(1-test_size))
    tr = df.iloc[:split].copy(); te = df.iloc[split:].copy()
    tr.to_csv(p.with_name(f"{p.stem}_train.csv"), index=False)
    te.to_csv(p.with_name(f"{p.stem}_test.csv"),  index=False)
    return tr, te

###############################################################################
# 3.  Reusable environment factory
###############################################################################

def make_env(csv_path: str | Path, window: int = 30, carry_cost: float = 0.001) -> EquityDailyEnv:
    """Load a CSV and return a configured EquityDailyEnv instance."""
    p = Path(csv_path)
    df = pd.read_csv(p, parse_dates=["date"]).dropna().reset_index(drop=True)
    return EquityDailyEnv(df, window=window, carry_cost=carry_cost)

###############################################################################
# 4.  Gaussian action-noise decay callback
###############################################################################
