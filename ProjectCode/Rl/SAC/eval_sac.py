#!/usr/bin/env python
"""evaluate_sac.py – Back‑test a trained Soft‑Actor‑Critic agent on daily
micro‑structure data and print / plot key performance metrics.

Compatible with both **Gymnasium ≥0.26** and legacy Gym <0.26 – no library
upgrade needed.

Example
-------
python evaluate_sac.py --csv Data/AKAM_daily.csv \
                       --model models/sac_microstrat.zip \
                       --window 30                # with plot
python evaluate_sac.py --csv Data/AKAM_daily.csv \
                       --model models/sac_microstrat.zip \
                       --window 30 --no-plot     # headless mode
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from train_sac import make_env  # reuse env factory from training script

###############################################################################
# 1.  Metrics helpers
###############################################################################

def _daily_returns(nav: np.ndarray) -> np.ndarray:
    return nav[1:] / nav[:-1] - 1.0


def calc_metrics(nav: np.ndarray) -> dict[str, float]:
    ret = _daily_returns(nav)
    total = nav[-1] / nav[0] - 1.0
    sharpe = (
        np.sqrt(252) * ret.mean() / ret.std(ddof=1)
        if ret.std(ddof=1) > 0
        else 0.0
    )
    max_dd = (1 - nav / np.maximum.accumulate(nav)).max()
    return {
        "Total Return": total,
        "Annualised Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }

###############################################################################
# 2.  Compat wrappers for old/new Gym APIs
###############################################################################

def _reset(env):
    out = env.reset()
    return out if isinstance(out, tuple) else (out, {})


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:  # Gymnasium / Gym ≥0.26
        return out
    obs, r, done, info = out  # legacy Gym
    return obs, r, done, False, info

###############################################################################
# 3.  Roll‑out logic
###############################################################################

def run_policy(model: SAC, env) -> np.ndarray:
    """Play one episode (deterministic); return NAV time‑series."""
    obs, _ = _reset(env)
    nav_hist: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = _step(env, action)
        done = bool(done or trunc)
        # DummyVecEnv wraps info in a list → unwrap
        if isinstance(info, list):
            info = info[0]
        nav_hist.append(info["nav"])
    return np.asarray(nav_hist, dtype=float)

###############################################################################
# 4.  CLI entry
###############################################################################

def main(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to test CSV file")
    p.add_argument("--model", required=True, help="Path to trained SAC .zip")
    p.add_argument("--window", type=int, default=30, help="Look‑back window size")
    p.add_argument("--no-plot", dest="no_plot", action="store_true", help="Skip equity plot for headless runs")
    args = p.parse_args(argv)

    # ── Build env & load agent ─────────────────────────────────────────────
    env = DummyVecEnv([lambda: make_env(args.csv, args.window)])
    model = SAC.load(args.model)

    # ── Roll‑out agent ─────────────────────────────────────────────────────
    nav_sac = run_policy(model, env)
    metrics_sac = calc_metrics(nav_sac)

    # ── Buy‑and‑Hold baseline (aligned length) ─────────────────────────────
    df = pd.read_csv(args.csv)
    bh_nav = (1 + df["RET"].iloc[args.window : args.window + len(nav_sac)].values).cumprod()
    metrics_bh = calc_metrics(bh_nav)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n⚡  SAC Performance")
    for k, v in metrics_sac.items():
        print(f"  {k:18}: {v:.2%}")

    print("\n📈  Buy‑and‑Hold")
    for k, v in metrics_bh.items():
        print(f"  {k:18}: {v:.2%}")

    # ── Optional plot ──────────────────────────────────────────────────────
    if not args.no_plot:
        plt.figure(figsize=(9, 4))
        plt.plot(nav_sac / nav_sac[0], label="SAC")
        plt.plot(bh_nav / bh_nav[0], label="Buy & Hold", linestyle="--")
        plt.title("Equity Curve – Growth of $1")
        plt.xlabel("Days")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
