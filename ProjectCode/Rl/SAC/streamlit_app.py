#!/usr/bin/env python
"""streamlit_app.py – Interactive dashboard for evaluating a trained SAC
trading agent **without any external evaluation module**.

✔  Handles both Gymnasium ≥0.26 and old Gym automatically.
✔  Auto‑aligns equity‑curve arrays so DataFrame lengths always match.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from train_sac import make_env

###############################################################################
# 1.  Self‑contained evaluation helpers
###############################################################################

def _reset(env):
    out = env.reset()
    return out if isinstance(out, tuple) else (out, {})


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:
        return out
    obs, r, done, info = out
    return obs, r, done, False, info


def _daily_returns(nav: np.ndarray) -> np.ndarray:
    return nav[1:] / nav[:-1] - 1.0


def calc_metrics(nav: np.ndarray) -> dict[str, float]:
    ret = _daily_returns(nav)
    total = nav[-1] / nav[0] - 1.0
    sharpe = (np.sqrt(252) * ret.mean() / ret.std(ddof=1)) if ret.std(ddof=1) > 0 else 0.0
    max_dd = (1 - nav / np.maximum.accumulate(nav)).max()
    return {
        "Total Return": total,
        "Annualised Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }


def run_policy(model: SAC, env) -> np.ndarray:
    obs, _ = _reset(env)
    nav_hist: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = _step(env, action)
        done = bool(done or trunc)
        if isinstance(info, list):
            info = info[0]
        nav_hist.append(info["nav"])
    return np.asarray(nav_hist, dtype=float)


def buy_hold_nav(df: pd.DataFrame, start_idx: int, length: int) -> np.ndarray:
    """Return buy‑and‑hold equity curve of the same *length* as NAV."""
    bh = (1 + df["RET"].iloc[start_idx : start_idx + length].values).cumprod()
    return bh

###############################################################################
# 2.  Streamlit UI
###############################################################################

st.set_page_config(page_title="SAC Evaluator", layout="wide")
st.title("📊 SAC Trading Agent Evaluation")

with st.sidebar:
    st.header("Inputs")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    model_file = st.file_uploader("Upload Trained Model (.zip)", type=["zip"])
    window = st.number_input("Window size", value=30, min_value=5, max_value=120)
    run_btn = st.button("Run Evaluation", type="primary")

if run_btn:
    if csv_file is None or model_file is None:
        st.error("Please upload both a CSV and a model file before running.")
        st.stop()

    with st.spinner("Loading data & model …"):
        tmp_csv = Path(tempfile.mkstemp(suffix=".csv")[1])
        tmp_csv.write_bytes(csv_file.getbuffer())
        tmp_model = Path(tempfile.mkstemp(suffix=".zip")[1])
        tmp_model.write_bytes(model_file.getbuffer())

        df = pd.read_csv(tmp_csv)
        env = DummyVecEnv([lambda: make_env(tmp_csv, window)])
        model = SAC.load(tmp_model)

    with st.spinner("Running policy …"):
        nav = run_policy(model, env)
        metrics = calc_metrics(nav)
        bh_nav = buy_hold_nav(df, window, len(nav))
        bh_metrics = calc_metrics(bh_nav)

    # ── Metrics ────────────────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SAC Agent**")
        for k, v in metrics.items():
            st.metric(k, f"{v:.2%}")
    with col2:
        st.markdown("**Buy & Hold**")
        for k, v in bh_metrics.items():
            st.metric(k, f"{v:.2%}")

    # ── Equity curve ───────────────────────────────────────────────────────
    st.subheader("Equity Curve – Growth of $1")
    sac_curve = nav / nav[0]
    bh_curve = bh_nav / bh_nav[0]
    curve_df = pd.DataFrame({"SAC": sac_curve, "Buy & Hold": bh_curve})
    st.line_chart(curve_df)

    # ── NAV table (optional) ───────────────────────────────────────────────
    with st.expander("NAV history table"):
        st.dataframe(pd.DataFrame({"NAV": nav}), use_container_width=True)
