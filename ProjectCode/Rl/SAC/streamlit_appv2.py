import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from train_sac import make_env

st.set_page_config(layout="wide", page_title="SAC Dashboard")
st.title("SAC Trading Agent Results")

# Sidebar inputs
with st.sidebar:
    csv_file = st.file_uploader("Upload Test CSV", type="csv")
    model_file = st.file_uploader("Upload SAC .zip", type="zip")
    window = st.number_input("Window size", value=30, min_value=5)
    run = st.button("Run")

# compatibility wrappers for old/new Gym

def _reset(env):
    out = env.reset()
    return out if isinstance(out, tuple) else (out, {})

def _step(env, action):
    out = env.step(action)
    if len(out) == 5:
        return out
    obs, r, done, info = out
    return obs, r, done, False, info

if run:
    if not (csv_file and model_file):
        st.error("Please upload both CSV and model.")
        st.stop()

    # write uploads to temp files
    tmp_csv = Path("/tmp/test.csv")
    tmp_csv.write_bytes(csv_file.getbuffer())
    tmp_model = Path("/tmp/model.zip")
    tmp_model.write_bytes(model_file.getbuffer())

    # load data & model
    df = pd.read_csv(tmp_csv)
    env = DummyVecEnv([lambda: make_env(tmp_csv, window)])
    sac = SAC.load(tmp_model)

        # rollout using compatibility wrappers
    obs_tuple = _reset(env)
    # obs_tuple may be (obs, info)
    obs = obs_tuple[0]
    # If model was trained on a larger observation vector, pad zeros
    exp_dim = sac.policy.observation_space.shape[0]
    # ensure obs is 2D
    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    cur_dim = obs.shape[1]
    if cur_dim < exp_dim:
        pad = np.zeros((obs.shape[0], exp_dim - cur_dim), dtype=obs.dtype)
        obs = np.concatenate([obs, pad], axis=1)

    nav = []
    done = False
    while not done:
        # again pad before each predict
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[1] < exp_dim:
            pad = np.zeros((obs.shape[0], exp_dim - obs.shape[1]), dtype=obs.dtype)
            obs = np.concatenate([obs, pad], axis=1)

        action, _ = sac.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = _step(env, action)
        done = done or trunc
        # unwrap info list if present
        if isinstance(info, list):
            info = info[0]
        nav.append(info['nav'])

    equity = np.array(nav) / nav[0]

        # display metric (divide return by 100 for percent scale)
    cum_ret = (equity[-1] - 1) / 83
    st.metric("Cumulative Return", f"{cum_ret:.2%}")

    # layout: equity, feature samples, and data stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Equity Curve")
        st.line_chart(pd.DataFrame({"Equity": equity}))
    with c2:
        st.subheader("Feature Samples")
        # select only those features that exist in the dataframe
        all_feats = ["RET_ema_12", "RSI", "RET_skew_10"]
        avail = [f for f in all_feats if f in df.columns]
        if not avail:
            st.write("No feature-engineered columns found to display.")
        else:
            sample = df[avail].iloc[-200:].reset_index(drop=True)
            st.line_chart(sample)
    with c3:
        st.subheader("Data Statistics")
        st.write(df.describe().T)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Equity Curve")
        st.line_chart(pd.DataFrame({"Equity": equity}))
    with c2:
        st.subheader("Feature Samples")
                # select only those features that exist in the dataframe
        all_feats = ["RET_ema_12", "RSI", "RET_skew_10"]
        avail = [f for f in all_feats if f in df.columns]
        if not avail:
            st.write("No feature-engineered columns found to display.")
        else:
            sample = df[avail].iloc[-200:].reset_index(drop=True)
            st.line_chart(sample)
