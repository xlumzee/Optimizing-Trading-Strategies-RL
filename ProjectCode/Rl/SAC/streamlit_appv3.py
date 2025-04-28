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

    # load data & display first 5 rows
    df = pd.read_csv(tmp_csv)
    st.subheader("Data Preview (first 5 rows)")
    st.write(df.head())

    # prepare environment and model
    env = DummyVecEnv([lambda: make_env(tmp_csv, window)])
    sac = SAC.load(tmp_model)

    # rollout using compatibility wrappers
    obs, _ = _reset(env)
    exp_dim = sac.policy.observation_space.shape[0]
    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] < exp_dim:
        pad = np.zeros((obs.shape[0], exp_dim - obs.shape[1]), dtype=obs.dtype)
        obs = np.concatenate([obs, pad], axis=1)

    nav = []
    done = False
    while not done:
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[1] < exp_dim:
            pad = np.zeros((obs.shape[0], exp_dim - obs.shape[1]), dtype=obs.dtype)
            obs = np.concatenate([obs, pad], axis=1)

        action, _ = sac.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = _step(env, action)
        done = done or trunc
        if isinstance(info, list):
            info = info[0]
        nav.append(info['nav'])

    equity = np.array(nav) / nav[0]

        # display metric (convert fraction to percent correctly)
    cum_percent = (equity[-1] - 1)
    st.metric("Cumulative Return", f"{cum_percent:.2%}")

    # layout: equity curve and data statistics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Equity Curve")
        st.line_chart(pd.DataFrame({"Equity": equity}))
        st.subheader("Cumulative Return %")
        # chart percent rather than raw fraction
        cumret = (equity - 1)
        st.line_chart(pd.DataFrame({"Cumulative Return (%)": cumret / 85}))
    with col2:
        st.subheader("Data Statistics")
        st.write(df.describe().T)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Equity Curve")
        st.line_chart(pd.DataFrame({"Equity": equity}))
        st.subheader("Cumulative Return %")
        cumret = equity - 1
        st.line_chart(pd.DataFrame({"Cumulative Return": cumret}))
    with col2:
        st.subheader("Data Statistics")
        st.write(df.describe().T)

