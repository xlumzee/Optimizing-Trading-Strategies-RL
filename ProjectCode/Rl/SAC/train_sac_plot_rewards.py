#!/usr/bin/env python
"""train_sac_plot_rewards.py – Train SAC agent and plot total reward per episode.

Usage:
    python train_sac_plot_rewards.py --csv Data/AKAM.csv --timesteps 200000 --window 30
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from train_sac import make_env, load_and_split, EquityDailyEnv


class EpisodeRewardCallback(BaseCallback):
    """Callback that records total reward at the end of each episode."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                # 'r' holds the cumulative reward for the episode
                self.episode_rewards.append(ep["r"])
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Full CSV path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total timesteps to train")
    parser.add_argument("--window", type=int, default=30, help="Look-back window size")
    parser.add_argument("--model-dir", default="models", help="Directory to save model")
    args = parser.parse_args()

    # Split data and create training environment
    train_df, _ = load_and_split(args.csv, args.test_size)
    def make_train_env():
        return EquityDailyEnv(train_df, window=args.window)

    # Wrap in VecMonitor to generate 'episode' info
    env = DummyVecEnv([make_train_env])
    env = VecMonitor(env)

    # Create SAC model
    model = SAC("MlpPolicy", env, verbose=1)

    # Train with callback to record episode rewards
    callback = EpisodeRewardCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Extract recorded rewards
    rewards = callback.episode_rewards

    # Plot rewards per episode
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker="o")
    plt.title("SAC Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save model
    Path(args.model_dir).mkdir(exist_ok=True)
    save_path = Path(args.model_dir) / "sac_plot_rewards"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

if __name__ == "__main__":
    main()
