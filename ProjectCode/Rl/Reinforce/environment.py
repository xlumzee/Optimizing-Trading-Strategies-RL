# environment.py

import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100000):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.total_steps = len(self.df) - 1
        self.current_step = 0

        # Actions: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)

        # Observations: All features except date
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.df.columns) - 1,), dtype=np.float32
        )

        # Initialize state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.total_asset = self.balance
        self.max_asset = self.balance
        self.trades = []
        self.portfolio_values = []

        self.actions_memory = [] # same as dqn - to store actions taken for plotting

    # updating due to prev total asset not being present
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset = self.balance
        self.max_asset = self.balance
        self.trades = []
        self.portfolio_values = []
        self.prev_total_asset = self.total_asset  # Initialize prev_total_asset
        self.actions_memory = [] # for plots
        return self._next_observation()


    def _next_observation(self):
        obs = self.df.iloc[self.current_step].drop('date').values.astype(np.float32)
        return obs

    def step(self, action):
        done = False
        current_price = self.df.iloc[self.current_step]['PRC']
        transaction_cost = self.df.iloc[self.current_step]['TRAN_COST']

        # Execute action
        if action == 1:  # Buy
            if self.balance >= current_price + transaction_cost:
                self.shares_held += 1
                self.balance -= current_price + transaction_cost
                self.trades.append({'step': self.current_step, 'type': 'buy', 'price': current_price})
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price - transaction_cost
                self.trades.append({'step': self.current_step, 'type': 'sell', 'price': current_price})

        
        self.actions_memory.append(action) # recording the action

        # Update total asset
        self.total_asset = self.balance + self.shares_held * current_price
        self.max_asset = max(self.max_asset, self.total_asset)
        self.portfolio_values.append(self.total_asset)

        # Calculate reward
        #reward = self.total_asset - self.max_asset  # Penalize loss in asset value
        #reward -= transaction_cost  # Subtract transaction cost

        reward = self.total_asset - self.prev_total_asset
        self.prev_total_asset = self.total_asset

        # next step
        self.current_step += 1
        if self.current_step >= self.total_steps:
            done = True
            # Close any open positions
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price - transaction_cost
                self.shares_held = 0
                self.total_asset = self.balance
                self.portfolio_values.append(self.total_asset)

        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)

        return obs, reward, done, {}

    def render(self, mode='human'):
        profit = self.total_asset - self.initial_balance
        print(f'Step: {self.current_step}, Total Asset: {self.total_asset:.2f}, Profit: {profit:.2f}')
