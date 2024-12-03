
import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100000):

        """
        Initialize the Trading Environment.

        Parameters:
        - df (pandas.DataFrame): The market data used for the environment.
        - initial_balance (float): The starting cash balance for the agent (default is 100,000).

        Main Attributes:
        - action_space: Defines the set of possible actions (Hold, Buy, Sell).
        - observation_space: Defines the shape and type of observations the agent receives.
        - balance: Current cash balance of the agent.
        - shares_held: Number of shares the agent currently holds.
        - total_asset: Total value of the agent's portfolio (cash + value of held shares).
        - portfolio_values: Records the portfolio value over time.
        - actions_memory: Stores the actions taken by the agent at each time step.
        """
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

        # adding to plot the results
        self.actions_memory = []  # To store actions taken


    def reset(self):

        """
        Resets the environment to its initial state at the beginning of an episode.

        This method resets all the environment's internal state variables to their
        initial values, clearing any previous episode's data. It prepares the environment
        for a new episode by resetting the current step, balances, holdings, and memory
        used for tracking actions and portfolio values.

        Returns:
            numpy.ndarray: The initial observation of the environment after resetting.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset = self.balance
        self.max_asset = self.balance
        self.trades = []
        self.portfolio_values = []
        self.actions_memory = [] # added for the memory to plot

        return self._next_observation()

    def _next_observation(self):
        """
        Retrieves the next observation from the environment.

        This method extracts the feature data for the current time step (self.current_step)
        from the DataFrame (self.df), excluding the 'date' column as it is not part of the obs. It converts the data
        into a NumPy array of type float32, which represents the state that will be
        provided to the agent.

        Returns:
            numpy.ndarray: The observation of the environment at the current time step.
        """
        obs = self.df.iloc[self.current_step].drop('date').values.astype(np.float32)
        return obs

    def step(self, action):
        """
        Executes one time step within the environment based on the action taken by the agent.

        This method updates the environment's state according to the action provided:
        - If the action is Buy (1), it checks if there is enough balance to buy one share plus transaction cost.
        - If the action is Sell (2), it checks if the agent has at least one share to sell.
        - Records the action taken.
        - Updates the balance, shares held, total asset value, and maximum asset value.
        - Calculates the reward based on the change in total asset value and transaction costs.
        - Advances the current step and checks if the episode is done.

        Parameters:
            action (int): The action taken by the agent (0=Hold, 1=Buy, 2=Sell).

        Returns:
            tuple:
                - obs (numpy.ndarray): The next observation of the environment.
                - reward (float): The reward received after taking the action.
                - done (bool): A flag indicating whether the episode has ended.
                - info (dict): Additional information (empty in this case).
        """
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

        # record the action
        self.actions_memory.append(action)

        # Update total asset
        self.total_asset = self.balance + self.shares_held * current_price
        self.max_asset = max(self.max_asset, self.total_asset)
        self.portfolio_values.append(self.total_asset)

        # Calculate reward
        reward = self.total_asset - self.max_asset  # Penalize loss in asset value
        reward -= transaction_cost  # Subtract transaction cost

        # Proceed to next step
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

    # build and incorporate
    def render(self, mode='human'):
        """"
        
        Returns : 
            
        
        """
        profit = self.total_asset - self.initial_balance
        print(f'Step: {self.current_step}, Total Asset: {self.total_asset:.2f}, Profit: {profit:.2f}')
