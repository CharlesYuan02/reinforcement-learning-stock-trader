import gym
import numpy as np
from gym import spaces


class CustomStockTradingEnv(gym.Env):
    def __init__(self, df, window_size=10, k=1000, num_features=6, starting_balance=100000):
        '''
        df: dataframe of stock prices
        window_size: number of previous days to consider
        k: max number of shares to buy or sell
        num_features: number of features to consider (i.e. Number of columns in dataframe not including date)
        starting_balance: starting balance of account - the higher this is, the more leeway the agent has to make mistakes and learn
        __init__ should initialize the action space and observation space        
        '''
        self.df = df
        self.window_size = window_size
        self.k = k
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-k, high=k, shape=(window_size, num_features), dtype=np.float32)
        self.starting_balance = starting_balance
        self.reset()
    
    def reset(self):
        '''
        Include the following lists over the training period:
        self.profits: list of profits at each step
        self.total_profits: list of cumulative profits at each step
        self.shares: list of number of shares owned at each step
        self.account_balance: list of account balance at each step
        self.rewards: list of rewards at each step
        etc
        '''
        self.current_step = self.window_size
        self.total_portfolio_value = [self.starting_balance] # This is the account balance + value of shares owned at each iteration
        self.account_balance = [self.starting_balance]
        self.rewards = []
        self.num_shares = [0] # This is the total number of shares owned for the stock for each iteration
        self.trades = 0 # Extra variable to track number of trades

        # Technical indicators
        self.prices = self.df['Close'].values
        self.macd = self.df['MACD'].values
        self.signal = self.df['Signal'].values
        self.rsi = self.df['RSI'].values
        self.cci = self.df['CCI'].values
        self.adx = self.df['ADX'].values
        return self._next_observation()
    
    def _next_observation(self):
        # Shape of obs: (window_size, num_features)
        obs = np.array([
            self.prices[(self.current_step - self.window_size):self.current_step],
            self.macd[(self.current_step - self.window_size):self.current_step],
            self.signal[(self.current_step - self.window_size):self.current_step],
            self.rsi[(self.current_step - self.window_size):self.current_step],
            self.cci[(self.current_step - self.window_size):self.current_step],
            self.adx[(self.current_step - self.window_size):self.current_step]
        ])
        return np.transpose(obs)
    
    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            done = True
        else:
            done = False

        obs = self._next_observation()
        return obs, sum(self.rewards), done, {"account_balance": self.account_balance, "num_shares": self.num_shares, "total_portfolio_value": self.total_portfolio_value}
    
    def calculate_reward(self):
        '''
        Calculate reward based on two factors:
        1. account_balance[-1] - account_balance[-2]
            * This includes the profit made upon selling some shares
        2. Current value of the portfolio (i.e. num_shares * current_price)
        '''
        # raw_reward = self.total_portfolio_value[-1] - self.total_portfolio_value[-2] # Basically the same thing after clipping
        raw_reward = self.account_balance[-1] - self.account_balance[-2] + self.num_shares[-1] * self._get_current_price()
        reward = np.clip(raw_reward, -1, 1) # Clip reward to be between -1 and 1
        return reward
    
    def _take_action(self, action):
        current_price = self._get_current_price() # The closing price for the day
        action = action[0] # Should be a float between -1 and 1

        # Buy if action > 0 and if you have enough money
        if action > 0 and self.account_balance[-1] > current_price:
            # Convert float to number of shares based on set "k" value (max number of shares to buy)
            # If you don't have enough money, just buy as many as you can
            shares_bought = min(int(self.account_balance[-1] / current_price), int(action * self.k))
            self.account_balance.append(self.account_balance[-1] - shares_bought * current_price)
            self.num_shares.append(self.num_shares[-1] + shares_bought)
            self.total_portfolio_value.append(self.account_balance[-1] + self.num_shares[-1] * current_price)
            self.trades += 1
            self.rewards.append(self.calculate_reward())
            
        # Sell if action < 0 and if you have enough shares
        elif action < 0 and self.num_shares[-1] > 0:
            # Convert float to number of shares based on set "k" value (max number of shares to sell)
            # If you don't have enough shares, just sell as many as you can
            shares_sold = min(self.num_shares[-1], int(-action * self.k))
            self.account_balance.append(self.account_balance[-1] + shares_sold * current_price)
            self.num_shares.append(self.num_shares[-1] - shares_sold)
            self.total_portfolio_value.append(self.account_balance[-1] + self.num_shares[-1] * current_price)
            self.trades += 1
            self.rewards.append(self.calculate_reward())
        
        # Otherwise, just hold
        else:
            self.account_balance.append(self.account_balance[-1])
            self.num_shares.append(self.num_shares[-1])
            self.total_portfolio_value.append(self.account_balance[-1] + self.num_shares[-1] * current_price)
            self.rewards.append(self.calculate_reward())

    def _get_current_price(self):
        '''
        Return the closing price for the current day
        '''
        return self.prices[self.current_step]     