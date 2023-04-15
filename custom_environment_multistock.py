import gym
import numpy as np
from gym import spaces


class CustomStockTradingEnv(gym.Env):
    def __init__(self, dfs, window_size=10, k=1000, num_features=6, starting_balance=100000):
        '''
        df: dataframe of multiple stocks concatenated together (shape (num_stocks, num_days, num_features))
        window_size: number of previous days to consider
        k: max number of shares to buy or sell
        num_features: number of features to consider (i.e. Number of columns in dataframe not including date)
        starting_balance: starting balance of account - the higher this is, the more leeway the agent has to make mistakes and learn
        __init__ should initialize the action space and observation space        
        '''
        self.dfs = dfs
        self.window_size = window_size
        self.k = k
        self.num_stocks = len(dfs) # Number of stocks in the portfolio
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks, 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-k, high=k, shape=(self.num_stocks, window_size, num_features), dtype=np.float32)
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

        # This is the total number of shares owned for each stock for each iteration
        # It will have shape (num_iterations, num_stocks)
        zeros_list = [0] * self.num_stocks
        self.num_shares = [zeros_list]
        self.trades = 0 # Extra variable to track number of trades

        # Technical indicators - we want them in shape (num_stocks, num_iterations)
        self.prices = np.array([df["Close"].values for df in self.dfs])
        self.macd = np.array([df["MACD"].values for df in self.dfs])
        self.signal = np.array([df["Signal"].values for df in self.dfs])
        self.rsi = np.array([df["RSI"].values for df in self.dfs])
        self.cci = np.array([df["CCI"].values for df in self.dfs])
        self.adx = np.array([df["ADX"].values for df in self.dfs])
        return self._next_observation()
    
    def _next_observation(self):
        # We need the obs to be of shape (window_size, num_features, num_stocks)
        # obs starts off as shape (num_features, num_stocks, window_size)
        obs = np.array([self.prices[:, self.current_step - self.window_size:self.current_step],
                        self.macd[:, self.current_step - self.window_size:self.current_step],
                        self.signal[:, self.current_step - self.window_size:self.current_step],
                        self.rsi[:, self.current_step - self.window_size:self.current_step],
                        self.cci[:, self.current_step - self.window_size:self.current_step],
                        self.adx[:, self.current_step - self.window_size:self.current_step]])
        return np.transpose(obs, (1, 2, 0))
    
    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        # np.array(self.dfs).shape[1] is the number of days in the dataset
        if self.current_step >= np.array(self.dfs).shape[1] - 1:
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
        # self._get_current_price() returns an array of size (num_stocks, 1)
        # Do element-wise multiplication for the amount of each stock at most recent time step with the current prices for each stock
        # current_portfolio_value = np.sum(np.array(self.num_shares[-1]) * np.array(self._get_current_price()))
        # raw_reward = self.account_balance[-1] - self.account_balance[-2] + current_portfolio_value - self.total_portfolio_value[-1]
        raw_reward = self.total_portfolio_value[-1] - self.total_portfolio_value[-2]
        reward = np.clip(raw_reward, -1, 1) # Clip reward to be between -1 and 1
        return reward
    
    def _take_action(self, action):
        current_price = self._get_current_price() # The closing price for the day (list of size (num_stocks, 1))
        actions = action # Should be a list of floats between -1 and 1

        # Append these after looping through all the stocks during this iteration
        new_account_balance = self.account_balance[-1] # Scalar
        new_num_shares = self.num_shares[-1].copy() # List
        new_total_portfolio_value = self.total_portfolio_value[-1] # Scalar
        # print(new_total_portfolio_value, new_account_balance, new_num_shares)

        # Loop through each stock and perform buy/sell action
        for i in range(len(actions)): # For each stock, do the same thing as for single-stock training
            # print(new_account_balance) # The balance never goes negative, so I know the model isn't cheating...
            # Buy if action > 0 and if you have enough money
            if actions[i] > 0 and new_account_balance > current_price[i]:
                # Convert float to number of shares based on set "k" value (max number of shares to buy)
                # If you don't have enough money, just buy as many as you can
                shares_bought = min(int(new_account_balance / current_price[i]), int(actions[i] * self.k))
                new_account_balance -= shares_bought * current_price[i]
                new_num_shares[i] += shares_bought # This is the new number of shares for this stock
                new_total_portfolio_value += shares_bought * current_price[i]
                self.trades += 1
                
            # Sell if action < 0 and if you have enough shares
            elif actions[i] < 0 and new_num_shares[i] > 0:
                # Convert float to number of shares based on set "k" value (max number of shares to sell)
                # If you don't have enough shares, just sell as many as you can
                shares_sold = min(new_num_shares[i], int(-actions[i] * self.k)) # Min because actions[i] is negative
                new_account_balance += shares_sold * current_price[i]
                new_num_shares[i] -= shares_sold # This is the new number of shares for this stock
                new_total_portfolio_value -= shares_sold * current_price[i]
                self.trades += 1
            
            # Otherwise, just hold
            else:
                # new_account_balance += 0
                # new_num_shares[i] = new_num_shares[i]
                new_total_portfolio_value += new_num_shares[i] * current_price[i]
        
        # Append the new values to the lists
        self.account_balance.append(new_account_balance)
        self.num_shares.append(new_num_shares)
        self.total_portfolio_value.append(new_total_portfolio_value)
        self.rewards.append(self.calculate_reward()) # Cuz we need to calculate it based on the updated global arrays

    def _get_current_price(self):
        '''
        Return the closing prices for each stock at the current step
        '''
        return self.prices[:, self.current_step]