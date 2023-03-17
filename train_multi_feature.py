import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
from datetime import *
from scipy.special import expit
from tqdm import tqdm
from data_extractor import dataloader, label_buy_sell_hold
from multi_feature_model import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def format_price(n):
    '''
    Formats a number into a string with 2 decimal places.
    '''
    # Convert n from numpy array to float
    n = float(n)
    
    if n < 0:
        return "-${0:2f}".format(abs(n))
    else:
        return "${0:2f}".format(abs(n))


def state_creator(data, timestep, window_size):
    '''
    Changes input data to be differences in stock prices,
    which represent price changes over time. 
    This will allow model to predict buy/sell/hold rather than the price itself.
    '''
    starting_id = timestep - window_size + 1
    if starting_id >= 0:
        windowed_data = data[starting_id:timestep + 1]
    else:
        windowed_data = -starting_id * [data[0]] + list(data[0: timestep + 1])
    
    state = []
    for i in range(window_size - 1):
        # expit is logistic sigmoid function, and avoids overflow errors associated w/ large diffs in stock price
        # https://i.stack.imgur.com/WY61Z.png
        state.append(expit(windowed_data[i + 1] - windowed_data[i]))
    return np.array([state])


def train_model(data, model, window_size, episodes, batch_size=32):
    for episode in range(1, episodes + 1): # For printing purposes
        print("Episode: {}/{}".format(episode, episodes))
        state = state_creator(data, 0, window_size + 1)
        total_profit = 0
        model.inventory = []

        for t in tqdm(range(len(data))):
            # print("Timestep: {}/{}".format(t, len(data)))
            action = model.trade(state)
            
            # If action is not a scalar, then take the element within all the embedded arrays
            # I don't know why it does this, someone feel free to fix it
            if not np.isscalar(action):
                print(action)
                # action = action[0][0][0]

            if t == len(data) - 1:
                # When the episode is done, we don't have a next state, so we set it to the last state
                next_state = state
            else:            
                next_state = state_creator(data, t + 1, window_size + 1)
            reward = 0

            # Buy stock
            if action == 1:
                model.inventory.append(data[t][0]) # Append the closing price
                # print("Buy: {}".format(format_price(data[t][0])))

            # Sell stock
            elif action == 2 and len(model.inventory) > 0:
                bought_price = model.inventory.pop(0) # This will be a scalar
                reward = max(data[t][0] - bought_price, 0)
                total_profit += data[t][0] - bought_price
                # print("Sell: {} | Profit: {}".format(format_price(data[t][0]), format_price(data[t][0] - bought_price)))

            # Hold stock
            elif action == 0:
                # print("Hold: {}".format(format_price(data[t][0])))
                pass
            
            # If the episode is done, we fit the model to the target
            done = True if t == len(data) - 1 else False    
            model.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: {}".format(format_price(total_profit)))
                print("--------------------------------")
            
            if len(model.memory) > batch_size:
                model.batch_train(batch_size)
        
        print("Total Profit: {}".format(format_price(total_profit)))
        print("Saving model...")
        model.model.save("models/model_multifeature.h5")


if __name__ == "__main__":
    # Hyperparameters  
    window_size = 10
    episodes = 10
    batch_size = 32
    stock = 'AAPL'
    
    data = dataloader(stock, 'data/', '2022-09-01', '2023-03-01')
    
    # We want to predict the closing price, but we also want to use the other features
    data = data[['Close']].values

    # Generate and append the buy/sell/hold signal to the data
    labels = np.array(label_buy_sell_hold(data))

    # Convert labels to 2D array of size (len(labels), 1)
    labels = labels.reshape(len(labels), 1)
    data = np.append(data, labels, axis=1)

    # Get the epsCurrentYear, epsForward, forwardPE, fiftyDayAverage, marketCap
    df = web.get_quote_yahoo(stock)
    df = df[["epsCurrentYear", "epsForward", "forwardPE", "fiftyDayAverage", "marketCap"]]

    # Extend df to match the length of data
    df = pd.concat([df] * len(data), ignore_index=True)
    
    # Append the epsCurrentYear, epsForward, forwardPE, fiftyDayAverage, marketCap to data
    data = np.append(data, df, axis=1)

    trader = Model(window_size, num_features=data.shape[1])
    trader.model = trader.model_builder()
    # trader.model.summary()
    
    train_model(data, trader, window_size, episodes, batch_size)

    ### Testing the model ###

    # Load the model
    trader.model = trader.model_builder()
    trader.model.load_weights("models/model_multifeature.h5")
    
    # Get the current date
    # today = datetime.today().strftime('%Y-%m-%d')
    today = '2022-08-25'
    today = '2022-07-21'

    # Get the date from a week ago
    week_ago = (datetime.strptime(today, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')

    # Get the stock closing price for the last week
    test_data = dataloader(stock, 'data/', week_ago, today)
    test_data = test_data[['Close']].values
    test_df = web.get_quote_yahoo(stock)
    test_df = test_df[["epsCurrentYear", "epsForward", "forwardPE", "fiftyDayAverage", "marketCap"]]
    test_df = pd.concat([test_df] * len(test_data), ignore_index=True)
    test_data = np.append(test_data, test_df, axis=1)

    # This is a placeholder signal, it will be replaced by the output of the sentiment analysis model
    signal = "buy" # Change this to get output from sentiment analysis model
    if signal == "buy":
        # Create an array of 1s the same length as test_data
        signal = np.ones((len(test_data), 1))
    elif signal == "sell":
        # Create an array of 2s the same length as test_data
        signal = 2 * np.ones((len(test_data), 1))
    elif signal == "hold":
        # Create an array of 0s the same length as test_data
        signal = np.zeros((len(test_data), 1))
    test_data = np.append(test_data, signal, axis=1)

    # Use the model to predict the stock price for tomorrow
    state = state_creator(test_data, 0, window_size + 1)
    action = trader.trade(state)

    actions = {
        0: "Hold",
        1: "Buy",
        2: "Sell"
    }
    print("Action for {} on {}: {}".format(stock, today, actions[action]))