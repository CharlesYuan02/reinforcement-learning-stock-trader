import os
import numpy as np
from datetime import *
from scipy.special import expit
from tqdm import tqdm
from data_extractor import dataloader
from model import Model
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
            if t == len(data) - 1:
                # When the episode is done, we don't have a next state, so we set it to the last state
                next_state = state
            else:            
                next_state = state_creator(data, t + 1, window_size + 1)
            reward = 0

            # Buy stock
            if action == 1:
                model.inventory.append(data[t])
                # print("Buy: {}".format(format_price(data[t])))

            # Sell stock
            elif action == 2 and len(model.inventory) > 0:
                bought_price = model.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                # print("Sell: {} | Profit: {}".format(format_price(data[t]), format_price(data[t] - bought_price)))

            # Hold stock
            elif action == 0:
                # print("Hold: {}".format(format_price(data[t])))
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
        model.model.save("models/model.h5")


if __name__ == "__main__":
    window_size = 10
    episodes = 10
    stock = 'AAPL'
    
    data = dataloader(stock, 'data/', '2022-09-01', '2023-03-01')
    
    # We only want the closing price
    data = data['Close'].values
    batch_size = 32

    trader = Model(window_size)
    trader.model = trader.model_builder()
    # trader.model.summary()
    # model.model.load_weights("models/model.h5")

    train_model(data, trader, window_size, episodes, batch_size)

    ### Testing the model ###

    # Load the model
    trader.model = trader.model_builder()
    trader.model.load_weights("models/model.h5")
    
    # Get the current date
    # today = datetime.today().strftime('%Y-%m-%d')
    today = '2023-03-10'

    # Get the date from a week ago
    week_ago = (datetime.strptime(today, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')

    # Get the stock closing price for the last week
    test_data = dataloader('AAPL', 'data/', week_ago, today)
    test_data = test_data['Close'].values

    # Use the model to predict the stock price for tomorrow
    state = state_creator(test_data, 0, window_size + 1)
    action = trader.trade(state)

    actions = {
        0: "Hold",
        1: "Buy",
        2: "Sell"
    }
    print("Action for {} on {}: {}".format(stock, today, actions[action]))