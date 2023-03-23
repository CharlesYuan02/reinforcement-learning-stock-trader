import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
from scipy.special import expit
from tqdm import tqdm
from data_extractor import dataloader
from single_feature_model import Model
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
                reward = max(data[t] - bought_price, 0) # reward >= 0
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
            
            # Once we have enough data in memory, we can start training the model in batches
            if len(model.memory) > batch_size:
                model.batch_train(batch_size)
        
        print("Total Profit: {}".format(format_price(total_profit)))
        print("Saving model...")
        model.model.save(f"models/{stock.lower()}.h5")


def test_model(data, model, window_size, stock, start_date, end_date):
    '''
    Test the trained model by having it trade for a set test period.    
    For this, we don't use the memory, and we don't need the reward.
    '''
    state = state_creator(data, 0, window_size + 1)
    total_profit = 0
    model.inventory = []
    profits = []

    for t in range(len(data)):
        print("Timestep: {}/{}".format(t, len(data)))
        action = model.trade(state, is_eval=True)
        print("Action: {}".format(action))
        if t == len(data) - 1:
            # When the episode is done, we don't have a next state, so we set it to the last state
            next_state = state
        else:            
            next_state = state_creator(data, t + 1, window_size + 1)

        # Buy stock
        if action == 1:
            model.inventory.append(data[t])
            print("Buy: {}".format(format_price(data[t])))

        # Sell stock
        elif action == 2 and len(model.inventory) > 0:
            bought_price = model.inventory.pop(0) 
            total_profit += data[t] - bought_price
            print("Sell: {} | Profit: {}".format(format_price(data[t]), format_price(data[t] - bought_price)))

        # Hold stock
        elif action == 0:
            print("Hold: {}".format(format_price(data[t])))
            pass
        state = next_state

        # Save the profit for each timestep
        profits.append(total_profit)
    print(profits)
    print("Overall Profit Over Testing Period: {}".format(format_price(total_profit)))

    # Use matplotlib to plot the profit over time
    plt.plot(profits)
    plt.xlabel('Time (Days)')
    plt.ylabel('Profit (USD)')
    plt.title(f'Profit Over Time for {stock} From {start_date} to {end_date}')
    plt.legend([f'{stock}'])
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{stock.lower()}.png')
    plt.show()


if __name__ == "__main__":
    window_size = 10
    episodes = 10
    stock = 'TSLA'
    
    data = dataloader(stock, 'data/', '2022-01-01', '2023-01-01')
    
    # We only want the closing price
    data = data['Close'].values
    batch_size = 32

    trader = Model(window_size)
    trader.model = trader.model_builder()
    # trader.model.summary()

    train_model(data, trader, window_size, episodes, batch_size)

    ### Testing the model ###

    # Load the model
    trader = Model(window_size)
    trader.model = trader.model_builder()
    trader.model.load_weights(f"models/{stock.lower()}.h5")

    # Set start and end date for testing period
    test_start = '2023-01-02'
    test_end = '2023-03-02'
    test_data = dataloader(stock, 'data/', test_start, test_end)
    test_data = test_data['Close'].values

    # Test the model
    test_model(test_data, trader, window_size, stock, test_start, test_end)
    quit()

    ### Testing the model to predict the signal for tomorrow ###

    # Get the current date
    today = datetime.today().strftime('%Y-%m-%d')

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