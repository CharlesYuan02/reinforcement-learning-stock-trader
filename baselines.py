import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def buy_and_sell(stocks, start_date, end_date, starting_balance=100000):
    '''
    This is the baseline where the account balance is allocated towards a group of stocks
    at the beginning of the period and holds them until the end. It then sells them, returning a final profit,
    and the value of the portfolio is plotted over time.    
    '''
    dfs = []
    for stock in stocks:
        df = pd.read_csv(f'data/{stock}.csv')
        if end_date is None or end_date == "Present":
            df = df[df["Date"] >= start_date]
            end_date = "Present"
        else:
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

        # We only need closing prices
        df = df[["Date", "Close"]]
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df.set_index("Date", inplace=True)
        dfs.append(df)

    # Allocate equal amount of money to each stock
    num_stocks = len(dfs)
    num_shares = [starting_balance / num_stocks / df["Close"][0] for df in dfs]
    num_shares = np.array(num_shares)

    # Calculate the value of the portfolio at each time step
    portfolio_value = []
    for i in range(len(dfs[0])):
        value = 0
        for j in range(num_stocks):
            value += dfs[j]["Close"][i] * num_shares[j]
        portfolio_value.append(value)

    # Plot the value of the portfolio over time
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(portfolio_value)), portfolio_value)
    plt.title(f"Portfolio Value")
    plt.xlabel(f"Day {start_date} - {end_date}")
    plt.ylabel("Portfolio Value ($)")
    plt.show()

    # Return the final profit
    return portfolio_value[-1] - starting_balance


if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"] 
    start_test = "2023-01-01"
    end_test = None
    profit = buy_and_sell(stocks, start_test, end_test)
    print(f"Profit: {profit}")