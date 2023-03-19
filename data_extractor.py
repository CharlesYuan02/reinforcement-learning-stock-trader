import datetime
import os
import requests
import pandas as pd
import yfinance as yf


def get_stock_names():
    '''
    Get all S&P 500 stock names from Wikipedia
    '''
    stock_names = []
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    df = pd.read_html(response.text)[0]
    for stock_name in df['Symbol']:
        stock_names.append(stock_name)
    return stock_names


def dataset_downloader(stock_name):
    '''
    Downloads stock data.
    '''
    # Replace all . with - in stock name (e.g. BRK.B -> BRK-B)
    stock_name = stock_name.replace('.', '-')
    data = yf.download(stock_name, progress=False) # progress=false to avoid printing progress bar
    return data


def create_dataset(stock_names, data_folder):
    '''
    Given an array of stock tickers, create a csv for the last 5 years of data
    for each stock and save it the data folder.
    '''
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    stock_count = 0
    for stock_name in stock_names:
        stock_count += 1
        try:
            print("Downloading data for " + stock_name + " [" + str(stock_count) + "/" + str(len(stock_names)) + "]")
            data = dataset_downloader(stock_name)
            data.to_csv(data_folder + stock_name + '.csv')
        except Exception as e:
            print(e)


def update_dataset(stock_names, data_folder, force=False):
    '''
    Given an array of stock tickers and a data folder, 
    append today's data to the end of the csv for each stock.
    '''
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    stock_count = 0
    for stock_name in stock_names:
        stock_count += 1

        # Get the date of the latest row of data
        try:
            last_date = pd.read_csv(data_folder + stock_name + '.csv').tail(1)['Date'].values[0]
            last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except:
            print("Error getting last date for " + stock_name + " [" + str(stock_count) + "/" + str(len(stock_names)) + "]")

        # Check if today's date is greater than the last date
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        if today <= last_date:
            print("Data for " + stock_name + " is up to date.")
            continue

        # If it's not, download the latest data and append it to the csv
        try:
            print("Updating data for " + stock_name + " [" + str(stock_count) + "/" + str(len(stock_names)) + "]")

            # Start date is last_date + one day, end date is tomorrow
            start_date = datetime.datetime.strptime(last_date, '%Y-%m-%d') + datetime.timedelta(days=1)
            end_date = datetime.datetime.today() + datetime.timedelta(days=1)
            data = yf.download(stock_name, start=start_date, end=end_date)

            # Append to existing csv
            with open(data_folder + stock_name + '.csv', mode='a') as f:
                for index, row in data.iterrows():
                    # Convert index to date string with format YYYY-MM-DD
                    index = index.strftime('%Y-%m-%d')
                    f.write(str(index) + ',' + str(row['Open']) + ',' + str(row['High']) + ',' + str(row['Low']) + ',' + str(row['Close']) + ',' + str(row['Adj Close']) + ',' + str(int(row['Volume'])))
                    f.write("\n")
        except Exception as e:
            print("Error updating data for " + stock_name + " [" + str(stock_count) + "/" + str(len(stock_names)) + "]")
            print(e)


def dataloader(stock_name, data_folder, start_date, end_date):
    '''
    Given a stock ticker, data folder, start date and end date, load the data
    from the csv file and return it as a pandas dataframe.
    It loads data from start_date to end_date inclusive, unless end_date is past the latest date.
    '''
    with open(data_folder + stock_name + '.csv', mode='r') as f:
        data = pd.read_csv(f)
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        return data


def label_buy_sell_hold(data, threshold=0.02):
    '''
    Given a column of values, calculate for each value the average of the next 5 values (1 week).
    If the value is [threshold] greater than the current price, append 1 for buy to the labels array.
    If the value is [threshold] less than the current price, append 2 for sell to the labels array.
    Otherwise, append 0 for hold.
    You can change the threshold to fit your trading risk tolerance.
    '''
    # Convert data from (124, 1) to (124,)
    data = data.reshape(-1)
    labels = []
    for i in range(len(data)):
        if i + 5 < len(data):
            avg = data[i+1:i+6].mean()
            if avg > data[i] * 1 + threshold:
                labels.append(1)
            elif avg < data[i] * 1 - threshold:
                labels.append(2)
            else:
                labels.append(0)
        else: # If the last 5 values are not available, append whatever the last value is
            labels.append(labels[-1])
    
    return labels # Labels is a list of 0, 1 or 2


if __name__ == "__main__":
    stock_names = get_stock_names()
    # print(len(stock_names))
    # print(stock_names)

    # This will download the entire history for each stock and save it to the data folder
    create_dataset(stock_names, 'data/')

    # This is the function you run daily to update the dataset with new data
    # update_dataset(stock_names, 'data/', force=True)

    # This is how you load the data
    data = dataloader('AAPL', 'data/', '2023-03-07', '2023-03-10')
    print(data)