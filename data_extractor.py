import datetime
import os
import pandas_datareader
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


def dataset_loader(stock_name):
    '''
    Load stock data.
    '''
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
            data = dataset_loader(stock_name)
            data.to_csv(data_folder + stock_name + '.csv')
        except Exception as e:
            print(e)


def update_dataset(stock_names, data_folder):
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
        last_date = pd.read_csv(data_folder + stock_name + '.csv').tail(1)['Date'].values[0]
        last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d').strftime('%Y-%m-%d')

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


if __name__ == "__main__":
    stock_names = get_stock_names()
    print(len(stock_names))
    print(stock_names)

    # This will download the entire history for each stock and save it to the data folder
    create_dataset(stock_names, 'data/')

    # This is the function you run daily to update the dataset with new data
    # update_dataset(stock_names, 'data/')