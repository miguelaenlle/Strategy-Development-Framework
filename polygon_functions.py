import os
import sys
import math
import pandas as pd
import numpy as np
import json, requests
from joblib import Parallel, delayed

from tqdm import tqdm
from scipy.signal import find_peaks
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import yfinance as yf
from zigzag import *
import talib as ta
def get_market_cap_for_stock(stock, date):
    try:
        print(date)
        date = pd.Timestamp(date)
        date = pd.Timestamp(date.date())
#         split_pct = get_stock_splits(stock, date)
        print(date)
        url = 'https://api.polygon.io/v2/reference/financials/{}?type=Q&apiKey=AKMZJGOJ8KO8NB5P32VG'.format(stock)
        print(url)
        data = json.loads(requests.get(url).text)

        results = pd.DataFrame(data['results'])
        results['calendarDate'] = pd.DatetimeIndex(results['calendarDate'])
        full_datas = {}
        for col in ['marketCapitalization', 'currentLiabilities', 'shares', 'cashAndEquivalents', 'cashAndCashEquivalentsUSD', 'operatingExpenses', 'currentRatio', 'priceToEarningsRatio', 'earningsPerBasicShare', 'debtToEquityRatio', 'debt', 'debtCurrent', 'interestExpense', 'revenues', 'priceSales']:
            try:

                column_data = results.set_index('calendarDate')[col]
        #         column_data = get_market_cap_for_stock(stock, date)
                column_data = column_data.reset_index()
#                 print(column_data['calendarDate'])
    #             return column_data, date
                column_data['distance'] = np.abs(pd.DatetimeIndex(column_data['calendarDate']) - date)

                full_datas.update({col: column_data.sort_values(by = 'distance')[col].values[0]})

            except Exception as e:
                print(e)
                full_datas.update({col: np.nan})
        return full_datas
    except Exception as e:
        print(stock, date)
        print(e)
        full_datas = {}
        for col in ['marketCapitalization', 'currentLiabilities', 'shares', 'cashAndEquivalents', 'cashAndCashEquivalentsUSD', 'operatingExpenses', 'currentRatio', 'priceToEarningsRatio', 'earningsPerBasicShare', 'debtToEquityRatio', 'debt', 'debtCurrent', 'interestExpense', 'revenues', 'priceSales']:
            full_datas.update({col: np.nan})
        return full_datas

def get_data_for_stock(stock, date, tz_origin = 'America/Chicago', tz_to_convert = 'America/New_York', token = 'AKMZJGOJ8KO8NB5P32VG'):
    try:
        date = pd.Timestamp(date)
        date = pd.Timestamp(date.date())
        start_date = date
        end_date = date
        request = 'https://api.polygon.io/v2/aggs/ticker/{}/range/1/minute/{}/{}?unadjusted=true&sort=asc&apiKey={}'.format(stock.upper(), str(start_date.date()), str(start_date.date()), token)
        raw_data = json.loads(requests.get(request).text)
        raw_data = pd.DataFrame(raw_data['results'])
        raw_data['Timestamp'] = [pd.Timestamp(datetime.datetime.fromtimestamp(t/1000)) for t in raw_data['t']]
        raw_data['Hour'] = [date.hour for date in raw_data['Timestamp']]
        raw_data['Minute'] = [date.minute for date in raw_data['Timestamp']]
        raw_data = raw_data[['v', 'o', 'c', 'h', 'l', 'Timestamp']]
        raw_data.columns = ['Volume', 'Open', 'Close', 'High', 'Low', 'Ts']
        raw_data = raw_data.set_index('Ts')
        return raw_data
    except Exception as e:
#         print(e)

#         print('No data found for {} on {}'.format(stock, start_date))
        pass
def get_data_for_stock5m(stock, date, tz_origin = 'America/Chicago', tz_to_convert = 'America/New_York', token = 'AKMZJGOJ8KO8NB5P32VG'):
    try:
        date = pd.Timestamp(date)
        date = pd.Timestamp(date.date())
        start_date = date
        end_date = date
        request = 'https://api.polygon.io/v2/aggs/ticker/{}/range/5/minute/{}/{}?unadjusted=true&sort=asc&apiKey={}'.format(stock.upper(), str(start_date.date()), str(start_date.date()), token)
        raw_data = json.loads(requests.get(request).text)
        raw_data = pd.DataFrame(raw_data['results'])
        raw_data['Timestamp'] = [pd.Timestamp(datetime.datetime.fromtimestamp(t/1000)) for t in raw_data['t']]
        raw_data['Hour'] = [date.hour for date in raw_data['Timestamp']]
        raw_data['Minute'] = [date.minute for date in raw_data['Timestamp']]
        raw_data = raw_data[['v', 'o', 'c', 'h', 'l', 'Timestamp']]
        raw_data.columns = ['Volume', 'Open', 'Close', 'High', 'Low', 'Ts']
        raw_data = raw_data.set_index('Ts')
        return raw_data
    except Exception as e:
#         print(e)

#         print('No data found for {} on {}'.format(stock, start_date))
        pass
def get_full_data_for_stock(stock, start_date, end_date):
    try:
        start_date, end_date = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
        start_date = pd.Timestamp(start_date, tz = 'America/New_York')
        end_date = pd.Timestamp(end_date, tz = 'America/New_York')
        date_range = pd.date_range(start_date, end_date)
        stock_datas = []
        for date in date_range:
            try:
                stock_data = get_data_for_stock(stock, date)
                if len(stock_data) > 0:
                    stock_datas.append(stock_data)
            except Exception as e:
#                 print(e)

                pass
        stock_datas = pd.concat(stock_datas)
        return stock_datas
        stock_datas.index = [ts.tz_localize('America/Chicago').tz_convert('UTC') for ts in stock_datas.index]
        try:
            stock_datas = stock_datas.reindex(pd.date_range(stock_datas.index[0], stock_datas.index[-1], freq = 'min'))
        except Exception as e:
#             print(e)
            pass
        stock_datas.index = [ts.tz_convert('America/New_York') for ts in stock_datas.index]
        stock_datas['Close'] = stock_datas['Close'].ffill()
        stock_datas['Volume'] = stock_datas['Volume'].fillna(0.0)

        # stock_datas.drop('index', axis = 1, inplace = True)
        stock_datas = stock_datas.T.bfill().ffill().T
        return stock_datas
    except Exception as e:
#         print(e)
        pass
def get_full_data_for_stock5m(stock, start_date, end_date):
    try:
        start_date, end_date = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
        start_date = pd.Timestamp(start_date, tz = 'America/New_York')
        end_date = pd.Timestamp(end_date, tz = 'America/New_York')
        date_range = pd.date_range(start_date, end_date)
        stock_datas = []
        for date in date_range:
            try:
                stock_data = get_data_for_stock5m(stock, date)
                if len(stock_data) > 0:
                    stock_datas.append(stock_data)
            except Exception as e:
#                 print(e)

                pass
        stock_datas = pd.concat(stock_datas)
        return stock_datas
        stock_datas.index = [ts.tz_localize('America/Chicago').tz_convert('UTC') for ts in stock_datas.index]
        try:
            stock_datas = stock_datas.reindex(pd.date_range(stock_datas.index[0], stock_datas.index[-1], freq = 'min'))
        except Exception as e:
#             print(e)
            pass
        stock_datas.index = [ts.tz_convert('America/New_York') for ts in stock_datas.index]
        stock_datas['Close'] = stock_datas['Close'].ffill()
        stock_datas['Volume'] = stock_datas['Volume'].fillna(0.0)

        # stock_datas.drop('index', axis = 1, inplace = True)
        stock_datas = stock_datas.T.bfill().ffill().T
        return stock_datas
    except Exception as e:
#         print(e)
        pass
import json, requests
def get_data_for_stock60(stock, date, tz_origin = 'America/Chicago', tz_to_convert = 'America/New_York', token = 'AKMZJGOJ8KO8NB5P32VG'):
    try:
        date = pd.Timestamp(date)
        date = pd.Timestamp(date.date())
        start_date = date
        end_date = date
        request = 'https://api.polygon.io/v2/aggs/ticker/{}/range/60/minute/{}/{}?unadjusted=true&sort=asc&apiKey={}'.format(stock.upper(), str(start_date.date()), str(start_date.date()), token)
        raw_data = json.loads(requests.get(request).text)
        raw_data = pd.DataFrame(raw_data['results'])
        raw_data['Timestamp'] = [pd.Timestamp(datetime.datetime.fromtimestamp(t/1000)) for t in raw_data['t']]
        raw_data['Hour'] = [date.hour for date in raw_data['Timestamp']]
        raw_data['Minute'] = [date.minute for date in raw_data['Timestamp']]
        raw_data = raw_data[['v', 'o', 'c', 'h', 'l', 'Timestamp']]
        raw_data.columns = ['Volume', 'Open', 'Close', 'High', 'Low', 'Ts']
        raw_data = raw_data.set_index('Ts')
        return raw_data
    except Exception as e:
#         print(e)

#         print('No data found for {} on {}'.format(stock, start_date))
        pass
def get_full_data_for_stock60(stock, start_date, end_date):
    try:
        start_date, end_date = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
        start_date = pd.Timestamp(start_date, tz = 'America/New_York')
        end_date = pd.Timestamp(end_date, tz = 'America/New_York')
        date_range = pd.date_range(start_date, end_date)
        stock_datas = []
        for date in date_range:
            try:
                stock_data = get_data_for_stock60(stock, date)
                if len(stock_data) > 0:
                    stock_datas.append(stock_data)
            except Exception as e:
#                 print(e)

                pass
        stock_datas = pd.concat(stock_datas)
        return stock_datas
        stock_datas.index = [ts.tz_localize('America/Chicago').tz_convert('UTC') for ts in stock_datas.index]
        try:
            stock_datas = stock_datas.reindex(pd.date_range(stock_datas.index[0], stock_datas.index[-1], freq = 'min'))
        except Exception as e:
#             print(e)
            pass
        stock_datas.index = [ts.tz_convert('America/New_York') for ts in stock_datas.index]
        stock_datas['Close'] = stock_datas['Close'].ffill()
        stock_datas['Volume'] = stock_datas['Volume'].fillna(0.0)

        # stock_datas.drop('index', axis = 1, inplace = True)
        stock_datas = stock_datas.T.bfill().ffill().T
        return stock_datas
    except Exception as e:
#         print(e)
        pass
def get_level(i):
    try:
#         scanner_data = premarket_gapper_data.copy()
        stock = move_data.loc[i, 'stock']
        timestamp = pd.Timestamp(move_data.loc[i, 'pd_date'])
        daily_data = yf.download(stock, (timestamp - pd.Timedelta('365 days')).date(), timestamp.date()).reset_index(drop = True)
        vol_threshold = daily_data['Volume'].sort_values(ascending = False).values[:int(len(daily_data) * 0.1)][-1]
        extreme_volume_levels = scipy.signal.find_peaks(daily_data['Volume'].values, threshold = None, height = vol_threshold)[0]
        significant_prices = daily_data.loc[extreme_volume_levels]['High']

        high_values = []
        for high in daily_data['High'].unique():
            high_values.append(round(high, 1))
        high_values = np.unique(high_values)
        volumes = {}
        for high_value in high_values:
            volume_levels = daily_data[daily_data['High'].round(1) == high_value]['Volume'].sum()
            volumes.update({high_value: volume_levels})
        price_data = []
        for level in significant_prices:
            try:
                round_price = round(level, 1)
                volume_for_level = volumes[round_price]
                price_data.append([level, volume_for_level])
            except:
                current_volumes = pd.Series(volumes)
                if current_volumes.index[-1] < level:
                    volume_for_level = current_volumes[current_volumes.index <= level].values[-1]
                else:
                    volume_for_level = current_volumes[current_volumes.index >= level].values[0]
                price_data.append([level, volume_for_level])
    #         print(round_price, volume_for_level)
        price_data = pd.DataFrame(price_data, columns = ['level', 'volume_on_level'])
        price_data = price_data.sort_values(by = 'level')
        levels = [price_data['level'].values[i] for i in scipy.signal.find_peaks(price_data['volume_on_level'].values, threshold = None)[0]]
        levels = pd.Series(levels)
        levels2 = pd.DataFrame({'distance_from_volume_area': (levels - daily_data['Close'].values[-1]).abs(),
                      'level': levels})
        level = levels2.sort_values(by = 'distance_from_volume_area')['level'].values[0]

        distance_from_level = (level - daily_data['Close'].values[-1]) / daily_data['Close'].values[-1]
    #     print(distance_from_level)
        return [level, distance_from_level, levels2['level'].unique()]
    except Exception as e:
        print(e)
        return [np.nan] * 3
def get_level60(i):
#     try:
#         scanner_data = premarket_gapper_data.copy()
    stock = move_data.loc[i, 'stock']
    timestamp = pd.Timestamp(move_data.loc[i, 'pd_date'])
    stock_data60 = get_full_data_for_stock60(stock, timestamp - pd.Timedelta('10 days'), timestamp)

    stock_data60.index = [pd.Timestamp(ts).tz_localize('America/Chicago').tz_convert('America/New_York') for ts in stock_data60.index ]
    stock_data60 = stock_data60[stock_data60['Volume'] > 0]
    stock_data60 = stock_data60.reset_index()

    forward_data60 = stock_data60[stock_data60['index'] >= timestamp]
    backward_data60 = stock_data60[stock_data60['index']<= timestamp]

    stock = move_data.loc[i, 'stock']
    timestamp = pd.Timestamp(move_data.loc[i, 'pd_date'])
#         daily_data = yf.download(stock, (timestamp - pd.Timedelta('365 days')).date(), timestamp.date()).reset_index(drop = True)
    vol_threshold = backward_data60['Volume'].sort_values(ascending = False).values[:int(len(backward_data60) * 0.1)][-1]
    extreme_volume_levels = scipy.signal.find_peaks(backward_data60['Volume'].values, threshold = None, height = vol_threshold)[0]
    significant_prices = backward_data60.loc[extreme_volume_levels]['High']

    high_values = []
    for high in backward_data60['High'].unique():
        high_values.append(round(high, 2))
    high_values = np.unique(high_values)

    volumes = {}
    for high_value in high_values:
        volume_levels = backward_data60[backward_data60['High'].round(2) == high_value]['Volume'].sum()
        volumes.update({high_value: volume_levels})

#     print(volumes)

    price_data = []
    for level in significant_prices:
        try:
            round_price = round(level, 2)
            volume_for_level = volumes[round_price]
            price_data.append([level, volume_for_level])
        except:
            current_volumes = pd.Series(volumes)
            if current_volumes.index[-1] < level:
                volume_for_level = current_volumes[current_volumes.index <= level].values[-1]
            else:
                volume_for_level = current_volumes[current_volumes.index >= level].values[0]
            price_data.append([level, volume_for_level])
#         print(round_price, volume_for_level)
    price_data = pd.DataFrame(price_data, columns = ['level', 'volume_on_level'])
    price_data = price_data.sort_values(by = 'level')
    levels = [price_data['level'].values[i] for i in scipy.signal.find_peaks(price_data['volume_on_level'].values, threshold = None)[0]]
    levels = pd.Series(levels)
    levels2 = pd.DataFrame({'distance_from_volume_area': (levels - backward_data60['Close'].values[-1]).abs(),
                  'level': levels})
#     print(distance_from_level)
    return levels2
#     except Exception as e:
#         print(e)
#         return [np.nan] * 3

import json
import requests
import datetime
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import scipy.stats
def get_tick_data(stock, date):
    all_data = []
    ts = 0
    while True:
        try:
            url = 'https://api.polygon.io/v2/ticks/stocks/nbbo/{}/{}?timestamp={}&reverse=false&limit=50000&apiKey=AKMZJGOJ8KO8NB5P32VG'.format(
                stock, str(date.date()), ts)
            print(url)
            response = json.loads(requests.get(url).text)
            results = pd.DataFrame(response['results'])
            results['date'] = [pd.Timestamp(datetime.datetime.fromtimestamp(date / 1000000000)).tz_localize('America/Chicago').tz_convert('America/New_York') for date in
                               results['t']]

            if len(results) == 50_000:
                all_data.append(results)
                ts = results['t'].values[-1]
                continue
            else:
                all_data.append(results)
                print(len(results))
                break
        except Exception as e:
            print(e)
            break
    all_data = pd.concat(all_data)
    return all_data


def get_data_trades(stock, date):
    all_data = []
    ts = 0
    ix = 0
    while True:
        try:
            url = 'https://api.polygon.io/v2/ticks/stocks/trades/{}/{}?timestamp={}&reverse=false&limit=50000&apiKey=AKMZJGOJ8KO8NB5P32VG'.format(
                stock, str(date.date()), ts)
            response = json.loads(requests.get(url).text)
            print(url)
            results = pd.DataFrame(response['results'])
            results['date'] = [pd.Timestamp(datetime.datetime.fromtimestamp(date / 1000000000)) for date in
                               results['t']]

            if len(results) == 50_000:
                all_data.append(results)
                ts = results['t'].values[-1]
                continue
            else:
                all_data.append(results)
                print(len(results))
                break
            ix += 1
#             if ix < 10:
#                 break
        except Exception as e:
            print(e)
            break

    all_data = pd.concat(all_data)
    return all_data


from joblib import Parallel, delayed
plt.rcParams['figure.figsize'] = 12, 10
plt.style.use('ggplot')
def convert_conditions_to_str(condition_list):
    conditions = []
    try:
        for condition in condition_list:
            condition_name = trade_conditions[str(condition)]
            conditions.append(condition_name)
    except:
        return []
    return conditions


import json
import requests
import datetime
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


def get_tick_data(stock, date):
    all_data = []
    ts = 0
    while True:
        try:
            url = 'https://api.polygon.io/v2/ticks/stocks/nbbo/{}/{}?timestamp={}&reverse=false&limit=50000&apiKey=AKMZJGOJ8KO8NB5P32VG'.format(
                stock, str(date.date()), ts)
            print(url)
            response = json.loads(requests.get(url).text)
            results = pd.DataFrame(response['results'])
            results['date'] = [pd.Timestamp(datetime.datetime.fromtimestamp(date / 1000000000)) for date in
                               results['t']]
            # results.columns = ['SIP_ts', 'exchange_ts', 'seq_number', 'conditions', 'bid_size', 'bid_price',
            #                    'bid_exchange_id', 'ask_size', 'ask_price', 'ask_exchange_id', 'ask_size', 'indicators',
            #                    'pd_date']
            if len(results) == 50_000:
                all_data.append(results)
                ts = results['t'].values[-1]
                continue
            else:
                all_data.append(results)
                print(len(results))
                break
        except Exception as e:
            print(e)
            break
    all_data = pd.concat(all_data)
    return all_data


def get_data_trades(stock, date):
    all_data = []
    ts = 0
    ix = 0
    while True:
        try:
            url = 'https://api.polygon.io/v2/ticks/stocks/trades/{}/{}?timestamp={}&reverse=false&limit=50000&apiKey=AKMZJGOJ8KO8NB5P32VG'.format(
                stock, str(date.date()), ts)
            print(url)
            response = json.loads(requests.get(url).text)
            print(url)
            results = pd.DataFrame(response['results'])
            results['date'] = [pd.Timestamp(datetime.datetime.fromtimestamp(date / 1000000000)) for date in
                               results['t']]

            if len(results) == 50_000:
                all_data.append(results)
                ts = results['t'].values[-1]
                continue
            else:
                all_data.append(results)
                print(len(results))
                break
            ix += 1
        #             if ix < 10:
        #                 break
        except Exception as e:
            print(e)
            break

    all_data = pd.concat(all_data)
    return all_data


from joblib import Parallel, delayed


def get_ask_bids(stock, date):
    ask_bids = []
    index = []
    trade_datas = get_data_trades(stock, date).reset_index(drop=True)
    trade_datas = trade_datas[trade_datas['s'] > 2500]
    trade_datas['SIP Timestamp'] = [
        pd.Timestamp(datetime.datetime.fromtimestamp(t / 1e9), tz='America/Chicago').tz_convert('America/New_York') for
        t in
        trade_datas['t']]
    trade_datas['Exchange Timestamp'] = [
        pd.Timestamp(datetime.datetime.fromtimestamp(t / 1e9), tz='America/Chicago').tz_convert('America/New_York') for
        t in
        trade_datas['y']]

    trade_datas = trade_datas[['SIP Timestamp', 'Exchange Timestamp', 'p', 's']]
    trade_datas.columns = ['SIP Timestamp', 'Exchange Timestamp', 'Price', 'Size']

    stock_datas = get_tick_data(stock, date).reset_index(drop=True)

    stock_datas['exchange_ts'] = pd.DatetimeIndex(stock_datas['pd_date'], tz='America/Chicago')
    stock_datas = stock_datas.drop('SIP_ts', axis=1)
    stock_datas = stock_datas[['exchange_ts', 'ask_size', 'ask_price', 'bid_size', 'bid_price']]
    #     if len(trade_datas) < 25000:

    trade_datas = trade_datas[trade_datas['SIP Timestamp'] >= date]

    for i in tqdm(trade_datas.index):
        ts = trade_datas.loc[i, 'SIP Timestamp']
        try:
            price = trade_datas.loc[i, 'Price']
            size = trade_datas.loc[i, 'Size']

            #             print(ts)

            ask_price = stock_datas[stock_datas['exchange_ts'] <= ts]['ask_price'].values[-1]
            bid_price = stock_datas[stock_datas['exchange_ts'] <= ts]['bid_price'].values[-1]

            if type(ask_price) != list and type(bid_price) != list:
                index.append(ts)
                ask_bids.append([ask_price, bid_price, price, size])
        except Exception as e:
            print(e)
            continue
    #             index.append(ts)
    #             ask_bids.append([np.nan] * 3)
    ask_bids = pd.DataFrame(ask_bids, columns=['ask_price', 'bid_price', 'price', 'size'])
    ask_bids = ask_bids.reset_index(drop=True)
    ask_bids.index = index
    #     ask_bids = ask_bids.dropna()
    #     break
    # return ask_bids

    ask_bids['distance_from_ask'] = ask_bids['ask_price'] - ask_bids['price']
    ask_bids['distance_from_bid'] = ask_bids['price'] - ask_bids['bid_price']
    ask_bids['ask/bid'] = ask_bids['distance_from_ask'] / ask_bids['distance_from_bid']

    price_above_ask = ask_bids[ask_bids['ask_price'] < ask_bids['price']].index
    price_below_bid = ask_bids[ask_bids['bid_price'] > ask_bids['price']].index
    ask_bids_normal = ask_bids[ask_bids['ask/bid'] > 0]
    at_ask = ask_bids_normal[ask_bids_normal['ask/bid'] > 3].index
    at_bid = ask_bids_normal[ask_bids_normal['ask/bid'] < 1 / 3].index
    inside = ask_bids_normal[(ask_bids_normal['ask/bid'] <= 1 / 3) & (ask_bids_normal['ask/bid'] <= 3)].index

    type_index = 0
    types = ['Above', 'AtAsk', 'Inside', 'AtBid', 'Below']
    ask_bids['type'] = np.nan
    ask_bids['type_code'] = np.nan
    for index in [price_above_ask, at_ask, inside, at_bid, price_below_bid]:
        price_type = types[type_index]
        ask_bids['type'].loc[index] = price_type
        ask_bids['type_code'].loc[index] = type_index
        type_index += 1

    ask_bids['time_change'] = [t.seconds for t in pd.Series(ask_bids.index).diff()]
    return ask_bids


def convert_conditions_to_str(condition_list):
    conditions = []
    try:
        for condition in condition_list:
            condition_name = trade_conditions[str(condition)]
            conditions.append(condition_name)
    except:
        return []
    return conditions


def get_424_data(stock, date):
    if stock + '.csv' in os.listdir('data/424_upgraded'):
        _424_data = pd.read_csv('data/424_upgraded/{}.csv'.format(stock), index_col=0)
        if len(_424_data) > 0:
            _424_data['date'] = pd.DatetimeIndex(_424_data['date'], tz='America/New_York')
            _424_data['s3_parent_release'] = pd.DatetimeIndex(_424_data['s3_parent_release'], tz='America/New_York')
            s3_data = pd.read_csv('data/s3_upgraded/{}.csv'.format(stock), index_col=0)

            def get_expiration_dates(s3_data):
                #     s3_data['date_cancelled'] = np.nan
                s3_data['date'] = pd.DatetimeIndex(s3_data['date'])
                for i in s3_data.index:
                    #         date_cancelled = s3_data.loc[i, 'date_cancelled']
                    date_cancelled = s3_data.loc[i, 'date_cancelled']

                    file_type = s3_data.loc[i, 'filing-types']
                    #                     print(date_cancelled)
                    try:
                        if math.isnan(date_cancelled) == True:
                            if 'S-3' in file_type:
                                s3_data.loc[i, 'date_cancelled'] = pd.Timestamp(s3_data.loc[i, 'date']) + pd.Timedelta(
                                    '1095 days')
                    except:
                        s3_data.loc[i, 'date_cancelled'] = pd.Timestamp(date_cancelled)
                return s3_data

            s3_data = get_expiration_dates(s3_data)
            previous_424s = _424_data[_424_data['date'] < date]
            usable = []
            for i in previous_424s.index:
                sec_id = previous_424s.loc[i, 'id']
                try:
                    end_of_effectiveness = s3_data[s3_data['id'] == sec_id]['date_cancelled'].values[0].tz_localize(
                        'America/New_York')

                    if date < end_of_effectiveness:
                        usable.append(True)
                    else:
                        usable.append(False)
                except Exception as e:
                    usable.append(False)
            previous_424s['usable'] = usable
            return previous_424s[previous_424s['usable']]
    return []
import pandas as pd
def get_market_cap_for_stock(stock, date):
    try:
        import pandas as pd
        print(date)
        date = pd.Timestamp(date)
        date = pd.Timestamp(date.date())
        print(date)
        url = 'https://api.polygon.io/v2/reference/financials/{}?type=Q&apiKey=AKMZJGOJ8KO8NB5P32VG'.format(stock)
        print(url)
        data = json.loads(requests.get(url).text)

        results = pd.DataFrame(data['results'])
        results['calendarDate'] = pd.DatetimeIndex(results['calendarDate'])
        full_datas = {}
        true_cols = ['Market Capitalization', 'Current Liabilities', 'Shares', 'Cash', 'Operating Expenses', 'Current Ratio', 'PE Ratio', 'Earnings/Share', 'Debt/Equity', 'Debt', 'Current Debt', 'Interest Expense', 'Revenue', 'Price/Sales']
        idx = 0
        for col in ['marketCapitalization', 'currentLiabilities', 'shares', 'cashAndEquivalents', 'operatingExpenses', 'currentRatio', 'priceToEarningsRatio', 'earningsPerBasicShare', 'debtToEquityRatio', 'debt', 'debtCurrent', 'interestExpense', 'revenues', 'priceSales']:
            try:

                column_data = results.set_index('calendarDate')[col]
        #         column_data = get_market_cap_for_stock(stock, date)
                column_data = column_data.reset_index()
#                 print(column_data['calendarDate'])
    #             return column_data, date
                column_data['distance'] = np.abs(pd.DatetimeIndex(column_data['calendarDate']) - date)

                full_datas.update({true_cols[idx]: column_data.sort_values(by = 'distance')[col].values[0]})
            except Exception as e:
                print(e)
                full_datas.update({col: np.nan})
            idx += 1
        return full_datas
    except Exception as e:
        print(stock, date)
        print(e)
        full_datas = {}
        for col in ['Market Capitalization', 'Current Liabilities', 'Shares', 'Cash', 'Operating Expenses', 'Current Ratio', 'PE Ratio', 'Earnings/Share', 'Debt/Equity', 'Debt', 'Current Debt', 'Interest Expense', 'Revenue', 'Price/Sales']:
            full_datas.update({col: np.nan})
        return full_datas
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from bs4 import BeautifulSoup
    from joblib import Parallel, delayed
    import time
    def get_short_interest(stock):

        options = Options()
        options.headless = True

        driver = webdriver.Firefox(options=options)
        try:
            driver.get('https://www.nasdaqtrader.com/Trader.aspx?id=ShortInterest')
            driver.find_element_by_xpath('//*[@id="searchSymbol"]').send_keys(stock)
            driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/div[1]/div/p[1]/span[1]/a').click()
            time.sleep(3)
            page = BeautifulSoup(driver.page_source)
            gentable = page.find('div', {'class': "genTable"})
            gentable = gentable.text

            driver.quit()

            data = gentable.split('\n\n')
            all_lines = []
            for line in data:

                if len(line) > 0:
                    all_lines.append(line.splitlines()[1:])
            all_lines = pd.DataFrame(all_lines[1:], columns=all_lines[0])
            return all_lines
        except:
            driver.quit()
            return []


import json, requests


def get_stock_splits(stock, date):
    try:
        url = 'https://api.polygon.io/v2/reference/splits/{}?apiKey=AKMZJGOJ8KO8NB5P32VG'.format(stock)
        data = json.loads(requests.get(url).text)['results']
        split_adjusted_multiplier = 1
        data = pd.DataFrame(data)
        data['exDate'] = pd.DatetimeIndex(data['exDate'], tz='America/New_York')

        data = data[data['exDate'] < date]
        print(data)
        for ratio in data['ratio'].values:
            split_adjusted_multiplier *= ratio
        return 1 / split_adjusted_multiplier
    except Exception as e:
        print(e)
        return 1

def get_424_data_processed(stock, date):
    date = pd.Timestamp(date, tz = 'America/Chicago').tz_convert('America/New_York')
    _424s = get_424_data(stock, date)
    if len(_424s) > 0:
        atms = _424s[_424s['money'].notna()]
        num_previous_atms = len(atms)

        y1_date = date - pd.Timedelta('365 days')

        last_year_424s = _424s[_424s['date'] > y1_date]

        qt_424s = len(last_year_424s)
        qt_atms = len(last_year_424s[last_year_424s['money'].notna()])

        previous_shares = last_year_424s['shares'].dropna()[:-1].sum()
        previous_atms = last_year_424s['money'].dropna()[:-1].sum()
        try:
            current_offering = _424s['shares'].dropna().values[-1]
        except:
            current_offering = 0.0
        try:
            current_offering_cash = _424s['money'].dropna().values[-1]
        except:
            current_offering_cash = 0.0

        atm_present = len(atms)
    else:
        num_previous_atms = 0
        qt_atms = 0
        atms = []
        qt_424s = []
        previous_shares = 0.0
        previous_atms = 0.0
        current_offering = 0.0
        current_offering_cash = 0.0
        atm_present = 0
    return {'Num. Active ATMs (all time)': num_previous_atms,
            'Num. Active 424s (all time)': len(_424s),
            'Num. Active 424B5s (past year)' : qt_atms,
            'Num. Active 424B3s (past year)': qt_424s,
            'Shares Raised in the Past Year': previous_shares,
            'Dollars Raised in the Past Year': previous_atms,
            'Current Offering (shares)': current_offering,
            'Current Offering (cash)': current_offering_cash,
            'ATM Present': atm_present > 0,
            'Offering Present': len(_424s) > 0}


def get_data_PM(stock, date):
    try:
        date = pd.Timestamp(date, tz='America/New_York')
        stock_data = get_full_data_for_stock(stock, date - pd.Timedelta('5 days'), date)

        stock_data = stock_data[stock_data['Volume'] > 0]
        stock_data['Date'] = [pd.Timestamp(date).date() for date in stock_data.index]
        stock_data['Hour'] = [pd.Timestamp(date).hour for date in stock_data.index]
        stock_data['Minute'] = [pd.Timestamp(date).minute for date in stock_data.index]

        stock_data2 = stock_data[(stock_data['Hour'] > 9) | ((stock_data['Hour'] == 9) & (stock_data['Minute'] >= 30))]
        stock_data2 = stock_data2[stock_data2['Date'] == date]
        stock_data2 = stock_data2[stock_data2['Hour'] < 16]
        change1m = stock_data2['Open'].pct_change()[1]
        change5m = stock_data2['Open'].pct_change(5)[5]

        premarket_data = stock_data[stock_data['Date'] == pd.Series(stock_data['Date'].unique()).max()]
        afterhours_data = stock_data[stock_data['Date'] == pd.Series(stock_data['Date'].unique()).max()]

        premarket_data = premarket_data[
            (premarket_data['Hour'] < 9) | ((premarket_data['Hour'] == 9) & (premarket_data['Minute'] < 30))]
        afterhours_data = afterhours_data[afterhours_data['Hour'] > 16]

        date_data = pd.concat([premarket_data, afterhours_data])
        print(date_data)
        date_data = date_data.reset_index().sort_values(by='Ts').set_index('Ts')

        if len(date_data) > 0:
            #             return date_data
            high_change = (date_data['High'].max() - date_data['Open'].values[0]) / date_data['Open'].values[0]
            detection_ts = date_data[date_data['High'] >= date_data['Open'].values[0] * 1.15].index[0]
            #             plt.plot(date_data[date_data['Date'] == date_data['Date'].values[-1]]['Open'])
            #             plt.axvline(detection_ts)
            #             plt.show()

            return high_change, detection_ts, change1m, change5m, premarket_data['Volume'].sum()
        else:
            return [np.nan] * 5
    except Exception as e:
        print(e)
        return [np.nan] * 5


# move_data = pm_gaps_only.copy()  # [pm_gaps_only['High Change'] > 0.25]


# for i in tqdm(move_data.index[::-1][20:]):
def fits_setup_criteria(move_data, i):
    try:
        high_change = move_data.loc[i, 'High Change']
        stock = move_data.loc[i, '0']
        timestamp = pd.Timestamp(move_data.loc[i, 'Detection TS']).tz_localize('America/Chicago').tz_convert(
            'America/New_York')
        date = timestamp
        mktopen = pd.Timestamp(year=date.year,
                               month=date.month,
                               day=date.day,
                               hour=9,
                               minute=30,
                               tz='America/New_York')
        pmopen = pd.Timestamp(year=date.year,
                              month=date.month,
                              day=date.day,
                              hour=4,
                              minute=0,
                              tz='America/New_York')
        timestamp = mktopen
        stock_data = get_full_data_for_stock(stock, timestamp - pd.Timedelta('5 days'),
                                             timestamp)  # get_data_for_stock(stock, timestamp)
        stock_data.index = [pd.Timestamp(ts).tz_localize('America/Chicago').tz_convert('America/New_York') for ts in
                            stock_data.index]
        stock_data = stock_data[stock_data['Volume'] > 0]
        stock_data = stock_data.reset_index()

        stock_data['vwap'] = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
        stock_data['ewm200'] = stock_data['Close'].ewm(200).mean()
        idx = 0

        forward_data = stock_data[stock_data['index'] >= mktopen]
        true_forward_data = forward_data.copy()
        backward_data = stock_data[stock_data['index'] <= mktopen]

        pm_open = stock_data.loc[stock_data[stock_data['index'] >= pmopen].index[0], 'Open']
        open_price = forward_data['Open'].values[0]

        #         plt.axvspan(stock_data[stock_data['index'] >= pmopen].index[0], stock_data[stock_data['index'] >= mktopen].index[0], alpha = 0.25)
        #         plt.plot(stock_data[stock_data['index'] >= pmopen].index[0], stock_data.loc[stock_data[stock_data['index'] >= pmopen].index[0], 'Open'], 'ro')
        #         plt.plot(stock_data[stock_data['index'] >= mktopen].index[0], stock_data.loc[stock_data[stock_data['index'] >= mktopen].index[0], 'Open'], 'ro')

        pm_data = stock_data[(stock_data['index'] >= pmopen) & (stock_data['index'] <= mktopen)]
        high_price = pm_data['Close'].max()

        high_index = pm_data[pm_data['Close'] == high_price].index[0]
        high_length = len(pm_data['Close'].loc[:high_index])
        retrace_length = len(pm_data) - high_length

        change_threshold = high_change / 5
        change_threshold = max(change_threshold, 0.1)
        pivots = peak_valley_pivots(pm_data['Open'].values, change_threshold, -change_threshold)
        plt.plot(pm_data[pivots != 0]['Open'], color='black')
        #         plt.show()

        pm_data['pivot'] = pivots
        trend_data = pm_data[pm_data['pivot'] != 0]
        #         trend_data = trend_data[trend_data]
        #         trend_data.loc[trend_data.index[-1], 'pivot'] = -trend_data.loc[trend_data.index[-1], 'pivot']
        trend_data = trend_data.reset_index()
        trend_datas = []
        for i in trend_data.index[1:]:
            pivot = trend_data.loc[i, 'pivot']
            start_index = trend_data.loc[i - 1, 'level_0']
            end_index = trend_data.loc[i, 'level_0']

            previous_index = trend_data.loc[i - 1, 'index']
            previous_price = trend_data.loc[i - 1, 'Open']

            current_index = trend_data.loc[i, 'index']
            current_price = trend_data.loc[i, 'Open']
            if current_price < previous_price:
                pivot = 'Downtrend'
            else:
                pivot = 'Uptrend'
            trend_length = (current_index - previous_index).total_seconds() / 60

            move_percent = (current_price - previous_price) / previous_price
            #             print(pivot)
            trend_datas.append([trend_length, move_percent, pivot, pm_open, open_price])
        trend_datas = pd.DataFrame(trend_datas,
                                   columns=['Trend Length (m)', 'Move Percent', 'Pivot', 'PM Open', 'Open Price'])
        return trend_datas, open_price, pm_open, high_price, high_length, retrace_length
    except Exception as e:
        print(e)
        return pd.DataFrame([], columns=['Trend Length (m)', 'Move Percent', 'Pivot', 'PM Open', 'Open Price'])


def get_trend_data(move_data, i):
    # for i in tqdm(move_data[move_data['Volume PM'] > 1e6].index[12:30]):
    try:
        high_change = move_data.loc[i, 'High Change']
        stock = move_data.loc[i, '0']
        timestamp = pd.Timestamp(move_data.loc[i, 'Detection TS']).tz_localize('America/Chicago').tz_convert(
            'America/New_York')
        date = timestamp
        mktopen = pd.Timestamp(year=date.year,
                               month=date.month,
                               day=date.day,
                               hour=9,
                               minute=30,
                               tz='America/New_York')
        pmopen = pd.Timestamp(year=date.year,
                              month=date.month,
                              day=date.day,
                              hour=4,
                              minute=0,
                              tz='America/New_York')
        timestamp = mktopen
        stock_data = get_full_data_for_stock(stock, timestamp - pd.Timedelta('5 days'),
                                             timestamp)  # get_data_for_stock(stock, timestamp)
        stock_data.index = [pd.Timestamp(ts).tz_localize('America/Chicago').tz_convert('America/New_York') for ts in
                            stock_data.index]
        stock_data = stock_data[stock_data['Volume'] > 0]
        stock_data = stock_data.reset_index()

        stock_data['vwap'] = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
        stock_data['ewm200'] = stock_data['Close'].ewm(200).mean()
        idx = 0

        forward_data = stock_data[stock_data['index'] >= mktopen]
        true_forward_data = forward_data.copy()
        backward_data = stock_data[stock_data['index'] <= mktopen]
        fig, (ax1, ax2) = plt.subplots(2)
        plt.title('{} on {}'.format(stock, timestamp))
        ax1.axvspan(stock_data[stock_data['index'] >= pmopen].index[0],
                    stock_data[stock_data['index'] >= mktopen].index[0], alpha=0.25)
        #         ax1.plot(stock_data[stock_data['index'] >= pmopen].index[0], stock_data.loc[stock_data[stock_data['index'] >= pmopen].index[0], 'Open'], 'ro')
        #         ax1.plot(stock_data[stock_data['index'] >= mktopen].index[0], stock_data.loc[stock_data[stock_data['index'] >= mktopen].index[0], 'Open'], 'ro')
        ax1.plot(forward_data['Open'])
        ax1.plot(backward_data['Open'])
        ax2.plot(forward_data['Volume'])
        ax2.plot(backward_data['Volume'])
        forward_mkt_data = stock_data[stock_data['index'] >= mktopen]
        range_of_data = (forward_mkt_data['High'].max() - forward_mkt_data['Low'].min()) / forward_mkt_data['Low'].min()

        change_threshold = range_of_data / 3
        #         change_threshold = max(change_threshold, 0.1)
        pivots = peak_valley_pivots(forward_mkt_data['Open'].values, change_threshold, -change_threshold)
        forward_mkt_data['pivot'] = pivots

        trend_data = forward_mkt_data[forward_mkt_data['pivot'] != 0]
        #         trend_data = trend_data[trend_data]
        #         trend_data.loc[trend_data.index[-1], 'pivot'] = -trend_data.loc[trend_data.index[-1], 'pivot']
        trend_data = trend_data.reset_index()
        trend_datas = []
        for i in trend_data.index[1:]:
            pivot = trend_data.loc[i, 'pivot']
            start_index = trend_data.loc[i - 1, 'level_0']
            end_index = trend_data.loc[i, 'level_0']

            previous_index = trend_data.loc[i - 1, 'index']
            previous_price = trend_data.loc[i - 1, 'Open']

            current_index = trend_data.loc[i, 'index']
            current_price = trend_data.loc[i, 'Open']
            if current_price < previous_price:
                pivot = 'Downtrend'
            else:
                pivot = 'Uptrend'
            trend_length = (current_index - previous_index).total_seconds() / 60

            move_percent = (current_price - previous_price) / previous_price
            print(pivot)
            trend_datas.append([trend_length, move_percent, pivot, start_index, end_index])
        trend_datas = pd.DataFrame(trend_datas,
                                   columns=['Trend Length (m)', 'Move Percent', 'Pivot', 'Start Index', 'End Index'])
        trend_datas['length'] = trend_datas['End Index'] - trend_datas['Start Index']
        #         trend_datas = trend_datas[trend_datas['length'] > 10]
        #         ax1.plot(forward_mkt_data[forward_mkt_data['pivot'] != 0]['Open'], '--bo', marker = 'o')
        #         ax1.axvline(forward_mkt_data[forward_mkt_data['pivot'] == -1].index[0])
        trend_datas['Pivot'][trend_datas['Move Percent'] < 0] = 'Downtrend'
        trend_datas['Pivot'][trend_datas['Move Percent'] > 0] = 'Uptrend'
        #         for i in trend_datas.index:
        #             start_index, end_index = trend_datas.loc[i, ['Start Index', 'End Index']].values
        #             if trend_datas.loc[i, 'Pivot'] == 'Downtrend':
        #                 color = 'r'
        #             else:
        #                 color = 'g'
        #             ax1.axvspan(start_index, end_index, alpha = 0.25, color = color)
        #         plt.show
        try:
            length_downtrends = trend_datas[trend_datas['Move Percent'] < 0]['length'].sum()
            max_downmove = trend_datas[trend_datas['Move Percent'] < 0]['Move Percent'].min()
        except:
            length_downtrends = 0.0
            max_downmove = 0.0

        try:
            length_uptrends = trend_datas[trend_datas['Move Percent'] > 0]['length'].sum()
            max_upmove = trend_datas[trend_datas['Move Percent'] > 0]['Move Percent'].max()
        except:
            length_uptrends = 0.0
            max_upmove = 0.0

        percentage_downtrend = length_downtrends / (length_uptrends + length_downtrends)
        return trend_datas, percentage_downtrend, max_downmove, max_upmove
    #         break
    except Exception as e:
        print(e)
        return [np.nan] * 4


# for i in tqdm(highvol_pumps.index[:12]):
def get_basic_priord_price_action_stats(move_data, i):
    try:
        #         sr_data = get_movement_data(i)
        high_change = move_data.loc[i, 'High Change']
        stock = move_data.loc[i, '0']
        timestamp = pd.Timestamp(move_data.loc[i, 'Detection TS']).tz_localize('America/Chicago').tz_convert(
            'America/New_York')
        date = timestamp
        threshold = 12500
        volume = move_data.loc[i, 'Volume PM']
        if volume > 1e6 and volume <= 5e6:
            threshold = 25000
        else:
            threshold = 35000
        mktopen = pd.Timestamp(year=date.year,
                               month=date.month,
                               day=date.day,
                               hour=9,
                               minute=30,
                               tz='America/New_York')
        pmopen = pd.Timestamp(year=date.year,
                              month=date.month,
                              day=date.day,
                              hour=4,
                              minute=0,
                              tz='America/New_York')
        stock_data = get_full_data_for_stock(stock, timestamp - pd.Timedelta('10 days'),
                                             timestamp)  # get_data_for_stock(stock, timestamp)
        stock_data.index = [pd.Timestamp(ts).tz_localize('America/Chicago').tz_convert('America/New_York') for ts in
                            stock_data.index]
        stock_data = stock_data[stock_data['Volume'] > 0]
        stock_data = stock_data.reset_index()
        stock_data['Variance'] = ta.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'])

        stock_data['hour_high'] = stock_data['Volume'].rolling(60).max() * 0.8
        stock_data['vwap'] = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
        stock_data['ewm200'] = stock_data['Close'].ewm(200).mean()
        idx = 0

        forward_data = stock_data[stock_data['index'] >= pmopen]
        true_forward_data = forward_data.copy()
        backward_data = stock_data[stock_data['index'] <= pmopen]
        lowest_value = backward_data['Close'].min()
        highest_value = backward_data['Close'].max()
        opening_value = backward_data['Close'].values[0]
        last_value = backward_data['Close'].values[-1]

        change_to_high = (highest_value - opening_value) / opening_value

        change_from_high = (last_value - highest_value) / highest_value

        change_from_open = (last_value - opening_value) / opening_value

        prior_day_range = (highest_value - lowest_value) / lowest_value

        shares_traded = backward_data['Volume'].sum()

        index_of_high = backward_data[backward_data['Close'] == highest_value].index[0]
        length_before = len(backward_data.loc[:index_of_high])
        length_after = len(backward_data.loc[index_of_high:])

        return {'Change to High': change_to_high,
                'Change from High': change_from_high,
                'Change from Open': change_from_open,
                'Prior Day Range': prior_day_range,
                'Length Prior': length_before,
                'Length After': length_after,
                'Shares Traded': shares_traded,
                'PD High': highest_value}
    except Exception as e:
        print(e)
        return {'Change to High': np.nan,
                'Change from High': np.nan,
                'Change from Open': np.nan,
                'Prior Day Range': np.nan,
                'Length Prior': np.nan,
                'Length After': np.nan,
                'Shares Traded': np.nan,
                'PD High': np.nan}
def get_basic_id_stats(move_data, i):
    try:
        #         sr_data = get_movement_data(i)
        high_change = move_data.loc[i, 'High Change']
        stock = move_data.loc[i, '0']
        timestamp = pd.Timestamp(move_data.loc[i, 'Detection TS']).tz_localize('America/Chicago').tz_convert(
            'America/New_York')
        date = timestamp
        threshold = 12500
        volume = move_data.loc[i, 'Volume PM']
        if volume > 1e6 and volume <= 5e6:
            threshold = 25000
        else:
            threshold = 35000
        mktopen = pd.Timestamp(year=date.year,
                               month=date.month,
                               day=date.day,
                               hour=9,
                               minute=30,
                               tz='America/New_York')
        pmopen = pd.Timestamp(year=date.year,
                              month=date.month,
                              day=date.day,
                              hour=4,
                              minute=0,
                              tz='America/New_York')
        timestamp = mktopen
        stock_data = get_full_data_for_stock(stock, timestamp - pd.Timedelta('5 days'),
                                             timestamp)  # get_data_for_stock(stock, timestamp)
        stock_data.index = [pd.Timestamp(ts).tz_localize('America/Chicago').tz_convert('America/New_York') for ts in
                            stock_data.index]
        stock_data = stock_data[stock_data['Volume'] > 0]
        stock_data = stock_data.reset_index()
        stock_data['Variance'] = ta.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'])

        stock_data['hour_high'] = stock_data['Volume'].rolling(60).max() * 0.8
        stock_data['vwap'] = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
        stock_data['ewm200'] = stock_data['Close'].ewm(200).mean()
        idx = 0

        forward_data = stock_data[stock_data['index'] >= mktopen]
        true_forward_data = forward_data.copy()
        backward_data = stock_data[stock_data['index'] <= mktopen]
        lowest_value = forward_data['Close'].min()
        highest_value = forward_data['Close'].max()

        return forward_data['Close'].values[0], highest_value, lowest_value
    except Exception as e:
        print(e)
        return np.nan, np.nan, np.nan