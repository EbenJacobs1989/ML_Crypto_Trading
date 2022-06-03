import pandas as pd
import numpy as np
import requests
from datetime import datetime
import ccxt
from binance.client import Client

# from kucoin.client import Client

# add API information Binance
api_key_binance = ''
api_secret_binance = ''

# add API information KuCoin
api_key_kucoin = ''
api_secret_kucoin = ''
api_passphrase_kucoin = ''


def GatherNewMarketInfo():
    MarketSymbols_Binance = requests.get('https://api.binance.com/api/v3/exchangeInfo')
    MarketSymbols_Binance = MarketSymbols_Binance.json()

    MarketSymbols_KuCoin = requests.get('https://api.kucoin.com./api/v1/symbols')
    MarketSymbols_KuCoin = MarketSymbols_KuCoin.json()

    return MarketSymbols_Binance, MarketSymbols_KuCoin


def findIfChange():
    MarketSymbols_Binance, MarketSymbols_KuCoin = GatherNewMarketInfo()

    if (MarketSymbols_Binance != MarketSymbols_Binance):
        print('Found New token Binance')

    if (MarketSymbols_KuCoin != MarketSymbols_KuCoin):
        print("Found New token KuCoin")