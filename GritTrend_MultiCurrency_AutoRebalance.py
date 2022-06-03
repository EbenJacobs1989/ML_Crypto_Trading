from binance.client import Client, AsyncClient
import pandas as pd
import requests
import numpy as np
from itertools import permutations
import ccxt
import API_INFO

# https://python-binance.readthedocs.io/en/latest/account.html#id2

# Assumption is that you should have one of your base currencies in your wallet before it can start
# Will always reference USDT as what needs to be done for rebalancing.


# add API information
api_key = ''
api_secret = ''

# first entry here is the base currency to be used.
BaseCurrencies = ['USDT', 'BTC', 'BNB']

# intializing client connection to Binanace.
client = Client(api_key, api_secret)
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': api_key,
    'secret': api_secret,
})

POSITION_SIZE = 0.001

# gridbot settings
NUM_BUY_GRID_LINES = 5
NUM_SELL_GRID_LINES = 2
# percentage above and below market values
GRID_SIZE = 2

CHECK_ORDERS_FREQUENCY = 1
CLOSED_ORDER_STATUS = 'closed'


class CurrencyChecks():

    def FindTradingPair(AllSymbolsDF, Curr1, Curr2):

        df_one = AllSymbolsDF[AllSymbolsDF['Base'] == Curr1]
        if (df_one.empty == False):
            df_two = df_one[df_one['Quoted'] == Curr2]
            if (df_two.empty == False):
                return df_two
            else:
                df_one = AllSymbolsDF[AllSymbolsDF['Base'] == Curr2]
                if (df_one.empty == False):
                    df_two = df_one[df_one['Quoted'] == Curr1]
                    if (df_two.empty == False):
                        return df_two
                    else:
                        return False

    def CheckCurrencies():
        MarketSymbols = requests.get('https://api.binance.com/api/v3/exchangeInfo')
        MarketSymbols = MarketSymbols.json()

        AllSymbols = []
        SymbolList = []

        # generating a list of all currently trading pairs
        for FindSymbols in MarketSymbols['symbols']:
            if (FindSymbols['status'] == 'TRADING'):
                AllSymbols.append([FindSymbols['symbol'], FindSymbols['quoteAsset'], FindSymbols['baseAsset']])

        AllSymbolsDF = pd.DataFrame(AllSymbols, columns=['TradingPair', 'Quoted', 'Base'])

        # check if there is Triangles using the base currencies
        # Generate the list for the base Trading Pairs

        Curr1, Curr2, Curr3 = BaseCurrencies[0], BaseCurrencies[1], BaseCurrencies[2]

        df1 = CurrencyChecks.FindTradingPair(AllSymbolsDF, Curr1, Curr2)
        df2 = CurrencyChecks.FindTradingPair(AllSymbolsDF, Curr2, Curr3)
        df3 = CurrencyChecks.FindTradingPair(AllSymbolsDF, Curr1, Curr3)

        BaseTradingPairs = [df1['TradingPair'].values[0], df2['TradingPair'].values[0], df3['TradingPair'].values[0]]
        BaseCurrency = [df1['Base'].values[0], df2['Base'].values[0], df3['Base'].values[0]]
        QoutedCurrency = [df1['Quoted'].values[0], df2['Quoted'].values[0], df3['Quoted'].values[0]]

        return BaseTradingPairs, BaseCurrency, QoutedCurrency


class Rebalance:

    def AutoRebalance():

        BaseTradingPairs, BaseCurrency, QoutedCurrency = CurrencyChecks.CheckCurrencies()

        MarketSymbols = requests.get('https://api.binance.com/api/v3/exchangeInfo')
        MarketSymbols = MarketSymbols.json()

        AllSymbols = []
        SymbolList = []

        # generating a list of all currently trading pairs
        for FindSymbols in MarketSymbols['symbols']:
            if (FindSymbols['status'] == 'TRADING'):
                AllSymbols.append([FindSymbols['symbol'], FindSymbols['quoteAsset'], FindSymbols['baseAsset']])

        AllSymbolsDF = pd.DataFrame(AllSymbols, columns=['TradingPair', 'Quoted', 'Base'])

        # Do rebalancing each day or x amount of time further or When bot is stareted
        balances = exchange.fetch_balance()
        Balance_InAccount = []

        FreeVal = balances['free']

        for symbols in FreeVal:
            if (FreeVal[symbols] != 0):
                Balance_InAccount.append([symbols, FreeVal[symbols]])

        Balance_InAccountDF = pd.DataFrame(Balance_InAccount, columns=['Symbol', 'value'])
        Balance_InAccountDF = Balance_InAccountDF.drop_duplicates(subset='Symbol')

        # check if balance contains any of the base currencies.
        Check = False
        for Curr in BaseCurrencies:
            for row in Balance_InAccountDF.iterrows():
                CheckToken = row[1]['Symbol']
                if (CheckToken == Curr):
                    Check = True

        if Check == False:
            print("No Currencies found in Wallet that allighns with Base Currencies")
        else:
            TotalValue_Account = 0
            TokenValues = []
            Total = 0;

            for Base in BaseCurrencies:
                FindAssociatedValue = Balance_InAccountDF[Balance_InAccountDF['Symbol'] == Base].value
                if (FindAssociatedValue.empty):
                    value_BaseCurrency = 0
                else:
                    value_BaseCurrency = float(FindAssociatedValue)
                Total = Total + value_BaseCurrency
                TokenValues.append([Base, value_BaseCurrency])

            TokenValues = pd.DataFrame(TokenValues, columns=['Symbol', 'Value'])

            # Sum and average the total value of tokens in USDT value
            # Calculate the difference between average and token values
            # Find Max value token and see how much need to be transferred to the various other ones

            # finding base equivelant values of everything in trade
            Holder = []
            for Curr in BaseCurrencies:
                if (Curr == BaseCurrencies[0]):
                    Holder.append(TokenValues['Value'][0])
                else:
                    check = CurrencyChecks.FindTradingPair(AllSymbolsDF, Curr, BaseCurrencies[0])
                    tokenAmount = TokenValues[TokenValues['Symbol'] == Curr]['Value'].values[0]
                    exchangeRate = exchange.fetch_ticker(check['TradingPair'].values[0])
                    if check['Base'].values[0] == Curr:
                        Holder.append(exchangeRate['bid'] * tokenAmount)
                    if check['Quoted'].values[0] == Curr:
                        Holder.append(tokenAmount / exchangeRate['ask'])

            TokenValues['BaseEquiv0'] = Holder

            Base = TokenValues['BaseEquiv0']
            ValueOfMax = Base.max()
            MaxValueToken = TokenValues[TokenValues['BaseEquiv0'] == ValueOfMax]['Symbol'].values[0]
            BalanceRequiredEachToken = Base.sum() / Base.count()

            TokenValues['TransfersBase'] = TokenValues['BaseEquiv0'] - BalanceRequiredEachToken
            RatioHighTokenTransfer = TokenValues[TokenValues['Symbol'] == MaxValueToken]['Value'].values[0] / \
                                     TokenValues[TokenValues['Symbol'] == MaxValueToken]['BaseEquiv0'].values[0]

            TransferValuesFromMax = []

            for row in TokenValues.iterrows():
                if row[1].Symbol == MaxValueToken:
                    TransferValuesFromMax.append(0)
                else:
                    ValueToBeTransfered = row[1].TransfersBase * RatioHighTokenTransfer
                    TransferValuesFromMax.append(np.abs(ValueToBeTransfered))

            TokenValues['TransferMaxToken'] = TransferValuesFromMax

            for row in TokenValues.iterrows():
                if row[1].Symbol == MaxValueToken:
                    print('Have this token')
                else:
                    TransferringDF = CurrencyChecks.FindTradingPair(AllSymbolsDF, MaxValueToken, row[1].Symbol)
                    AmountTransfer = row[1].TransferMaxToken
                    if (TransferringDF['Quoted'].values[0] == MaxValueToken):
                        # buying action
                        exchange.create_market_buy_order(TransferringDF['TradingPair'].values[0], AmountTransfer)
                    if (TransferringDF['Base'].values[0] == MaxValueToken):
                        # selling action
                        exchange.create_market_sell_order(TransferringDF['TradingPair'].values[0], AmountTransfer)


class GridTrend:

    def __init__(self):
        self.Percentage = Percentage

    def CurrentBidAsk(symbol):
        Info = exchange.fetch_ticker(symbol)
        bid = Info['bid']
        ask = Info['ask']
        return ask, bid

    def LookOpenOrders(symbol):
        orders = exchange.fetchOpenOrders(symbol)
        if (orders == []):
            # nothing is open and open buys and sells
            PlaceLimitBuy(symbol, GRID_SIZE)
            PlaceLimitSell(symbol, GRID_SIZE)

    def PlaceLimitBuy(symbol, percentage):
        ask, bid = GridTrend.CurrentBidAsk(symbol)
        new_limit_buy = ask -
        new_limit_buy_order = exchange.create_limit_buy_order(symbol, amount, new_limit_buy)

    def PlaceLimitSell(symbol, percentage):
        ask, bid = GridTrend.CurrentBidAsk(symbol)
        new_limit_sell = bid +
        exchange.create_limit_sell_order(symbol, amount, new_limit_sell)

        closed_order_ids.append(order_info['id'])
        print("buy order executed at {}".format(order_info['price']))
        new_sell_price = float(order_info['price']) + config.GRID_SIZE
        print("creating new limit sell order at {}".format(new_sell_price))
        new_sell_order = exchange.create_limit_sell_order(config.SYMBOL, config.POSITION_SIZE, new_sell_price)
        sell_orders.append(new_sell_order)

    order_info = order['info']

    if order_info['status'] == config.CLOSED_ORDER_STATUS:
        closed_order_ids.append(order_info['id'])
        print("sell order executed at {}".format(order_info['price']))
        new_buy_price = float(order_info['price']) - config.GRID_SIZE
        print("creating new limit buy order at {}".format(new_buy_price))
        new_buy_order = exchange.create_limit_buy_order(config.SYMBOL, config.POSITION_SIZE, new_buy_price)
        buy_orders.append(new_buy_order)


def TradingStart():


































