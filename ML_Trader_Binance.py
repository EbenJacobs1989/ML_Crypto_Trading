# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:50:39 2022

@author: JacobsEb
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error , mean_squared_error
from ta.trend import ema_indicator
from ta.volatility import BollingerBands
import talib
import ccxt
from binance.client import Client, AsyncClient
import API_INFO


# add API information 
api_key = API_INFO.api_key
api_secret=API_INFO.api_secret

#first entry here is the base currency to be used. 
BaseCurrencies = ['USDT','BTC','BNB']

#intializing client connection to Binanace.
client =  Client(api_key, api_secret)
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': api_key,
    'secret': api_secret,
})


symbol="BTCUSDT"
LookBack = 200
Future = 40
ForeCast = 12
timeframes = '30m'



class ML_Trading:

    def __init__(self, LookBack, symbol, Future, ForeCast,timeframes):
        self.LookBack = LookBack
        self.symbol = symbol
        self.Future = Future
        self.ForeCast = ForeCast
        self.timeframes = timeframes

    def DataPrep(self):
        df = pd.DataFrame(exchange.fetchOHLCV(symbol,timeframe = timeframes,limit=300))
        df.columns = ['time','open','high','low','close','volume']
        df['EMA'] = ema_indicator(df['close'], window=5, fillna=True)
        indicator_bb = BollingerBands(df['close'], window=14, window_dev=1.5, fillna=True)
        df['BB_Low'] = indicator_bb.bollinger_lband()
        df['BB_High'] = indicator_bb.bollinger_hband()
        df.dropna()

        #df['time'] = pd.to_datetime(df['time'], unit='s')

        # df['time'] = pd.to_datetime(df['time'])
        date_str = datetime.today().strftime('%Y%m%d')
        df.set_index(df['time'], inplace=False, append=False)
        train_dates = pd.to_datetime(df.time)

        # df_for_training = df[['close','open','high','low','tick_volume','EMA','BB_Low','BB_High','log_returns']].astype(float)
        df_for_training = df[['close', 'EMA', 'BB_Low', 'BB_High']].astype(float)
        #df_for_training = df[['close']].astype(float)

        scaler = StandardScaler()
        scaler = scaler.fit(df_for_training)
        df_for_training_scaled = scaler.fit_transform(df_for_training)

        trainX = []
        trainY = []

        n_future = 5
        n_past = 30

        for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
            trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training_scaled.shape[1]])
            trainY.append(df_for_training_scaled[i + n_future - 1: i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)

        return trainX, trainY, scaler, df_for_training

    def LearnNew(self, bolEnable=True, emaEnable=True):

        trainX, trainY, scaler, df_training = ML_Trading.DataPrep(symbol)

        model = Sequential()
        model.add(LSTM(1263, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(LSTM(132, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        history = model.fit(trainX, trainY, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

        return scaler, model

    def NewForeCast(self, scaler, model, bolEnable=True, emaEnable=True):

        trainX, trainY, scaler_intermedaite, df_for_training = ML_Trading.DataPrep(symbol)
        n_future = ForeCast
        forecast = model.predict(trainX[-n_future:])

        forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

        offset = df_for_training[-1:].close.values[0] - y_pred_future[0]

        y_pred_future = y_pred_future + offset
        # Below code only used for visualisation 
        # forecast_index = np.linspace(df_for_training.index.stop + 1, df_for_training.index.stop + n_future, n_future)

        return y_pred_future




class TradingActions():
    
    def __init__(self):
        self.TradingAction=TradingAction
    
    def OpenTrade(self,TradingAction):
    
        if (TradeAction =="LONG"):
            exchange.e
    


def get_supertrend(high, low, close, lookback, multiplier):            
        
    # ATR

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(lookback).mean()

    # H/L AVG AND BASIC UPPER & LOWER BAND

    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()

    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns=['upper', 'lower'])
    final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i - 1, 0]) | (close[i - 1] > final_bands.iloc[i - 1, 0]):
                final_bands.iloc[i, 0] = upper_band[i]
            else:
                final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]

    # FINAL LOWER BAND

    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i - 1, 1]) | (close[i - 1] < final_bands.iloc[i - 1, 1]):
                final_bands.iloc[i, 1] = lower_band[i]
            else:
                final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]

    # SUPERTREND

    supertrend = pd.DataFrame(columns=[f'supertrend_{lookback}'])
    supertrend.iloc[:, 0] = [x for x in final_bands['upper'] - final_bands['upper']]

    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]

    # ST UPTREND/DOWNTREND

    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close[i+1] >= supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i+1] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index

    return st, upt, dt

class TechnicalPatterns:
    
    def __init__(self):
        self.LookBack = LookBack 
        self.timeframes = timeframes
        self.symbol = symbol
        
    def DataPrep(self):
        df = pd.DataFrame(exchange.fetchOHLCV(symbol,timeframe = timeframes,limit=300))
        df.columns = ['time','open','high','low','close','volume']
        op = df.open
        hi = df.high
        lo = df.low
        cl = df.close
        return op,hi,lo,cl
        
        
    def Patterns():
        candle_names = talib.get_function_groups()['Pattern Recognition']
        op,hi,lo,cl = TechnicalPatterns.DataPrep(symbol)
        df = pd.DataFrame()
        for candle in candle_names:
            df[candle]= getattr(talib,candle)(op,hi,lo,cl)

        decisionMat=[]
        for row in df.iterrows():
            CandlesFound=0
            SumOfCandleValues=0
            for columnName in candle_names:
                currentVal = row[1][columnName]
                if currentVal!=0:
                    CandlesFound = CandlesFound+1
                    SumOfCandleValues = SumOfCandleValues+currentVal
            
            decision = "Nothing"
            if (SumOfCandleValues == CandlesFound*-100) & (CandlesFound!=0):
                decision = "Sell"
            elif (SumOfCandleValues == CandlesFound*100) & (CandlesFound!=0):
                decision = "Buy"
            elif (CandlesFound!=0):
                decision="Inconclusive"
            
            decisionMat.append(decision)
        
        df['Decision']=decisionMat
        return df

scaler,model = ML_Trading.LearnNew(symbol,Future,LookBack)
f = ML_Trading.NewForeCast(symbol,scaler,model)

