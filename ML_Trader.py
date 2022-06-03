# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:50:39 2022

@author: JacobsEb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from ta.trend import ema_indicator
from ta.volatility import BollingerBands
import talib

# import MT_LOGIN


account_no1 = XXXXXXXXXXXX
account_no1_password = "XXXXXXXXXXXXXXXXX"
mt5.initialize(login=account_no1, server="XXXXXXXXXXXXX", password=account_no1_password)

Symbol = "GBPUSD"
Lot = 0.01
LookBack = 200
Future = 40
ForeCast = 12
timeframe = mt5.TIMEFRAME_M30


class ML_Trading:

    def __init__(self, LookBack, Symbol, Future, ForeCast, timeframe):
        self.LookBack = LookBack
        self.Symbol = Symbol
        self.Future = Future
        self.ForeCast = ForeCast
        self.timeframe = timeframe

    def DataPrep(self):
        df = pd.DataFrame(mt5.copy_rates_from(Symbol, timeframe, datetime.now(), LookBack))
        df['EMA'] = ema_indicator(df['close'], window=5, fillna=True)
        indicator_bb = BollingerBands(df['close'], window=14, window_dev=1.5, fillna=True)
        df['BB_Low'] = indicator_bb.bollinger_lband()
        df['BB_High'] = indicator_bb.bollinger_hband()

        st, upt, dt = Get_Supertrend.get_supertrend(Symbol, 3, 1)
        df['Super'] = st

        df = df.dropna()

        df = df.drop(labels=['time', 'open', 'low', 'high', 'tick_volume', 'real_volume', 'spread'], axis=1)
        df_for_training = df.astype(float)
        scaler = StandardScaler()
        scaler = scaler.fit(df_for_training)
        df_for_training_scaled = scaler.fit_transform(df_for_training)

        trainX = []
        trainY = []

        n_future = 12
        n_past = 30

        for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
            trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training_scaled.shape[1]])
            trainY.append(df_for_training_scaled[i + n_future - 1: i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)

        return trainX, trainY, scaler, df_for_training

    def LearnNew(self, bolEnable=True, emaEnable=True):
        trainX, trainY, scaler, df_training = ML_Trading.DataPrep(Symbol)

        model = Sequential()
        model.add(LSTM(1263, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(LSTM(132, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
        history = model.fit(trainX, trainY, epochs=200, batch_size=16, validation_split=0.2, verbose=1, callbacks=[es])

        return scaler, model

    def NewForeCast(self, scaler, model, bolEnable=True, emaEnable=True):
        trainX, trainY, scaler_intermedaite, df_for_training = ML_Trading.DataPrep(Symbol)
        n_future = ForeCast

        forecast = model.predict(trainX[-n_future:])
        forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)

        y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]
        offset = df_for_training[-1:].close.values[0] - y_pred_future[0]

        y_pred_future = y_pred_future + offset

        return y_pred_future


class TradingActions():

    def __init__(self):
        self.Symbol = Symbol

    def CountTrades(self):
        positions = mt5.positions_get(symbol=Symbol)

        count = 0
        for pos in positions:
            count = count + 1

        return count

    def open_trade(self, action, SL, TP, lot, deviation):
        '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
        '''
        # prepare the buy request structure
        if action == 'Buy':
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(Symbol).ask
        elif action == 'Sell':
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(Symbol).bid

        if (SL == 0) & (TP == 0):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": Symbol,
                "volume": lot,
                "type": trade_type,
                "price": price,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python ML script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        if (TP == 0) & (SL != 0):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": Symbol,
                "volume": lot,
                "type": trade_type,
                "price": price,
                "sl": SL,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python ML script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        if (SL == 0) & (TP != 0):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": Symbol,
                "volume": lot,
                "type": trade_type,
                "price": price,
                "tp": TP,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python ML script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.mt5.ORDER_FILLING_FOK,
            }
        if (SL != 0) & (TP != 0):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": Symbol,
                "volume": lot,
                "type": trade_type,
                "price": price,
                "sl": SL,
                "tp": TP,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python ML script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            # send a trading request
        result = mt5.order_send(request)

        return result, request

    def close_trade(self, position, deviation):
        '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
        '''
        # create a close request
        if position.type == 0:
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(Symbol).bid
        elif position.type == 1:
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(Symbol).ask

        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": Symbol,
            "volume": position.volume,
            "type": trade_type,
            "position": position.ticket,
            "price": price,
            "deviation": deviation,
            "magic": 234000,
            "comment": "python ML script close",
            "type_time": mt5.ORDER_TIME_GTC,  # good till cancelled
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        # send a close request
        result = mt5.order_send(close_request)

        return result, close_request

    def ModifySL(self, newSL, position):

        mod_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": Symbol,
            "sl": newSL,
            "tp": position.tp,
            "position": position.ticket,
        }
        result = mt5.order_send(mod_request)

        return result, mod_request

    def PreviousOpenTrade(self):
        positions = mt5.positions_get(symbol=Symbol)

        if (positions == ()):
            return 'NoOpenTrades'
        else:
            if (positions[-1].type == 0):
                return 'Buy'
            if (positions[-1].type == 1):
                return 'Sell'

    def CloseAll(self):
        positions = mt5.positions_get(symbol=Symbol)
        for pos in positions:
            TradingActions.close_trade(Symbol, pos, 5)


class Get_Supertrend:

    def __init__(self):
        self.Symbol = Symbol
        return

    def get_supertrend(self, lookback, multiplier):

        df = pd.DataFrame(mt5.copy_rates_from(Symbol, timeframe, datetime.now(), LookBack))
        high = df['high']
        low = df['low']
        close = df['close']
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
            if close[i + 1] >= supertrend.iloc[i, 0]:
                upt.append(supertrend.iloc[i, 0])
                dt.append(np.nan)
            elif close[i + 1] < supertrend.iloc[i, 0]:
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
        self.timeframe = timeframe
        self.Symbol = Symbol

    def DataPrep(self):
        df = pd.DataFrame(mt5.copy_rates_from(Symbol, timeframe, datetime.now(), LookBack))
        op = df.open
        hi = df.high
        lo = df.low
        cl = df.close
        return op, hi, lo, cl

    def Patterns():
        candle_names = talib.get_function_groups()['Pattern Recognition']
        op, hi, lo, cl = TechnicalPatterns.DataPrep(Symbol)
        df = pd.DataFrame()
        for candle in candle_names:
            df[candle] = getattr(talib, candle)(op, hi, lo, cl)
        return df


def StartTrading():
    scaler, model = ML_Trading.LearnNew(Symbol, Future, LookBack)
    hourServer = mt5.copy_rates_from(Symbol, timeframe, datetime.now(), 1)[0][0]
    dayNow = datetime.now().day

    UpPredic = False
    DwPredic = False

    while True:
        if dayNow != datetime.now().day:
            dayNow = datetime.now().day
            print("New Day Retrain a model")
            scaler, model = ML_Trading.LearnNew(Symbol, Future, LookBack)
        else:
            if hourServer == mt5.copy_rates_from(Symbol, timeframe, datetime.now(), 1)[0][0]:
                time.sleep(10)
            if (hourServer != mt5.copy_rates_from(Symbol, timeframe, datetime.now(), 1)[0][0]):
                hourServer = mt5.copy_rates_from(Symbol, timeframe, datetime.now(), 1)[0][0]
                # this is where we do a new predictoin.
                Prediction = ML_Trading.NewForeCast(Symbol, scaler, model)
                st, upt, dt = Get_Supertrend.get_supertrend(Symbol, 5, 2)
                Up = upt[-1:].isnull().values[0] == False
                Dw = dt[-1:].isnull().values[0] == False

                CurrentTickInfo = mt5.symbol_info_tick(Symbol)
                Ask = CurrentTickInfo.ask
                Bid = CurrentTickInfo.bid

                MaxPredict = Prediction.max()
                MinPredict = Prediction.min()

                previousUpPredic = UpPredic
                previousDwPredic = DwPredic

                UpPredic = Prediction[0] < Prediction[4]
                DwPredic = Prediction[0] >= Prediction[4]

                print("Super Trend Indicator is for Up ", Up, " for Down is ", Dw)
                print("Predictions are showing ", UpPredic, " showing down ", DwPredic)
                for i in np.linspace(0, 6, 7):
                    print("Price Predictions ", Prediction[int(i)])
                print("------------------------------------------------------------------------------")

                if (TradingActions.CountTrades(Symbol) == 0):
                    if (previousDwPredic & UpPredic):
                        print("Buy Trade Initiated")
                        TradeSL = upt[-1:].values[0]
                        TradeTP = (MaxPredict - Ask) / 2 + Ask
                        TradingActions.open_trade(Symbol, 'Buy', TradeSL, TradeTP, Lot, 5)

                    if (previousUpPredic & DwPredic):
                        print("Sell Trade Initiated")
                        TradeSL = dt[-1:].values[0]
                        TradeTP = Bid - (Bid - MinPredict) / 2
                        TradingActions.open_trade(Symbol, 'Sell', TradeSL, TradeTP, Lot, 5)
                else:
                    if (previousDwPredic != DwPredic) | (previousUpPredic != UpPredic):
                        TradingActions.CloseAll(Symbol)


