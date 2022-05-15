# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:32:19 2022

@author: JacobsEb
"""

import MetaTrader5 as mt5
import pandas as pd 
import talib
from datetime import datetime
import numpy as np 
import MT_LOGIN

account_no1 =MT_LOGIN.account_no1 
account_password_no1 =MT_LOGIN.account_no1_password
mt5.initialize(login=account_no1, server="Exness-MT5Trial",password=account_password_no1)

Symbol="GBPUSD"
LookBack = 200
Future = 40
timeframe = mt5.TIMEFRAME_H1

class TechnicalPatterns:
    
    def __init__(self):
        self.LookBack = LookBack 
        self.timeframe = timeframe
        self.Symbol = Symbol
        
    def DataPrep(self):
        df = df = pd.DataFrame(mt5.copy_rates_from(Symbol, timeframe, datetime.now(), LookBack))
        op = df.open
        hi = df.high
        lo = df.low
        cl = df.close
        return op,hi,lo,cl
        
        
    def Patterns():
        candle_names = talib.get_function_groups()['Pattern Recognition']
        op,hi,lo,cl = TechnicalPatterns.DataPrep(Symbol)
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
    



class TradingActions():
    
    def __init__(self):
        self.Symbol = Symbol
    
    def get_info(self):
        '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolinfo_py
        '''
        # get symbol properties
        info=mt5.symbol_info(Symbol)
        return info

    def open_trade(self, action, lot, sl_value, tp_value, deviation):
        '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
        '''
        # prepare the buy request structure
        symbol_info = TradingActions.get_info(Symbol)
    
        if action == 'buy':
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(Symbol).ask
        elif action =='sell':
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(Symbol).bid
        point = mt5.symbol_info(Symbol).point
    
        buy_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": Symbol,
            "volume": lot,
            "type": trade_type,
            "price": price,
            "sl": sl_value,
            "tp": tp_value,
            "deviation": deviation,
            "magic": 111,
            "comment": "sent by python",
            "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        # send a trading request
        result = mt5.order_send(buy_request)  
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("2. order_send failed, retcode={}".format(result.retcode))
            # request the result as a dictionary and display it element by element
            result_dict=result._asdict()
            for field in result_dict.keys():
                print("   {}={}".format(field,result_dict[field]))
                # if this is a trading request structure, display it element by element as well
                if field=="request":
                    traderequest_dict=result_dict[field]._asdict()
                    for tradereq_filed in traderequest_dict:
                        print("       traderequest: {}={}".format(tradereq_filed,traderequest_dict[tradereq_filed]))
                        
            print("shutdown() and quit")
            mt5.shutdown()
            
        return result, buy_request 
    
    def close_trade(self,action, buy_request, result, deviation):
        '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
        '''
        # create a close request
        symbol = buy_request['symbol']
        if action == 'buy':
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        elif action =='sell':
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        position_id=result.order
        lot = buy_request['volume']
    
        close_request={
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": trade_type,
            "position": position_id,
            "price": price,
            "deviation": deviation,
            "magic": 111,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        # send a close request
        result=mt5.order_send(close_request)