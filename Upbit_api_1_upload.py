#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############################### 모듈 import #####################################

import pyupbit
import jwt
import uuid
import hashlib
from urllib.parse import urlencode
import pandas as pd
import time
import webbrowser
import numpy as np
import time
import requests
import datetime 


############################### 프로그램 상수 #####################################
access_key = "???"
secret_key = "???"
myToken = "?"

#투자금액
invest_money = 300000

#거래할 코인
krw_tickers = ['KRW-BTC', 'KRW-ETH', 'KRW-SAND', 'KRW-BORA', 'KRW-XRP','KRW-PLA','KRW-BTT','KRW-MLK','KRW-MANA','KRW-EOS'
              ,'KRW-HIVE','KRW-WAXP','KRW-ADA', 'KRW-AQT', 'KRW-QTUM','KRW-WAVES','KRW-BAT','KRW-ETC','KRW-LINK','KRW-DOT'
              , 'KRW-VET','KRW-STORJ','KRW-LSK','KRW-XLM','KRW-POWR','KRW-ICX','KRW-ATOM','KRW-MTL','KRW-OMG','KRW-LTC','KRW-BCH'
              ,'KRW-BTG','KRW-NEO','KRW-ARK','KRW-IOTA', 'KRW-GLM', 'KRW-STEEM','KRW-GRS']

#익절,손절 퍼센트
goodsell_percent = 1.07
deadsell_percent = 0.95


################################# 함수 ####################################

#슬랙 message
def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )


#스토캐스틱rsi 1week (반환값 매수조건만족시 True 나머지는 False)
def stockrsiweeks(symbol):
    url = "https://api.upbit.com/v1/candles/weeks"

    querystring = {"market":symbol,"count":"200"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)

    series=df['trade_price'].iloc[::-1]

    df = pd.Series(df['trade_price'].values)

    period=14
    smoothK=3
    smoothD=3

    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] )
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] )
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() /          downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    
    

    condition = False;
    yyester_K=stochrsi_K.iloc[-3]*100
    yyester_D=stochrsi_D.iloc[-3]*100
    yester_K=stochrsi_K.iloc[-2]*100
    yester_D=stochrsi_D.iloc[-2]*100
    today_K=stochrsi_K.iloc[-1]*100
    today_D=stochrsi_D.iloc[-1]*100
    if(yyester_K<yester_K and yester_K<today_K and today_K > today_D):
        condition=True
    return condition


#스토캐스틱 1day (반환값 매수조건만족시 True 나머지는 False)
def stockrsidays(symbol):
    url = "https://api.upbit.com/v1/candles/days"

    querystring = {"market":symbol,"count":"200"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)

    series=df['trade_price'].iloc[::-1]

    df = pd.Series(df['trade_price'].values)

    period=14
    smoothK=3
    smoothD=3

    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] )
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] )
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() /          downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()


    condition = False;
    yyester_K=stochrsi_K.iloc[-3]*100
    yyester_D=stochrsi_D.iloc[-3]*100
    yester_K=stochrsi_K.iloc[-2]*100
    yester_D=stochrsi_D.iloc[-2]*100
    today_K=stochrsi_K.iloc[-1]*100
    today_D=stochrsi_D.iloc[-1]*100
    if(yyester_K<yester_K and yester_K<today_K and today_K > today_D):
        condition=True
    return condition

#macd 1day (반환값 매수조건만족시 True 나머지는 False)
def macddays(symbol):

    url = "https://api.upbit.com/v1/candles/days"


    querystring = {"market":symbol,"count":"200"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)

    df=df.iloc[::-1]

    df=df['trade_price']

    exp1 = df.ewm(span=12, adjust=False).mean() 
    exp2 = df.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()  #signal

    condition = False
    if(macd[2]<macd[1] and macd[1]<macd[0] and macd[0] >exp3[0]):
        condition = True

    return condition

#macd 60분 (반환값 매수조건만족시 True 나머지는 False)
def macd60m(symbol):

    url = "https://api.upbit.com/v1/candles/minutes/60"


    querystring = {"market":symbol,"count":"200"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)

    df=df.iloc[::-1]

    df=df['trade_price']

    exp1 = df.ewm(span=12, adjust=False).mean() 
    exp2 = df.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()  #signal

    condition = False
    if((macd[3]-exp3[3])>0 and (macd[3]-exp3[3])<(macd[2]-exp3[2]) and (macd[2]-exp3[2])<(macd[1]-exp3[1]) and 
       (macd[1]-exp3[1])<(macd[0]-exp3[0])):
        condition = True

    return condition

#MACD 30분 (반환값 매수조건만족시 True 나머지는 False)
def macd30m(symbol):

    url = "https://api.upbit.com/v1/candles/minutes/30"


    querystring = {"market":symbol,"count":"200"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)

    df=df.iloc[::-1]

    df=df['trade_price']

    exp1 = df.ewm(span=12, adjust=False).mean() 
    exp2 = df.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()  #signal
    condition = False
    if((macd[1]-exp3[1])<(macd[0]-exp3[0])):
        condition = True
        
    return condition

#OBV 값 구하는 함수
def OBV(tradePrice, volume):
    obv = pd.Series(index=tradePrice.index)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1,len(tradePrice)):
        if tradePrice.iloc[i] > tradePrice.iloc[i-1] : 
            obv.iloc[i] = obv.iloc[i-1] + volume[i]
            
        elif tradePrice.iloc[i] < tradePrice.iloc[i-1] :
            obv.iloc[i] = obv.iloc[i-1] - volume[i]
            
        else:
            obv.iloc[i] = obv.iloc[i-1]
            
    return obv

#코인의 OBV 매수조건 테스트 (반환값 매수조건만족시 True 나머지는 False)
def obv(symbol):
    
    url = "https://api.upbit.com/v1/candles/days"
    querystring = {"market":symbol,"count":"200"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)
    df=df.iloc[::-1]

    obv = OBV(df['trade_price'],df['candle_acc_trade_volume'])
    condition= False
    if(obv[2]<obv[1] and obv[1]<obv[0]):
        condition = True
    
    return condition    


#내 KRW 자산 조회
def get_my_KRW_Balance():
    return upbit.get_balance("KRW")



# 모든 매수조건 만족 테스트
def buy_test (symbol):
    test = False
    if(stockrsidays(symbol) and macddays(symbol) and obv(symbol) and stockrsiweeks(symbol) and macd60m(symbol) and
       macd30m(symbol)):
        test = True
    return test

        


########### 메인 로직 ##############################################
upbit = pyupbit.Upbit(access_key, secret_key) # pyupbit 사용하기 위함
post_message(myToken,"#upbit", "autotrade start")

#익절의 경우 이틀전종가, 하루전종가, 오늘최고가 기준 -2퍼아래로 떨어졌을 경우 매도.
good_sell_list = []
#매수할 때 -2퍼 조정이 왔을 때 매수.
buy_list = []

while True:
    sell_list = []
    
    try:
        
        #매수조건 테스트 및 매수된 종목들 sell_list에 넣는 작업
        for symbol in krw_tickers:
            #최근 체결된 코인 정렬
            
            order_done = upbit.get_order(symbol, state="cancel") +upbit.get_order(symbol, state="done") 
            if ( [] != order_done): #빈 배열이 있을때 오류가 나지않게하기위해서
                order_done_sorted = sorted(order_done, key=(lambda x: x['created_at']), reverse=True)
            
            #매수 조건 만족 시 
            #0 코인명, 1 매수 조건 만족 했을 때 가격, 2 매수 조건이 왔을 때 +12시간
            if(order_done_sorted[0]['side'] == 'ask' or order_done ==[]):
                if(buy_test(symbol)):
                    current_price = pyupbit.get_current_price(symbol)
                    current_time = datetime.datetime.now() 
                    twelve_hour_later = current_time + datetime.timedelta(hours=12) 
                    
                    temp_buy_list = []
                    temp_buy_list.append(symbol)
                    temp_buy_list.append(current_price)
                    temp_buy_list.append(twelve_hour_later)
                    buy_list.append(temp_buy_list)        
            
            
            
            #최근 체결이 매수인 코인들을 temp_list에 넣고 마지막에 2차원배열 형태로 sell_list에 insert함
            if(order_done_sorted[0]['side'] == 'bid'):
                
                #구매할 때 코인가격
                coin_buy_price = int(float(order_done_sorted[0]['price'])/float(order_done_sorted[0]['executed_volume']))
                #코인개수
                coin_count = float(order_done_sorted[0]['executed_volume'])
                
                #0 코인명, 1 구매할때 코인가격, 2 코인개수
                temp_list =[]
                temp_list.append(order_done_sorted[0]['market'])
                temp_list.append(coin_buy_price)
                temp_list.append(coin_count)
                sell_list.append(temp_list)
        print(sell_list)        
        time.sleep(1)
        
        #매수
        #매수 조건이 온 후 12시간이 지났을 때 매수종목 제거
        # + 조정이 왔을 때 매수
        if(buy_list != []):
            current_time = datetime.datetime.now() 
            for i in range(len(buy_list)):
                if(buy_list[i][2]<current_time):
                    del buy_list[i]

            for buy in buy_list:
                current_price = pyupbit.get_current_price(buy[0])
                if(buy[1]*0.98 > current_price):
                    buy_result = upbit.buy_market_order(buy[0], invest_money)
                    post_message(myToken,"#upbit", buy[0]+"coin buy : " +str(buy_result))
                    
                    for i in range(len(buy_list)):
                        if(buy_list[i][0] == buy[0]):
                            del buy_list[i]
        print(buy_list)        
        
        #매도
        if(sell_list != []):
            for sell_symbol in sell_list:
                current_price = pyupbit.get_current_price(sell_symbol[0])

                #손절
                if(sell_symbol[1]*deadsell_percent > current_price):
                    sell_result = upbit.sell_market_order(sell_symbol[0], sell_symbol[2])
                    post_message(myToken,"#upbit", sell_symbol[0]+"coin dead sell : " +str(sell_result))



                #익절

                #good_sell_list 중복체크
                overlap_test = True
                if([] != good_sell_list):
                    for i in good_sell_list:
                        if(i == sell_symbol[0]):
                            overlap_test = False
                #현재가가 매수 * goodsell_percent 보다 높아졌을 때 good_sell_list에 코인명을 추가
                if(overlap_test and sell_symbol[1] * goodsell_percent < current_price):
                    good_sell_list.append(sell_symbol[0])

                #good_sell_list에 있는 코인을 익절조건 테스트
                if([] != good_sell_list):
                    for good_sell in good_sell_list:
                        if(good_sell == sell_symbol[0]):
                            df = pyupbit.get_ohlcv(sell_symbol[0], count=3)

                            #이틀전,하루전 종가와 당일고가 중 가장 높은 가격을 구함
                            most_high = df.iloc[2]['high'] #당일 고가
                            if(most_high < df.iloc[0]['close']): 
                                most_high = df.iloc[0]['close'] #이틀전 종가
                            if(most_high < df.iloc[1]['close']):
                                most_high = df.iloc[1]['close'] #하루전 종가

                            #current_price와 most_high를 비교해서 매도
                            if(most_high *0.98 >current_price):
                                sell_result = upbit.sell_market_order(sell_symbol[0], sell_symbol[2])
                                post_message(myToken,"#upbit", sell_symbol[0]+"coin good sell : " +str(sell_result))

                                #good_sell_list에 있는 코인을 제거해줌
                                for i in range(len(good_sell_list)):
                                    if(good_sell_list[i] == sell_symbol[0]):
                                        del good_sell_list[i]
                 
        print(good_sell_list)
        

    except Exception as e:
        print(e)
        post_message(myToken,"#upbit", e+"error!!")
        time.sleep(1)

        

