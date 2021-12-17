#!/usr/bin/env python
# coding: utf-8

#  
# 
# ## 주봉
# 
# 스토캐스틱 %K가 %D보다 위에
# 
# 스토캐스틱 상승추세 
# 
#  
# 
# ## 일봉
# 
# 스토캐스틱 상승추세
# 
#  Macd Histororim  이틀연속 상승추세
# 
#  OBV Signal  하루전 이틀전보다 더 높은경우 (상승추세)
# 
#  
# ## 60분봉
# 
# Macd 3봉이네 돌파
# 
# Macd 상승추세
# 
#  
# ## 30분봉
# 
#  Macd 상승추세

# In[8]:


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

access_key = "??"
secret_key = "??"
myToken = "xoxb"


payload = {
    'access_key': access_key,
    'nonce': str(uuid.uuid4()),
}

jwt_token = jwt.encode(payload, secret_key)
authorize_token = 'Bearer {}'.format(jwt_token)
headers = {"Authorization": authorize_token}

res = requests.get("https://api.upbit.com/v1/accounts", headers=headers)

data = res.json()

upbit = pyupbit.Upbit(access_key, secret_key)

# 슬랙 메시지
def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )

#요건 그냥 RSI

def rsiindex(symbol):
    url = "https://api.upbit.com/v1/candles/days"

    querystring = {"market":symbol,"count":"500"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)

    df=df.reindex(index=df.index[::-1]).reset_index()

    df['close']=df["trade_price"]



    def rsi(ohlc: pd.DataFrame, period: int = 14):
        ohlc["close"] = ohlc["close"]
        delta = ohlc["close"].diff()

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        _gain = up.ewm(com=(period - 1), min_periods=period).mean()
        _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

        RS = _gain / _loss
        return pd.Series(100 - (100 / (1 + RS)), name="RSI")

    rsi = rsi(df, 14).iloc[-1]
    print(symbol)
    print('upbit 1day RSI:', rsi)
    print('')
    time.sleep(1)

#################################### sto RSI #######################################


def stockrsiweeks(symbol):
    url = "https://api.upbit.com/v1/candles/weeks"

    querystring = {"market":symbol,"count":"500"}

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
    #print(stochrsi_K.iloc[-49]*100, stochrsi_K.iloc[-50]*100,stochrsi_K.iloc[-51]*100,stochrsi_K.iloc[-52]*100)
    #print("\n")
    #print(stochrsi_D.iloc[-49]*100, stochrsi_D.iloc[-50]*100,stochrsi_D.iloc[-51]*100,stochrsi_D.iloc[-52]*100)
    return condition


# stockrsiweeks("KRW-BTC")
#rsiindex("KRW-ETH")
#stockrsi("KRW-BTC")
#rsiindex("KRW-XRP")
#stockrsi("KRW-XRP")


def stockrsidays(symbol):
    url = "https://api.upbit.com/v1/candles/days"

    querystring = {"market":symbol,"count":"500"}

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
    '''
    print(symbol)    
    print('upbit 1day stoch_rsi_K: ', stochrsi_K.iloc[-3]*100)
    print('upbit 1day stoch_rsi_D: ', stochrsi_D.iloc[-3]*100)
    print('')
    print('upbit 1day stoch_rsi_K: ', stochrsi_K.iloc[-2]*100)
    print('upbit 1day stoch_rsi_D: ', stochrsi_D.iloc[-2]*100)
    print('')
    print('upbit 1day stoch_rsi_K: ', stochrsi_K.iloc[-1]*100)
    print('upbit 1day stoch_rsi_D: ', stochrsi_D.iloc[-1]*100)
    print('') '''

    
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
    
#stockrsidays("KRW-BTC")
#rsiindex("KRW-ETH")
#stockrsi("KRW-BTC")
#rsiindex("KRW-XRP")
#stockrsi("KRW-XRP")



#################################### MACD #######################################

def macddays(symbol):

    url = "https://api.upbit.com/v1/candles/days"


    querystring = {"market":symbol,"count":"500"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)

    df=df.iloc[::-1]

    df=df['trade_price']

    exp1 = df.ewm(span=12, adjust=False).mean() 
    exp2 = df.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()  #signal
    '''
    print('MACD: ',macd[0])
    print('Signal: ',exp3[0])
    print("ocilator:",macd[0]-exp3[0])
    print("\n")
    test1=exp3[0]-macd[0]
    test2=exp3[1]-macd[1]

    call='매매 필요없음'
    if test1<0 and test2>0:
       call='매도'
    if test1>0 and test2<0:
       call='매수'
    time.sleep(1)
    '''#36 37 38
    #print(macd[0], exp3[0] ," ww ",macd[1], exp3[1], " ww ", macd[2], exp3[2])
    condition = False
    if(macd[2]<macd[1] and macd[1]<macd[0] and macd[0] >exp3[0]):
        condition = True

    return condition

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
    '''
    print('MACD: ',macd[0])
    print('Signal: ',exp3[0])
    print("ocilator:",macd[0]-exp3[0])
    print("\n")
    test1=exp3[0]-macd[0]
    test2=exp3[1]-macd[1]

    call='매매 필요없음'
    if test1<0 and test2>0:
       call='매도'
    if test1>0 and test2<0:
       call='매수'
    time.sleep(1)
    '''#36 37 38
    #print(macd[0], exp3[0] ," ww ",macd[1], exp3[1], " ww ", macd[2], exp3[2])
    condition = False
    if((macd[3]-exp3[3])>0 and (macd[3]-exp3[3])<(macd[2]-exp3[2]) and (macd[2]-exp3[2])<(macd[1]-exp3[1]) and 
       (macd[1]-exp3[1])<(macd[0]-exp3[0])):
        condition = True

    return condition

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
    '''
    print('MACD: ',macd[0])
    print('Signal: ',exp3[0])
    print("ocilator:",macd[0]-exp3[0])
    print("\n")
    test1=exp3[0]-macd[0]
    test2=exp3[1]-macd[1]

    call='매매 필요없음'
    if test1<0 and test2>0:
       call='매도'
    if test1>0 and test2<0:
       call='매수'
    time.sleep(1)
    '''#36 37 38
    #print(macd[0], exp3[0] ," ww ",macd[1], exp3[1], " ww ", macd[2], exp3[2])
    condition = False
    if((macd[1]-exp3[1])<(macd[0]-exp3[0])):
        condition = True
    #print(macd[0], macd[1], exp3[0], exp3[1])
    return condition
# macd30m("KRW-BTC")

#################################### OBV #######################################

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

def obv(symbol):
    
    url = "https://api.upbit.com/v1/candles/days"
    querystring = {"market":symbol,"count":"500"}

    response = requests.request("GET", url, params=querystring)

    data = response.json()

    df = pd.DataFrame(data)
    df=df.iloc[::-1]

    obv = OBV(df['trade_price'],df['candle_acc_trade_volume'])
    #print(obv[3], obv[2],obv[1],obv[0])
    condition= False
    if(obv[2]<obv[1] and obv[1]<obv[0]):
        condition = True
    
    return condition    







# 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 시작 




# 시작 메세지 슬랙 전송



sell_list = []    # 빈 리스트 생성

post_message(myToken,"#upbit", "autotrade start")

# buy = False
# #print(stockrsidays(coin) ,macddays(coin) ,obv(coin) ,stockrsiweeks(coin) , macd60m(coin),macd30m(coin))
# if(stockrsidays(symbol) and macddays(symbol) and obv(symbol) and stockrsiweeks(symbol) and macd60m(symbol) and macd30m(symbol)):

#     buy = True
krw_tickers = ['KRW-BTC', 'KRW-ETH', 'KRW-SAND', 'KRW-BORA', 'KRW-XRP','KRW-PLA','KRW-BTT','KRW-MLK','KRW-MANA','KRW-EOS'
              ,'KRW-HIVE','KRW-WAXP','KRW-ADA', 'KRW-AQT', 'KRW-QTUM','KRW-WAVES','KRW-BAT','KRW-ETC',"KRW-TRX",'KRW-LINK','KRW-DOT'
              , 'KRW-VET','KRW-STORJ','KRW-LSK','KRW-XLM','KRW-POWR','KRW-ICX','KRW-ATOM','KRW-MTL','KRW-OMG','KRW-LTC','KRW-BCH'
              ,'KRW-BTG','KRW-NEO','KRW-ARK','KRW-IOTA', 'KRW-GLM', 'KRW-STEEM','KRW-GRS']
print("autotrade start")

while True:
    try:

        for symbol in krw_tickers:

            if(float(data[0]['balance'])>300000*1.0005):
                if(stockrsidays(symbol) and macddays(symbol) and obv(symbol) and stockrsiweeks(symbol) and macd60m(symbol) and
                   macd30m(symbol)):
                    buy_result = upbit.buy_market_order(symbol, 300000)
                    forselling = []
                    current_price = pyupbit.get_current_price(symbol)
                    forselling.append(buy_result['market'])
                    forselling.append(buy_result['price'])
                    forselling.append(current_price)
                    symbol_count = float(float(buy_result['price']) / current_price)
                    forselling.append(symbol_count)

                    sell_list.append(forselling)
                    post_message(myToken,"#upbit", symbol+"coin buy : " +str(buy_result))

        # 0 이름, 1 구매한 총 가격 300000, 2 구매했을때 코인 가격, 3 구매한 코인 개수
        for coin_selling in sell_list:
            coin_price = pyupbit.get_current_price(coin_selling[0])

            if (coin_selling[2]*1.08 < coin_price):
                sell_result = upbit.sell_market_order(coin_selling[0], coin_selling[3])
                post_message(myToken,"#upbit", coin_selling[0]+"coin good sell : " +str(sell_result))

            if(coin_selling[2]*0.95 > coin_price):
                sell_result = upbit.sell_market_order(coin_selling[0], coin_selling[3])
                post_message(myToken,"#upbit", coin_selling[0]+"coin dead sell : " +str(sell_result))
        time.sleep(1)
        print(sell_list)
        
    except Exception as e:
        print(e)
        post_message(myToken,"#upbit", e)
        time.sleep(1)




# In[ ]:





# In[5]:





# In[7]:





# In[6]:





# In[ ]:




