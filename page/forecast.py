from tkinter import ALL
import plotly.graph_objs as go
import yfinance as yf
import streamlit as st
import requests
from streamlit_lottie import st_lottie_spinner
import investpy
from datetime import datetime, date, time
import pandas as pd

from crypto import ada_trading, bit_trading, eth_trading, xlm_trading, ltc_trading, usdt_trading, dot_trading, bch_trading, doge_trading, shib_trading


def get_forecast():

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_url = "https://assets7.lottiefiles.com/packages/lf20_p9ce3saq.json"
    lottie_json = load_lottieurl(lottie_url)

    st.warning("NOTE:- The forecast section contains technical analysis forecast, sentiment analysis and machine learning algorithm forecasting models. The figures and recommendations are for reference only. It does not guarantee future exact values. The website owner shall not be liable to any loss or damage incurred by any person caused by direct or indirect usage of the information or its content stated herein.")


    crypto = st.selectbox('Select Cryptocurrency',
                              ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Binance Coin (BNB)', 'Solana (SOL)'])



    if crypto == "Bitcoin (BTC)":
                # with st_lottie_spinner(lottie_json):
        get_chart('BTC-USD','BTCUSDT')
        st.markdown("***")
        news_sentiment()
        st.markdown("***")
        twitter_sentiment()
        st.markdown("***")
        random_forest()
        st.markdown("***")
        lstm()


    if crypto == "Ethereum (ETH)":
        get_chart('ETH-USD','ETHUSDT')


    if crypto == "Binance Coin (BNB)":
        get_chart('BNB-USD','BNBUSDT')


    if crypto == "Solana (SOL)":
        get_chart('SOL-USD','SOLUSDT')



def get_chart(ticker,symbol):
    data = yf.download(tickers=ticker, period='8d', interval='90m')
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'], name='market data'))
    fig.update_layout(title=ticker,
                      xaxis_title="Date",
                      yaxis_title="USD", )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=3, label="3d", step="day", stepmode="backward"),
                dict(count=5, label="5d", step="day", stepmode="backward"),
                dict(count=7, label="WTD", step="day", stepmode="todate"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Technical Analysis Forecast')
    
    time_int = st.selectbox('Select Time Interval',['5 Minutes', '15 Minutes', '30 Minutes', '1 Hour', 'Daily'])

    if time_int == "5 Minutes":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="5mins")
        t_indicator_5mins.to_csv("csv/ta_5mins.csv",index=False)           
        ta_5mins = pd.read_csv("csv/ta_5mins.csv")
        st.dataframe(ta_5mins)
        
    if time_int == "15 Minutes":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="15mins")
        t_indicator_5mins.to_csv("csv/ta_15mins.csv",index=False)           
        ta_15mins = pd.read_csv("csv/ta_15mins.csv")
        st.dataframe(ta_15mins)

    if time_int == "30 Minutes":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="30mins")
        t_indicator_5mins.to_csv("csv/ta_30mins.csv",index=False)           
        ta_30mins = pd.read_csv("csv/ta_30mins.csv")
        st.dataframe(ta_30mins)

    if time_int == "1 Hour":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="1hour")
        t_indicator_5mins.to_csv("csv/ta_1h.csv",index=False)           
        ta_1hr = pd.read_csv("csv/ta_1h.csv")
        st.dataframe(ta_1hr)

    if time_int == "Daily":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="daily")
        t_indicator_5mins.to_csv("csv/ta_daily.csv",index=False)           
        ta_daily = pd.read_csv("csv/ta_daily.csv")
        st.dataframe(ta_daily)


def news_sentiment():
    st.subheader('News Sentiment Analysis')
    st.warning("Under Construction")

def twitter_sentiment():
    st.subheader('Twitter Sentiment Analysis')
    st.warning("Under Construction")

def random_forest():
    st.subheader('Random Forest Model')
    st.warning("Under Construction")

def lstm():
    st.subheader('LSTM Model')
    st.warning("Under Construction")

#https://tradingview.brianthe.dev/
#https://investpy.readthedocs.io/_info/introduction.html

#https://github.com/twopirllc/pandas-ta#candles-64
#https://pypi.org/project/candlestick-patterns-subodh101/
#https://www.aquaq.co.uk/identifying-japanese-candle-sticks-using-python/
#https://github.com/SpiralDevelopment/candlestick-patterns

#news
#https://steemit.com/cryptocurrency/@firstaeon/fetch-cryptopanic-api-for-cryptocurrency-news-using-python

#https://github.com/SanjoShaju/Cryptocurrency-Analysis-Python#cryptocurrency-sentiment-analysisipynb

#https://www.atoti.io/articles/how-im-failing-my-twitter-sentiment-analysis-for-cryptocurrency-prediction/

#https://cryptopanic.com/news/14520233/95M-of-Shorts-Liquidated-as-Bitcoin-Ether-Rise-8

#https://newsdata.io/docs
#https://newsapi.org/docs/endpoints

#Twitter ***
#API key: fDNn9PgvwBNf8b5YkqVdexWy8
# API Secret: sjW9xy6ckF0wRDJKpdKNa1qoBg7lDaYk8421wSqDCloslXxWhq
# Bearer Token: AAAAAAAAAAAAAAAAAAAAAEpsdAEAAAAAZv9ZkZEGYDInn%2B1dlv%2Fd%2Bxn1j%2Bk%3D3MDLpPo82rfpQjpBRhDVyzukRHsT2KHT9dERAib0GLZJS3GnW1
#1530761395138359296-Q2BphWd2qzGTh0dl1AhTk2ZHiJ33uV
#i1D7upqRuoNHYxRE4lWluT5Z3LTQlrDxmMC0vPq4MX5Wr
#https://colab.research.google.com/drive/1afUKtKBCdi6nO2n39iIGCaSfqUg3GAMJ#scrollTo=VDNM2RrQBapg
#https://github.com/Drabble/TwitterSentimentAndCryptocurrencies
#https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/
#https://medium.com/thedevproject/powerful-twitter-sentiment-analysis-in-under-35-lines-of-code-a80460db24f6
#https://github.com/Amey23/Twitter-Sentimental-Analysis/blob/master/Twitter%20Sentiment%20Analysis.ipynb
#https://colab.research.google.com/drive/1H7CMXEdlrigAdv3ensC6u8-Y-KNOUbu3#scrollTo=177256bf
#https://colab.research.google.com/github/mrdbourke/tensorflow-deep-learning/blob/main/10_time_series_forecasting_in_tensorflow.ipynb


#social media
#https://pypi.org/project/lunarcrush/

#https://twelvedata.com/docs#exchange-rate


#Price Prediction
#https://thecleverprogrammer.com/2021/12/27/cryptocurrency-price-prediction-with-machine-learning/

#https://medium.com/analytics-vidhya/bitcoin-price-prediction-with-random-forest-and-technical-indicators-python-560800d6f3cd


#Random Forest
#https://medium.com/@maryamuzakariya/project-predict-stock-prices-using-random-forest-regression-model-in-python-fbe4edf01664
#https://medium.com/analytics-vidhya/bitcoin-price-prediction-with-random-forest-and-technical-indicators-python-560800d6f3cd
#ta
#https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb


#https://twelvedata.com/docs#technical-indicators