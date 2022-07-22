from tkinter import ALL
import plotly.graph_objs as go
import yfinance as yf
import streamlit as st
import requests
import investpy
from datetime import datetime, date, time
import pandas as pd
import flair
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import traceback
import urllib.request
from urllib.request import Request, urlopen
import json
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import finta
from finta import TA
import mplfinance as mpf
from pandas_datareader import data as web
from pandas import json_normalize
from PIL import Image


def get_forecast():

    st.warning("NOTE:- The forecast section contains technical analysis forecast, sentiment analysis and machine learning algorithm forecasting models. The figures and recommendations are for reference only. It does not guarantee future exact values. The website owner shall not be liable to any loss or damage incurred by any person caused by direct or indirect usage of the information or its content stated herein.")


    crypto = st.selectbox('Select Cryptocurrency',
                              ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Binance Coin (BNB)', 'Solana (SOL)'])



    if crypto == "Bitcoin (BTC)":
                # with st_lottie_spinner(lottie_json):
        get_chart('BTC-USD','BTCUSDT')
        st.markdown("***")
        btc_news_sentiment()
        st.markdown("***")
        twitter_sentiment("bitcoin")
        st.markdown("***")
        rf_btc()
        random_forest('BTC-USD')
        st.markdown("***")
        lstm('BTC-USD')


    if crypto == "Ethereum (ETH)":
        get_chart('ETH-USD','ETHUSDT')
        st.markdown("***")
        eth_news_sentiment()
        st.markdown("***")
        twitter_sentiment("ethereum")
        st.markdown("***")
        rf_eth()
        random_forest('ETH-USD')
        st.markdown("***")
        lstm('ETH-USD')

    if crypto == "Binance Coin (BNB)":
        get_chart('BNB-USD','BNBUSDT')
        st.markdown("***")
        bnb_news_sentiment()
        st.markdown("***")
        twitter_sentiment("binance")
        st.markdown("***")
        rf_bnb()
        random_forest('BNB-USD')
        st.markdown("***")
        lstm('BNB-USD')

    if crypto == "Solana (SOL)":
        get_chart('SOL-USD','SOLUSDT')
        st.markdown("***")
        sol_news_sentiment()
        st.markdown("***")
        twitter_sentiment("solana")
        st.markdown("***")
        rf_sol()
        random_forest('SOL-USD')
        st.markdown("***")
        lstm('SOL-USD')


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
        buy_findings = ta_5mins['signal'].str.contains('buy')
        sell_findings = ta_5mins['signal'].str.contains('sell')
        overbought_findings = ta_5mins['signal'].str.contains('overbought')
        oversold_findings = ta_5mins['signal'].str.contains('oversold')
        buy_total = buy_findings.sum()
        sell_total = sell_findings.sum()
        overbought_total = overbought_findings.sum()
        oversold_total = oversold_findings.sum()
        if (buy_total + oversold_total) > (sell_total + overbought_total):
            st.info("Technical Analysis suggests a general BUY signal")
        elif (buy_total + oversold_total) == (sell_total + overbought_total):
            st.warning("Technical Analysis suggests a general NEUTRAL signal")
        else:
            st.error("Technical Analysis suggests a general SELL signal")
        st.dataframe(ta_5mins)


    if time_int == "15 Minutes":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="15mins")
        t_indicator_5mins.to_csv("csv/ta_15mins.csv",index=False)           
        ta_15mins = pd.read_csv("csv/ta_15mins.csv")
        buy_findings = ta_15mins['signal'].str.contains('buy')
        sell_findings = ta_15mins['signal'].str.contains('sell')
        overbought_findings = ta_15mins['signal'].str.contains('overbought')
        oversold_findings = ta_15mins['signal'].str.contains('oversold')
        buy_total = buy_findings.sum()
        sell_total = sell_findings.sum()
        overbought_total = overbought_findings.sum()
        oversold_total = oversold_findings.sum()
        if (buy_total + oversold_total) > (sell_total + overbought_total):
            st.info("Technical Analysis suggests a general BUY signal")
        elif (buy_total + oversold_total) == (sell_total + overbought_total):
            st.warning("Technical Analysis suggests a general NEUTRAL signal")
        else:
            st.error("Technical Analysis suggests a general SELL signal")
        st.dataframe(ta_15mins)

    if time_int == "30 Minutes":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="30mins")
        t_indicator_5mins.to_csv("csv/ta_30mins.csv",index=False)           
        ta_30mins = pd.read_csv("csv/ta_30mins.csv")
        buy_findings = ta_30mins['signal'].str.contains('buy')
        sell_findings = ta_30mins['signal'].str.contains('sell')
        overbought_findings = ta_30mins['signal'].str.contains('overbought')
        oversold_findings = ta_30mins['signal'].str.contains('oversold')
        buy_total = buy_findings.sum()
        sell_total = sell_findings.sum()
        overbought_total = overbought_findings.sum()
        oversold_total = oversold_findings.sum()
        if (buy_total + oversold_total) > (sell_total + overbought_total):
            st.info("Technical Analysis suggests a general BUY signal")
        elif (buy_total + oversold_total) == (sell_total + overbought_total):
            st.warning("Technical Analysis suggests a general NEUTRAL signal")
        else:
            st.error("Technical Analysis suggests a general SELL signal")
        st.dataframe(ta_30mins)

    if time_int == "1 Hour":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="1hour")
        t_indicator_5mins.to_csv("csv/ta_1h.csv",index=False)           
        ta_1hr = pd.read_csv("csv/ta_1h.csv")
        buy_findings = ta_1hr['signal'].str.contains('buy')
        sell_findings = ta_1hr['signal'].str.contains('sell')
        overbought_findings = ta_1hr['signal'].str.contains('overbought')
        oversold_findings = ta_1hr['signal'].str.contains('oversold')
        buy_total = buy_findings.sum()
        sell_total = sell_findings.sum()
        overbought_total = overbought_findings.sum()
        oversold_total = oversold_findings.sum()
        if (buy_total + oversold_total) > (sell_total + overbought_total):
            st.info("Technical Analysis suggests a general BUY signal")
        elif (buy_total + oversold_total) == (sell_total + overbought_total):
            st.warning("Technical Analysis suggests a general NEUTRAL signal")
        else:
            st.error("Technical Analysis suggests a general SELL signal")
        st.dataframe(ta_1hr)

    if time_int == "Daily":
                # with st_lottie_spinner(lottie_json):
        search_result = investpy.search_quotes(text=symbol, products=['cryptos'],n_results=1)
        t_indicator_5mins = search_result.retrieve_technical_indicators(interval="daily")
        t_indicator_5mins.to_csv("csv/ta_daily.csv",index=False)           
        ta_daily = pd.read_csv("csv/ta_daily.csv")
        buy_findings = ta_daily['signal'].str.contains('buy')
        sell_findings = ta_daily['signal'].str.contains('sell')
        overbought_findings = ta_daily['signal'].str.contains('overbought')
        oversold_findings = ta_daily['signal'].str.contains('oversold')
        buy_total = buy_findings.sum()
        sell_total = sell_findings.sum()
        overbought_total = overbought_findings.sum()
        oversold_total = oversold_findings.sum()
        if (buy_total + oversold_total) > (sell_total + overbought_total):
            st.info("Technical Analysis suggests a general BUY signal")
        elif (buy_total + oversold_total) == (sell_total + overbought_total):
            st.warning("Technical Analysis suggests a general NEUTRAL signal")
        else:
            st.error("Technical Analysis suggests a general SELL signal")
        st.dataframe(ta_daily)
  

def btc_news_sentiment():
    st.subheader('News Sentiment Analysis')
    url_cryptopanic="https://cryptopanic.com/api/posts/?auth_token=aec4b1e9aec008ffa2461ec16e25f24fb11c0a5b&kind=news&currencies=btc"
    for i in range(0,20):
        try:
            response_cryptopanic = urllib.request.urlopen(url_cryptopanic)
        except IOError as e:
            print("Could not get data from cryptopanic.com")
            continue
        break

    try:
        data_cryptopanic = json.loads(response_cryptopanic.read())
        crypto_news = pd.json_normalize(data_cryptopanic['results'])
        crypto_news.to_csv("csv/news_sentiment.csv",index=False)
        Indexlist_news = ['kind','domain','title','published_at','url','source.title']
        crypto_n = pd.read_csv("csv/news_sentiment.csv", usecols=Indexlist_news)[Indexlist_news]
        crypto_n.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title']
        analyzer = SentimentIntensityAnalyzer()
        scores = crypto_n['Title'].apply(analyzer.polarity_scores)
        df_scores = pd.DataFrame(scores.apply(pd.Series))

        new_sentiment = []
        new_sentiment = crypto_n.join(df_scores, rsuffix='_right')
        new_mean_sentiment = new_sentiment['compound'].mean()
        if new_mean_sentiment > 0:
            st.info("News Sentiment Analysis using for recent news is POSITIVE")
        else:
            st.error("News Sentiment Analysis using for recent news is NEGATIVE")

        st.write("News Sentiment Score using for recent news is",new_mean_sentiment)
        st.dataframe(new_sentiment)
    except Exception as e:
        print(traceback.format_exc())

def eth_news_sentiment():
    st.subheader('News Sentiment Analysis')
    url_cryptopanic="https://cryptopanic.com/api/posts/?auth_token=aec4b1e9aec008ffa2461ec16e25f24fb11c0a5b&kind=news&currencies=ETH"
    for i in range(0,20):
        try:
            response_cryptopanic = urllib.request.urlopen(url_cryptopanic)
        except IOError as e:
            print("Could not get data from cryptopanic.com")
            continue
        break

    try:
        data_cryptopanic = json.loads(response_cryptopanic.read())
        crypto_news = pd.json_normalize(data_cryptopanic['results'])
        crypto_news.to_csv("csv/news_sentiment.csv",index=False)
        Indexlist_news = ['kind','domain','title','published_at','url','source.title']
        crypto_n = pd.read_csv("csv/news_sentiment.csv", usecols=Indexlist_news)[Indexlist_news]
        crypto_n.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title']
        analyzer = SentimentIntensityAnalyzer()
        scores = crypto_n['Title'].apply(analyzer.polarity_scores)
        df_scores = pd.DataFrame(scores.apply(pd.Series))

        new_sentiment = []
        new_sentiment = crypto_n.join(df_scores, rsuffix='_right')
        new_mean_sentiment = new_sentiment['compound'].mean()
        if new_mean_sentiment > 0:
            st.info("News Sentiment Analysis using for recent news is POSITIVE")
        else:
            st.error("News Sentiment Analysis using for recent news is NEGATIVE")

        st.write("News Sentiment Score using for recent news is",new_mean_sentiment)
        st.dataframe(new_sentiment)
    except Exception as e:
        print(traceback.format_exc())

def bnb_news_sentiment():
    st.subheader('News Sentiment Analysis')
    url_cryptopanic="https://cryptopanic.com/api/posts/?auth_token=aec4b1e9aec008ffa2461ec16e25f24fb11c0a5b&kind=news&currencies=BNB"
    for i in range(0,20):
        try:
            response_cryptopanic = urllib.request.urlopen(url_cryptopanic)
        except IOError as e:
            print("Could not get data from cryptopanic.com")
            continue
        break

    try:
        data_cryptopanic = json.loads(response_cryptopanic.read())
        crypto_news = pd.json_normalize(data_cryptopanic['results'])
        crypto_news.to_csv("csv/news_sentiment.csv",index=False)
        Indexlist_news = ['kind','domain','title','published_at','url','source.title']
        crypto_n = pd.read_csv("csv/news_sentiment.csv", usecols=Indexlist_news)[Indexlist_news]
        crypto_n.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title']
        analyzer = SentimentIntensityAnalyzer()
        scores = crypto_n['Title'].apply(analyzer.polarity_scores)
        df_scores = pd.DataFrame(scores.apply(pd.Series))

        new_sentiment = []
        new_sentiment = crypto_n.join(df_scores, rsuffix='_right')
        new_mean_sentiment = new_sentiment['compound'].mean()
        if new_mean_sentiment > 0:
            st.info("News Sentiment Analysis using for recent news is POSITIVE")
        else:
            st.error("News Sentiment Analysis using for recent news is NEGATIVE")

        st.write("News Sentiment Score using for recent news is",new_mean_sentiment)
        st.dataframe(new_sentiment)
    except Exception as e:
        print(traceback.format_exc())

def sol_news_sentiment():
    st.subheader('News Sentiment Analysis')
    url_cryptopanic="https://cryptopanic.com/api/posts/?auth_token=aec4b1e9aec008ffa2461ec16e25f24fb11c0a5b&kind=news&currencies=SOL"
    for i in range(0,20):
        try:
            response_cryptopanic = urllib.request.urlopen(url_cryptopanic)
        except IOError as e:
            print("Could not get data from cryptopanic.com")
            continue
        break

    try:
        data_cryptopanic = json.loads(response_cryptopanic.read())
        crypto_news = pd.json_normalize(data_cryptopanic['results'])
        crypto_news.to_csv("csv/news_sentiment.csv",index=False)
        Indexlist_news = ['kind','domain','title','published_at','url','source.title']
        crypto_n = pd.read_csv("csv/news_sentiment.csv", usecols=Indexlist_news)[Indexlist_news]
        crypto_n.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title']
        analyzer = SentimentIntensityAnalyzer()
        scores = crypto_n['Title'].apply(analyzer.polarity_scores)
        df_scores = pd.DataFrame(scores.apply(pd.Series))

        new_sentiment = []
        new_sentiment = crypto_n.join(df_scores, rsuffix='_right')
        new_mean_sentiment = new_sentiment['compound'].mean()
        if new_mean_sentiment > 0:
            st.info("News Sentiment Analysis using for recent news is POSITIVE")
        else:
            st.error("News Sentiment Analysis using for recent news is NEGATIVE")

        st.write("News Sentiment Score using for recent news is",new_mean_sentiment)
        st.dataframe(new_sentiment)
    except Exception as e:
        print(traceback.format_exc())

def twitter_sentiment(ticker):
    st.subheader('Twitter Sentiment Analysis')

    main(ticker)
    


class TwitterClient(object):

    def __init__(self):

        consumer_key = 'fDNn9PgvwBNf8b5YkqVdexWy8'
        consumer_secret = 'sjW9xy6ckF0wRDJKpdKNa1qoBg7lDaYk8421wSqDCloslXxWhq'
        access_token = '1530761395138359296-Q2BphWd2qzGTh0dl1AhTk2ZHiJ33uV'
        access_token_secret = 'i1D7upqRuoNHYxRE4lWluT5Z3LTQlrDxmMC0vPq4MX5Wr'

        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
  
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
  
    def get_textblob_sentiment_score(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        score = analysis.sentiment.polarity
        return score

    def get_flair_sentiment_score(self, tweet):
        sentiment_model = flair.models.TextClassifier.load('en-sentiment')
        sentence = flair.data.Sentence(self.clean_tweet(tweet))
        sentiment_model.predict(sentence)
        prob = sentence.labels[0].score
        sentiment = 1 if sentence.labels[0].value == 'POSITIVE' else -1
        flair_sentiment = prob * sentiment
        return flair_sentiment

    def get_vader_sentiment_score(self, tweet):
        vader_analyzer = SentimentIntensityAnalyzer()
        vader_sentiment = vader_analyzer.polarity_scores(self.clean_tweet(tweet)).get("compound")
        return vader_sentiment


    def get_tweets(self, query, count = 10):

        tweets = []
  
        try:
            fetched_tweets = self.api.search_tweets(q = query, count = count,lang="en")
            
            for tweet in fetched_tweets:
                tweet_date = tweet.created_at
                retweeted = tweet.retweet_count
                user = tweet.user.name
                follower = tweet.user.followers_count
                parsed_tweet = {}
                parsed_tweet['date'] = tweet_date
                parsed_tweet['text'] = tweet.text
                parsed_tweet['user'] = user 
                parsed_tweet['follower'] = follower
                parsed_tweet['retweeted_count'] = retweeted
                parsed_tweet['textblob_sentiment'] = self.get_textblob_sentiment_score(tweet.text)

                if parsed_tweet['textblob_sentiment'] > 0:
                  parsed_tweet['textblob_result'] = "POSITIVE"
                elif parsed_tweet['textblob_sentiment'] < 0:
                  parsed_tweet['textblob_result'] = "NEGATIVE"
                else:
                  parsed_tweet['textblob_result'] = "NEUTRAL"

                parsed_tweet['vader_sentiment'] = self.get_vader_sentiment_score(tweet.text)

                if parsed_tweet['vader_sentiment'] > 0:
                  parsed_tweet['vader_result'] = "POSITIVE"
                elif parsed_tweet['vader_sentiment'] < 0:
                  parsed_tweet['vader_result'] = "NEGATIVE"
                else:
                  parsed_tweet['vader_result'] = "NEUTRAL"

                if (parsed_tweet['textblob_sentiment'] > 0 and parsed_tweet['vader_sentiment'] > 0 ) > 0:
                  parsed_tweet['overall_sentiment'] = "POSITIVE"
                elif (parsed_tweet['textblob_sentiment'] < 0 and parsed_tweet['vader_sentiment'] < 0 ) < 0:
                  parsed_tweet['overall_sentiment'] = "NEGATIVE"
                else:
                  parsed_tweet['overall_sentiment'] = "NEUTRAL"

                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            return tweets
        except tweepy.errors.TweepyException as e:
            print("Error : " + str(e))



def main(ticker):
    api = TwitterClient()
    tweets = api.get_tweets(query = ticker, count = 200)
    #tweets.to_csv("csv/tweets.csv",index=False)   
    #df_tweet = pd.read_csv("csv/tweets.csv")
    df_tweet = pd.DataFrame(tweets)
    mean_sentiment = df_tweet["textblob_sentiment"].mean()
    #flair_sentiment = df_tweet["flair_sentiment"].mean()
    #st.write("The flair mean sentiment score for recent tweets is",flair_sentiment)
    vader_sentiment = df_tweet["vader_sentiment"].mean()
    if mean_sentiment > 0:
        st.info("Sentiment Analysis using Textblob NLP Model for recent Tweets is POSITIVE")
    else:
        st.error("Sentiment Analysis using Textblob NLP Model for recent Tweets is NEGATIVE")
    if vader_sentiment > 0:
        st.info("Sentiment Analysis using VADER NLP Model for recent Tweets is POSITIVE")
    else:
        st.error("Sentiment Analysis using VADER NLP Model for recent Tweets is NEGATIVE")
    st.write("The textblob mean sentiment score for recent tweets is",mean_sentiment)
    st.write("The vader mean sentiment score for recent tweets is",vader_sentiment)

    st.dataframe(df_tweet)



def random_forest(rf_ticker):

    yf.pdr_override()
    start = datetime.now() - timedelta(days=30)
    end = datetime.now() - timedelta(days=1)
    #st.info("Random Forest using Technical Indicators")
    # Get the data from remote source
    data = web.DataReader(rf_ticker, data_source ='yahoo', start=start, end=end)
    data.rename(columns = {'High':'high', 'Low':'low', 'Open':'open', 'Close':'close', 'Volume':'volume'}, inplace = True)
    data.drop(data.columns[len(data.columns)-2], axis=1, inplace=True)

    SMA = TA.SMA(data)
    SMM = TA.SMM(data)
    SSMA = TA.SSMA(data)
    EMA = TA.EMA(data)
    DEMA = TA.DEMA(data)
    TEMA = TA.TEMA(data)
    TRIMA = TA.TRIMA(data)
    TRIX = TA.TRIX(data)
    VAMA = TA.VAMA(data)
    ER = TA.ER(data)
    KAMA = TA.KAMA(data)
    ZLEMA = TA.ZLEMA(data)
    WMA = TA.WMA(data)
    HMA = TA.HMA(data)
    EVWMA = TA.EVWMA(data)
    VWAP = TA.VWAP(data)
    SMMA = TA.SMMA(data)
    FRAMA = TA.FRAMA(data)
    MACD = TA.MACD(data)
    PPO = TA.PPO(data)
    MOM = TA.MOM(data)
    ROC = TA.ROC(data)
    RSI = TA.RSI(data)
    IFT_RSI = TA.IFT_RSI(data)
    TR = TA.TR(data)
    ATR = TA.ATR(data)
    BBANDS = TA.BBANDS(data)
    BBWIDTH = TA.BBWIDTH(data)
    MOBO = TA.MOBO(data)
    PERCENT_B = TA.PERCENT_B(data)
    KC = TA.KC(data)
    DO = TA.DO(data)
    DMI = TA.DMI(data)
    ADX = TA.ADX(data)
    PIVOT = TA.PIVOT(data)
    PIVOT_FIB = TA.PIVOT_FIB(data)
    STOCH = TA.STOCH(data)
    STOCHD = TA.STOCHD(data)
    STOCHRSI = TA.STOCHRSI(data)
    WILLIAMS = TA.WILLIAMS(data)
    AO = TA.AO(data)
    MI = TA.MI(data)
    VORTEX = TA.VORTEX(data)
    KST = TA.KST(data)
    TSI = TA.TSI(data)
    TP = TA.TP(data)
    ADL = TA.ADL(data)
    CHAIKIN = TA.CHAIKIN(data)
    MFI = TA.MFI(data)
    OBV = TA.OBV(data)
    WOBV = TA.WOBV(data)
    VZO = TA.VZO(data)
    PZO = TA.PZO(data)
    EFI = TA.EFI(data)
    CFI = TA.CFI(data)
    EBBP = TA.EBBP(data)
    EMV = TA.EMV(data)
    CCI = TA.CCI(data)
    COPP = TA.COPP(data)
    BASP = TA.BASP(data)
    BASPN = TA.BASPN(data)
    CMO = TA.CMO(data)
    CHANDELIER = TA.CHANDELIER(data)
    QSTICK = TA.QSTICK(data)
    WTO = TA.WTO(data)
    FISH = TA.FISH(data)
    ICHIMOKU = TA.ICHIMOKU(data)
    APZ = TA.APZ(data)
    SQZMI = TA.SQZMI(data)
    VPT = TA.VPT(data)
    VFI = TA.VFI(data)
    MSD = TA.MSD(data)
    STC = TA.STC(data)


    data = pd.merge(data, SMA, how='inner', on = 'Date')
    data = pd.merge(data, SMM, how='inner', on = 'Date')
    data = pd.merge(data, SSMA, how='inner', on = 'Date')
    data = pd.merge(data, EMA, how='inner', on = 'Date')
    data = pd.merge(data, DEMA, how='inner', on = 'Date')
    data = pd.merge(data, TEMA, how='inner', on = 'Date')
    data = pd.merge(data, TRIMA, how='inner', on = 'Date')
    data = pd.merge(data, TRIX, how='inner', on = 'Date')
    data = pd.merge(data, VAMA, how='inner', on = 'Date')
    data = pd.merge(data, ER, how='inner', on = 'Date')
    data = pd.merge(data, KAMA, how='inner', on = 'Date')
    data = pd.merge(data, ZLEMA, how='inner', on = 'Date')
    data = pd.merge(data, WMA, how='inner', on = 'Date')
    data = pd.merge(data, HMA, how='inner', on = 'Date')
    data = pd.merge(data, EVWMA, how='inner', on = 'Date')
    data = pd.merge(data, VWAP, how='inner', on = 'Date')
    data = pd.merge(data, SMMA, how='inner', on = 'Date')
    data = pd.merge(data, FRAMA, how='inner', on = 'Date')
    data = pd.merge(data, MACD, how='inner', on = 'Date')
    data = pd.merge(data, PPO, how='inner', on = 'Date')
    data = pd.merge(data, MOM, how='inner', on = 'Date')
    data = pd.merge(data, ROC, how='inner', on = 'Date')
    data = pd.merge(data, RSI, how='inner', on = 'Date')
    data = pd.merge(data, IFT_RSI, how='inner', on = 'Date')
    data = pd.merge(data, TR, how='inner', on = 'Date')
    data = pd.merge(data, ATR, how='inner', on = 'Date')
    data = pd.merge(data, SMMA, how='inner', on = 'Date')
    data = pd.merge(data, BBANDS, how='inner', on = 'Date')
    data = pd.merge(data, BBWIDTH, how='inner', on = 'Date')
    data = pd.merge(data, MOBO, how='inner', on = 'Date')
    data = pd.merge(data, PERCENT_B, how='inner', on = 'Date')
    data = pd.merge(data, KC, how='inner', on = 'Date')
    data = pd.merge(data, DO, how='inner', on = 'Date')
    data = pd.merge(data, DMI, how='inner', on = 'Date')
    data = pd.merge(data, ADX, how='inner', on = 'Date')
    data = pd.merge(data, PIVOT, how='inner', on = 'Date')
    data = pd.merge(data, PIVOT_FIB, how='inner', on = 'Date')
    data = pd.merge(data, STOCH, how='inner', on = 'Date')
    data = pd.merge(data, STOCHD, how='inner', on = 'Date')
    data = pd.merge(data, STOCHRSI, how='inner', on = 'Date')
    data = pd.merge(data, WILLIAMS, how='inner', on = 'Date')
    data = pd.merge(data, AO, how='inner', on = 'Date')
    data = pd.merge(data, MI, how='inner', on = 'Date')
    data = pd.merge(data, VORTEX, how='inner', on = 'Date')
    data = pd.merge(data, KST, how='inner', on = 'Date')
    data = pd.merge(data, TSI, how='inner', on = 'Date')
    data = pd.merge(data, TP, how='inner', on = 'Date')
    data = pd.merge(data, ADL, how='inner', on = 'Date')
    data = pd.merge(data, CHAIKIN, how='inner', on = 'Date')
    data = pd.merge(data, MFI, how='inner', on = 'Date')
    data = pd.merge(data, OBV, how='inner', on = 'Date')
    data = pd.merge(data, WOBV, how='inner', on = 'Date')
    data = pd.merge(data, VZO, how='inner', on = 'Date')
    data = pd.merge(data, PZO, how='inner', on = 'Date')
    data = pd.merge(data, EFI, how='inner', on = 'Date')
    data = pd.merge(data, CFI, how='inner', on = 'Date')
    data = pd.merge(data, EBBP, how='inner', on = 'Date')
    data = pd.merge(data, EMV, how='inner', on = 'Date')
    data = pd.merge(data, CCI, how='inner', on = 'Date')
    data = pd.merge(data, COPP, how='inner', on = 'Date')
    data = pd.merge(data, BASP, how='inner', on = 'Date')
    data = pd.merge(data, BASPN, how='inner', on = 'Date')
    data = pd.merge(data, CMO, how='inner', on = 'Date')
    data = pd.merge(data, CHANDELIER, how='inner', on = 'Date')
    data = pd.merge(data, QSTICK, how='inner', on = 'Date')
    data = pd.merge(data, WTO, how='inner', on = 'Date')
    data = pd.merge(data, FISH, how='inner', on = 'Date')
    data = pd.merge(data, ICHIMOKU, how='inner', on = 'Date')
    data = pd.merge(data, APZ, how='inner', on = 'Date')
    data = pd.merge(data, SQZMI, how='inner', on = 'Date')
    data = pd.merge(data, VPT, how='inner', on = 'Date')
    data = pd.merge(data, VFI, how='inner', on = 'Date')
    data = pd.merge(data, MSD, how='inner', on = 'Date')
    data = pd.merge(data, STC, how='inner', on = 'Date')
    st.dataframe(data)


def rf_btc():
  try:
    st.subheader('Random Forest Model')
    base_url = "https://us-central1-fintechcoo.cloudfunctions.net/getRandomForestResult?ticker=BTC-USD"
    rf_input_date = datetime.today().strftime('%Y-%m-%d')
    response = (base_url + '&date=' + rf_input_date)
    url_response = urlopen(response)
    data_json = json.loads(url_response.read())
    rf_prediction = data_json["prediction"]
    rf_accuracy = data_json["accuracy"]
    rf_date = data_json["date"]
    rf_ticker = data_json["ticker"]
    if rf_prediction == "increase":
        st.info("The Random Forest Model predicts there will be increase in price of BTC-USD 3 days after.")
    else:
        st.error("The Random Forest Model predicts there will be decrease in price of BTC-USD 3 days after.")
    #st.write("The Random Forest Model predicts there will be " , rf_prediction , " in price of " , rf_ticker , " 3 days after.")
    st.write("The Accuracy of the Random Forest Model is" , rf_accuracy )
  except:
    #print("Errors in getting the values")
    pass

def rf_eth():
  try:
    st.subheader('Random Forest Model')
    base_url = "https://us-central1-fintechcoo.cloudfunctions.net/getRandomForestResult?ticker=ETH-USD"
    rf_input_date = datetime.today().strftime('%Y-%m-%d')
    response = (base_url + '&date=' + rf_input_date)
    url_response = urlopen(response)
    data_json = json.loads(url_response.read())
    rf_prediction = data_json["prediction"]
    rf_accuracy = data_json["accuracy"]
    rf_date = data_json["date"]
    rf_ticker = data_json["ticker"]
    if rf_prediction == "increase":
        st.info("The Random Forest Model predicts there will be increase in price of ETH-USD 3 days after.")
    else:
        st.error("The Random Forest Model predicts there will be decrease in price of ETH-USD 3 days after.")
    #st.write("The Random Forest Model predicts there will be " , rf_prediction , " in price of " , rf_ticker , " 3 days after.")
    st.write("The Accuracy of the Random Forest Model is" , rf_accuracy )
  except:
    #print("Errors in getting the values")
    pass

def rf_bnb():
  try:
    st.subheader('Random Forest Model')
    base_url = "https://us-central1-fintechcoo.cloudfunctions.net/getRandomForestResult?ticker=BNB-USD"
    rf_input_date = datetime.today().strftime('%Y-%m-%d')
    response = (base_url + '&date=' + rf_input_date)
    url_response = urlopen(response)
    data_json = json.loads(url_response.read())
    rf_prediction = data_json["prediction"]
    rf_accuracy = data_json["accuracy"]
    rf_date = data_json["date"]
    rf_ticker = data_json["ticker"]
    if rf_prediction == "increase":
        st.info("The Random Forest Model predicts there will be increase in price of BNB-USD 3 days after.")
    else:
        st.error("The Random Forest Model predicts there will be decrease in price of BNB-USD 3 days after.")
    #st.write("The Random Forest Model predicts there will be " , rf_prediction , " in price of " , rf_ticker , " 3 days after.")
    st.write("The Accuracy of the Random Forest Model is" , rf_accuracy )
  except:
    #print("Errors in getting the values")
    pass

def rf_sol():
  try:
    st.subheader('Random Forest Model')
    base_url = "https://us-central1-fintechcoo.cloudfunctions.net/getRandomForestResult?ticker=SOL-USD"
    rf_input_date = datetime.today().strftime('%Y-%m-%d')
    response = (base_url + '&date=' + rf_input_date)
    url_response = urlopen(response)
    data_json = json.loads(url_response.read())
    rf_prediction = data_json["prediction"]
    rf_accuracy = data_json["accuracy"]
    rf_date = data_json["date"]
    rf_ticker = data_json["ticker"]
    if rf_prediction == "increase":
        st.info("The Random Forest Model predicts there will be increase in price of SOL-USD 3 days after.")
    else:
        st.error("The Random Forest Model predicts there will be decrease in price of SOL-USD 3 days after.")
    #st.write("The Random Forest Model predicts there will be " , rf_prediction , " in price of " , rf_ticker , " 3 days after.")
    st.write("The Accuracy of the Random Forest Model is" , rf_accuracy )
  except:
    #print("Errors in getting the values")
    pass

def lstm(ticker):
  try:
    st.subheader("LSTM Model")
    st.info("LSTM model trend prediction 3 days after")
    lstm_price_url = "https://us-central1-fintechcoo.cloudfunctions.net/getLSTMResult"
    lstm_prediction_url = "https://us-central1-fintechcoo.cloudfunctions.net/getLSTMPriceResult"
    lstm_input_date = datetime.today().strftime('%Y-%m-%d')
    #response = (lstm_price_url + '&date=' + lstm_input_date)
    response = (lstm_price_url + '?ticker=' + ticker + '&date=' + lstm_input_date)
    url_response = urlopen(response)
    data_lstm_price = json.loads(url_response.read())
    lstm_prediction = data_lstm_price["prediction"]
    lstm_accuracy = data_lstm_price["accuracy"]
    lstm_date = data_lstm_price["date"]
    lstm_ticker = data_lstm_price["ticker"]
    st.write("The LSTM Model predicts there will be " , lstm_prediction , " in price of " , lstm_ticker , " 3 days after.")
    st.write("The Accuracy of the LSTM Model is" , lstm_accuracy )
    st.write("")
    
    st.info("LSTM model next day price prediction")
    prediction_response = (lstm_prediction_url + '?ticker=' + ticker + '&date=' + lstm_input_date)
    url_response2 = urlopen(prediction_response)
    data_lstm_predict = json.loads(url_response2.read())
    lstm_predict_ticker = data_lstm_predict["ticker"]
    lstm_figure_priceall = data_lstm_predict["figure_price_all"]
    lstm_figure_price = data_lstm_predict["figure_price"]
    lstm_loss_value = data_lstm_predict["loss_value"]
    lstm_predict = data_lstm_predict["predicted_price"]
    lstm_figure_loss = data_lstm_predict["figure_loss_value"]
    lstm_predict_date = data_lstm_predict["date"]
    st.write("The LSTM Model predicts the next date price for ", lstm_predict_ticker , "is: ", lstm_predict)
    st.write("The loss value of the LSTM Model is: ", lstm_loss_value)
    st.write("")

    st.info("LSTM Model price prediction - All Period")
    urllib.request.urlretrieve(lstm_figure_priceall , 'demo_price_all.png')
    img_priceall = Image.open('demo_price_all.png')
    img_priceall = img_priceall.save("Images/lstm_priceall.png")
    image_lstm_priceall = Image.open('Images/lstm_priceall.png')
    
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        st.write(' ')

    with col2:
        st.image(image_lstm_priceall, caption='LSTM Model price prediction - All Period')

    with col3:
        st.write(' ')
    
        

    st.info("LSTM Model price prediction")
    urllib.request.urlretrieve(lstm_figure_price , 'demo_price.png')
    img_price = Image.open('demo_price.png')
    img_price = img_price.save("Images/lstm_price.png")
    image_lstm_price = Image.open('Images/lstm_price.png')
    
    col4, col5, col6 = st.columns([1, 5, 1])
    with col4:
        st.write(' ')

    with col5:
        st.image(image_lstm_price, caption='LSTM Model price prediction')

    with col6:
        st.write(' ')

    st.info("LSTM Model loss value")
    urllib.request.urlretrieve(lstm_figure_loss , 'demo_loss_value.png')
    img_lstm_loss = Image.open('demo_loss_value.png')
    img_lstm_loss = img_lstm_loss.save("Images/lstm_loss.png")
    image_lstm_loss = Image.open('Images/lstm_loss.png')
    #st.image(image_lstm_loss, caption='LSTM Model loss value')
    #img_lstm_loss = img.save("lstm_loss.png") 
    col7, col8, col9 = st.columns([1, 5, 1])
    with col7:
        st.write(' ')

    with col8:
        st.image(image_lstm_loss, caption='LSTM Model loss value')

    with col9:
        st.write(' ')


  except:
    #print("Errors in getting the values")
    pass



