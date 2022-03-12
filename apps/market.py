import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt
import talib 
import matplotlib.ticker as mticker
import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup as soup
import urllib.request
from urllib.request import Request, urlopen
import plotly.graph_objects as go
import pytrends
import csv
import json
import time
from whalealert.whalealert import WhaleAlert
import sys, traceback


def market(url):

    parameters = {
        'start': '1',
        'limit': '10',
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '6d6d5a32-8c60-4040-8f72-095ea1bb5b19',
    }

    resp = requests.get(url, params=parameters, headers=headers)
    jsondata = json.loads(resp.text)
    CoinDF = pd.json_normalize(jsondata['data'])

    CoinDF.to_csv("market.csv",index=False)

def totalmarket(url2):

    parameters = {
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '6d6d5a32-8c60-4040-8f72-095ea1bb5b19',
    }

    resp = requests.get(url2, params=parameters, headers=headers)
    jsondata = json.loads(resp.text)
    totalCoinDF = pd.json_normalize(jsondata['data'])

    totalCoinDF.to_csv("totalmarket.csv",index=False)

def transaction():

    whale = WhaleAlert()

    # Specify transactions from the last 10 minutes
    start_time = int(time.time() - 600)
    api_key = 'ZilOPOaaJ7D3oYk8sM9CS3KjyrCeBvGx'
    transaction_count_limit = 30
    success, transactions, status = whale.get_transactions(start_time, api_key=api_key, limit=transaction_count_limit)
    df_trans = pd.DataFrame(transactions,index=range(len(transactions)))
    df_status = pd.DataFrame(status,index=range(len(status)))
    df_trans.to_csv("transaction.csv",index=False)
    df_status.to_csv("status.csv",index=False)

def color_df(val):
    color = 'green' if val>0 else 'red'
    return f'color: {color}'


def news():

    url_cryptopanic="https://cryptopanic.com/api/posts/?auth_token=aec4b1e9aec008ffa2461ec16e25f24fb11c0a5b&kind=news"
    for i in range(0,10):
        try:
            response_cryptopanic = urllib.request.urlopen(url_cryptopanic)
        except IOError as e:
            print("Could not get data from cryptopanic.com")
            continue
        break

    try:
        data_cryptopanic = json.loads(response_cryptopanic.read())
        crypto_news = pd.json_normalize(data_cryptopanic['results'])
        crypto_news.to_csv("news.csv",index=False)
    except Exception as e:
        print(traceback.format_exc())

def media():

    url_cryptopanic1="https://cryptopanic.com/api/posts/?auth_token=aec4b1e9aec008ffa2461ec16e25f24fb11c0a5b&kind=media"
    for i in range(0,10):
        try:
            response_cryptopanic1 = urllib.request.urlopen(url_cryptopanic1)
        except IOError as e:
            print("Could not get data from cryptopanic.com")
            continue
        break

    try:
        data_cryptopanic1 = json.loads(response_cryptopanic1.read())
        crypto_media = pd.json_normalize(data_cryptopanic1['results'])
        crypto_media.to_csv("media.csv",index=False)
    except Exception as e:
        print(traceback.format_exc())


def etf():
    import csv
    from bs4 import BeautifulSoup


    url = "https://etfdb.com/themes/blockchain-etfs/"
    page = requests.get(url)
    html = page.text

    outfile = open("table_data.csv","w",newline='')
    writer = csv.writer(outfile)

    tree = BeautifulSoup(html,"lxml")
    table_tag = tree.select("table")[0]
    tab_data = [[item.text for item in row_data.select("th,td")]
                    for row_data in table_tag.select("tr")]
    for data in tab_data:
        writer.writerow(data)
        #print(' '.join(data))



def app():

    st.title('Crypto Market At-A-Glance')
    st.write("*(Results might have a few mins delay. Pricing data is updated frequently.)")
    st.markdown("***")

    #Market Infographics
    #coinmarketcap
    st.subheader('Market Infographics')
    url2 = 'https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest'

    #totalmarket(url2)
    Indexlist1 = ['quote.USD.total_market_cap','quote.USD.total_market_cap_yesterday','quote.USD.total_market_cap_yesterday_percentage_change' ]
    Indexlist2 = ['quote.USD.total_volume_24h','quote.USD.total_volume_24h_yesterday','quote.USD.total_volume_24h_yesterday_percentage_change']
    Indexlist3 = ['active_cryptocurrencies','total_cryptocurrencies','active_market_pairs']

    total_crypto1 = pd.read_csv("totalmarket.csv", usecols=Indexlist1)[Indexlist1]
    total_crypto2 = pd.read_csv("totalmarket.csv", usecols=Indexlist2)[Indexlist2]
    total_crypto3 = pd.read_csv("totalmarket.csv", usecols=Indexlist3)[Indexlist3]

    total_crypto1.columns = ['Total Market Cap', 'Total Market Cap Yesterday', 'Total Market Cap Yesterday %_change']
    total_crypto2.columns = ['Total Volume (24 hrs)','Total Volume (24 hrs) Yesterday','Total Volume (24 hrs) Yesterday %_change']
    total_crypto3.columns = ['Active Cryptocurrencies','Total Cryptocurrencies','Active Market Pairs']

    #st.table(total_crypto1.style.applymap(color_df, subset=['Total Market Cap Yesterday %_change']))
    #st.table(total_crypto2.style.applymap(color_df, subset=['Total Volume (24 hrs) Yesterday %_change']))
    #st.table(total_crypto3)

    metric1 = total_crypto1.iloc[0]['Total Market Cap']
    metric2 = total_crypto1.iloc[0]['Total Market Cap Yesterday %_change']
    metric3 = total_crypto2.iloc[0]['Total Volume (24 hrs)']
    metric4 = total_crypto2.iloc[0]['Total Volume (24 hrs) Yesterday %_change']
    metric5 = total_crypto3.iloc[0]['Active Cryptocurrencies']
    metric6 = total_crypto3.iloc[0]['Total Cryptocurrencies']
    metric7 = total_crypto3.iloc[0]['Active Market Pairs']

    metric1_f = '${:,.2f}'.format(metric1)
    metric2_f = '{:,.2f}%'.format(metric2)
    metric3_f = '{:,.2f}'.format(metric3)
    metric4_f = '{:,.2f}%'.format(metric4)
    metric5_f = '{:,}'.format(metric5)
    metric6_f = '{:,}'.format(metric6)
    metric7_f = '{:,}'.format(metric7)

    col1, col2 = st.columns(2)
    col1.metric('Total Crypto Market Capitalization (USD)', metric1_f, metric2_f)
    col2.metric('Total Market Volume (24 hrs)', metric3_f, metric4_f)
    col3, col4, col5 = st.columns(3)
    col3.metric('Active Cryptocurrencies', metric5_f)
    col4.metric('Total Cryptocurrencies', metric6_f)    
    col5.metric('Active Market Pairs', metric7_f)  
    st.markdown("***")


    #Top 10 Cryptocurrencies
    st.subheader('Top 10 Cryptocurrencies')
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    #market(url) 

    #Plot Bar Chart for Top 10 Cryptocurrencies
    top10_simple = pd.read_csv("market.csv", usecols=['symbol','quote.USD.market_cap_dominance'])
    top_simple = top10_simple.iloc[0:]
    top_simple.columns = ['Symbol','Market Dominance']
    other = (100 - top_simple['Market Dominance'].sum())    
    df_other = {'Symbol':'Other','Market Dominance': other }
    top_simple = top_simple.append(df_other, ignore_index = True)
    top_simple.groupby(['Symbol']).sum().plot(kind='pie', y='Market Dominance', autopct='%1.0f%%').figure.savefig('market_dominance.png')
    from PIL import Image
    market_dominance = Image.open('market_dominance.png')
    st.image(market_dominance)

    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    #st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    Indexlist = ['cmc_rank','name','symbol','quote.USD.market_cap','quote.USD.price','quote.USD.percent_change_1h','quote.USD.percent_change_24h','quote.USD.percent_change_7d','quote.USD.percent_change_30d','quote.USD.percent_change_60d', 'quote.USD.percent_change_90d','quote.USD.volume_24h','quote.USD.volume_change_24h' ]
    pd.set_option('display.precision', 2)
    top10_crypto = pd.read_csv("market.csv", usecols=Indexlist)[Indexlist]
    top = top10_crypto.iloc[0:]
    top.columns = ['Rank', 'Name', 'Symbol','Market Cap', 'Price (USD)',  'Price %_Change (1 hr)', 'Price %_Change (24 hrs)', 'Price %_Change (7 days)', 'Price %_Change (30 days)', 'Price %_Change (60 days)', 'Price %_Change (90 days)','Volume(24 hrs)', 'Volume %_Change (24 hrs)']

    st.dataframe(top.style.applymap(color_df, subset=['Price %_Change (1 hr)', 'Price %_Change (24 hrs)', 'Price %_Change (7 days)', 'Price %_Change (30 days)', 'Price %_Change (60 days)', 'Price %_Change (90 days)','Volume %_Change (24 hrs)']),height=350)
    st.markdown("***")


    #Whale
    #From whale-alert
    #transaction()
    st.subheader("Whale Alerts")
    st.write("The term “whale” is used to describe an individual or organization that holds a large amount of a particular cryptocurrency.")
    st.write("This section will display recent 30 whale alerts with transaction amount > US$500K.(Source: whale-alert.io)")
    #Indexlist_trans = ['blockchain','symbol','id','transaction_type','hash','from','to','timestamp','amount','amount_usd','transaction_count']

    crypto_trans = pd.read_csv("transaction.csv")
    crypto_trans.columns = ['Blockchain','Symbol','ID','Transaction Type','Hash','From','To','Timestamp','Amount','Amount (USD)','Transaction Count']
    crypto_trans['Timestamp'] = pd.to_datetime(crypto_trans['Timestamp'],unit='s')
    crypto_trans['Timestamp'] = crypto_trans['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(crypto_trans)
    st.markdown("***")
    
    #Cryptocurrency News
    st.subheader("Market News and Media")
    st.write("This section will display the recent market news and published media related to Crypto Market. (Source: cryptopanic.com)")
    #news()
    #media()
    Indexlist_news = ['kind','domain','title','published_at','url','source.title','votes.negative','votes.positive','votes.important','votes.liked','votes.disliked','currencies']
    crypto_news = pd.read_csv("news.csv", usecols=Indexlist_news)
    crypto_news.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title','Negative Votes','Positive Votes','Important Votes','Likes Count','Dislikes Count','Currencies']
    st.dataframe(crypto_news)

    st.subheader("Media Podcast")
    Indexlist_media = ['kind','domain','title','published_at','url','source.title','votes.negative','votes.positive','votes.important','votes.liked','votes.disliked','currencies']
    crypto_media = pd.read_csv("media.csv", usecols=Indexlist_media)[Indexlist_media]
    crypto_media.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title','Negative Votes','Positive Votes','Important Votes','Likes Count','Dislikes Count','Currencies']
    st.dataframe(crypto_media)
    st.markdown("***")  

    #Nasdaq Crypto index (NCID)
    st.subheader("Nasdaq Crypto index (NCID)")

    st.markdown("***")

    #Cryptocurrency ETFs
    st.subheader("Blockchain/ Cryptocurrency related ETFs Performance")
    st.write("This is a list of major Blockchain ETFs traded in the USA.(* Assets in thousands of U.S. Dollars.)(Source: etfdb.com) ")
    #etf()
    crypto_etf = pd.read_csv("table_data.csv")
    crypto_etf.drop(crypto_etf.tail(1).index,inplace=True)
    crypto_etf = crypto_etf.iloc[: , :-24]
    st.dataframe(crypto_etf)
    #   






