import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt
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
import streamlit.components.v1 as components

import warnings
warnings.filterwarnings('ignore')  # Hide warnings

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px
import investpy




st.set_option('deprecation.showPyplotGlobalUse', False)

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

    CoinDF.to_csv("csv/market.csv",index=False)

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

    totalCoinDF.to_csv("csv/totalmarket.csv",index=False)

def transaction():

    whale = WhaleAlert()

    # Specify transactions from the last 10 minutes
    start_time = int(time.time() - 600)
    api_key = 'ZilOPOaaJ7D3oYk8sM9CS3KjyrCeBvGx'
    transaction_count_limit = 30
    success, transactions, status = whale.get_transactions(start_time, api_key=api_key, limit=transaction_count_limit)
    df_trans = pd.DataFrame(transactions,index=range(len(transactions)))
    df_status = pd.DataFrame(status,index=range(len(status)))
    df_trans.to_csv("csv/transaction.csv",index=False)
    df_status.to_csv("csv/status.csv",index=False)

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
        crypto_news.to_csv("csv/news.csv",index=False)
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
        crypto_media.to_csv("csv/media.csv",index=False)
    except Exception as e:
        print(traceback.format_exc())

def nci():
    today = datetime.today().strftime('%d/%m/%Y')
    search_result = investpy.search_quotes(text='ncid', products=['indices'], countries=['united states'], n_results=1)

    historical_data = search_result.retrieve_historical_data(from_date='16/04/2021', to_date=today)
    historical_data.to_csv('csv/ncid1.csv')
    t_indicator_1h = search_result.retrieve_technical_indicators(interval="1hour")
    t_indicator_1h.to_csv("csv/ncid_1h.csv",index=False)     
    
    t_indicator_1d = search_result.retrieve_technical_indicators(interval="daily")
    t_indicator_1d.to_csv("csv/ncid_1d.csv",index=False)  
    

def etf():
    import csv
    from bs4 import BeautifulSoup


    url = "https://etfdb.com/themes/blockchain-etfs/"
    page = requests.get(url)
    html = page.text

    outfile = open("csv/table_data.csv","w",newline='')
    writer = csv.writer(outfile)

    tree = BeautifulSoup(html,"lxml")
    table_tag = tree.select("table")[0]
    tab_data = [[item.text for item in row_data.select("th,td")]
                    for row_data in table_tag.select("tr")]
    for data in tab_data:
        writer.writerow(data)
        #print(' '.join(data))



def get_market():

    # st.title('Crypto Market At-A-Glance')
    # st.write("*(Results might have a few mins delay. Pricing data is updated frequently.)")
    # st.markdown("***")

    #Market Infographics
    #coinmarketcap
    st.subheader('Market Infographics')
    url2 = 'https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest'

    totalmarket(url2)
    Indexlist1 = ['quote.USD.total_market_cap','quote.USD.total_market_cap_yesterday','quote.USD.total_market_cap_yesterday_percentage_change' ]
    Indexlist2 = ['quote.USD.total_volume_24h','quote.USD.total_volume_24h_yesterday','quote.USD.total_volume_24h_yesterday_percentage_change']
    Indexlist3 = ['active_cryptocurrencies','total_cryptocurrencies','active_market_pairs']

    total_crypto1 = pd.read_csv("csv/totalmarket.csv", usecols=Indexlist1)[Indexlist1]
    total_crypto2 = pd.read_csv("csv/totalmarket.csv", usecols=Indexlist2)[Indexlist2]
    total_crypto3 = pd.read_csv("csv/totalmarket.csv", usecols=Indexlist3)[Indexlist3]

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
    market(url) 

    #Plot Bar Chart for Top 10 Cryptocurrencies
    top10_simple = pd.read_csv("csv/market.csv", usecols=['symbol','quote.USD.market_cap_dominance'])
    top_simple = top10_simple.iloc[0:]
    top_simple.columns = ['Symbol','Market Dominance']
    other = (100 - top_simple['Market Dominance'].sum())    
    df_other = {'Symbol':'Other','Market Dominance': other }
    top_simple = top_simple.append(df_other, ignore_index = True)
    top_simple.groupby(['Symbol']).sum().plot(kind='pie', y='Market Dominance', autopct='%1.0f%%').figure.savefig('Images/market_dominance.png')
    
    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write(' ')

    with col2:
        from PIL import Image
        image = Image.open('Images/market_dominance.png')
        st.image(image)

    with col3:
        st.write(' ')



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
    top10_crypto = pd.read_csv("csv/market.csv", usecols=Indexlist)[Indexlist]
    top = top10_crypto.iloc[0:]
    top.columns = ['Rank', 'Name', 'Symbol','Market Cap', 'Price (USD)',  'Price %_Change (1 hr)', 'Price %_Change (24 hrs)', 'Price %_Change (7 days)', 'Price %_Change (30 days)', 'Price %_Change (60 days)', 'Price %_Change (90 days)','Volume(24 hrs)', 'Volume %_Change (24 hrs)']


    st.dataframe(top.style.applymap(color_df, subset=['Price %_Change (1 hr)', 'Price %_Change (24 hrs)', 'Price %_Change (7 days)', 'Price %_Change (30 days)', 'Price %_Change (60 days)', 'Price %_Change (90 days)','Volume %_Change (24 hrs)']),height=350)
    
    st.markdown("***")

    #Heatmap
    st.subheader('Heatmap')
    components.html(
        """
        <qc-heatmap height="380px" num-of-coins="10" currency-code="USD"></qc-heatmap>
        <script src="https://quantifycrypto.com/widgets/heatmaps/js/qc-heatmap-widget.js"></script>
        """,height= 400
    )    
    st.markdown("***")   
    
    
    #Correlation Matrix
    st.subheader('Correlation Matrix')
    st.write("Please select appropriate dates")
    start = st.date_input(
        "Input start date",
        date(2021, 1, 1))
    end = st.date_input(
        "Input end date",
        date.today())

    btc = yf.download("BTC-USD", start=start, end=end)
    btc.reset_index(inplace=True)
    crypto= btc[['Date','Adj Close']]
    crypto= crypto.rename(columns = {'Adj Close':'BTC'})
    
    eth = yf.download("ETH-USD", start=start, end=end)
    eth.reset_index(inplace=True)
    crypto["ETH"]= eth["Adj Close"]
    
    bnb = yf.download("BNB-USD", start=start, end=end)
    bnb.reset_index(inplace=True)
    crypto["BNB"]= bnb["Adj Close"]
    
    sol = yf.download("SOL-USD", start=start, end=end)
    sol.reset_index(inplace=True)
    crypto["SOL"]= sol["Adj Close"] 
    
    crypto.set_index("Date", inplace=True)

    # Scaled Price
    st.subheader("Scaled Price")
    new = crypto[['BTC', 'ETH', 'BNB','SOL']].copy()
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    scaled = min_max_scaler.fit_transform(new)
    df_scale = pd.DataFrame(scaled, columns = new.columns)
    #Visualize the scaled data
    my_crypto = df_scale
    plt.figure(figsize=(12.4, 4.5))
    for c in my_crypto.columns.values:
       plt.plot(my_crypto[c], label=c)
    plt.title('Cryptocurrency Scaled Graph')
    plt.xlabel('Days')
    plt.ylabel('Crypto Scaled Price ($)')
    plt.legend(my_crypto.columns.values, loc = 'upper left')
    #plt.show()
    
    plt.savefig('Images/crypto_scaled.png')
    from PIL import Image
    image_scaled = Image.open('Images/crypto_scaled.png')
    st.image(image_scaled, caption='Prices of the cryptos are scaled between 0 to 100')
    
    # Matrix
    plt.figure(figsize = (5,5))
    corr_matrix = sns.heatmap(crypto[['BTC','ETH','BNB','SOL']].corr(),annot=True, cmap='Oranges')
    
    fig_corr = corr_matrix.get_figure()
    fig_corr.savefig('Images/corr.png')
    
    
    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write(' ')

    with col2:
        from PIL import Image
        image_corr = Image.open("Images/corr.png")
        st.image(image_corr, caption='Correlation Matrix based on adjusted close price')


    with col3:
        st.write(' ')



    st.markdown("***") 
    
    #Whale
    #From whale-alert
    transaction()
    st.subheader("Whale Alerts")
    st.write("The term “whale” is used to describe an individual or organization that holds a large amount of a particular cryptocurrency.")
    st.write("This section will display recent 30 whale alerts with transaction amount > US$500K.(Source: whale-alert.io)")
    #Indexlist_trans = ['blockchain','symbol','id','transaction_type','hash','from','to','timestamp','amount','amount_usd','transaction_count']

    crypto_trans = pd.read_csv("csv/transaction.csv")
    crypto_trans.columns = ['Blockchain','Symbol','ID','Transaction Type','Hash','From','To','Timestamp','Amount','Amount (USD)','Transaction Count']
    crypto_trans['Timestamp'] = pd.to_datetime(crypto_trans['Timestamp'],unit='s')
    crypto_trans['Timestamp'] = crypto_trans['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(crypto_trans)
    st.markdown("***")
    
    #Cryptocurrency News
    st.subheader("Market News and Media")
    st.write("This section will display the recent market news and published media related to Crypto Market. (Source: cryptopanic.com)")
    news()
    media()
    Indexlist_news = ['kind','domain','title','published_at','url','source.title','votes.negative','votes.positive','votes.important','votes.liked','votes.disliked','currencies']
    crypto_news = pd.read_csv("csv/news.csv", usecols=Indexlist_news)
    crypto_news.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title','Negative Votes','Positive Votes','Important Votes','Likes Count','Dislikes Count','Currencies']
    st.dataframe(crypto_news)

    st.subheader("Media Podcast")
    Indexlist_media = ['kind','domain','title','published_at','url','source.title','votes.negative','votes.positive','votes.important','votes.liked','votes.disliked','currencies']
    crypto_media = pd.read_csv("csv/media.csv", usecols=Indexlist_media)[Indexlist_media]
    crypto_media.columns = ['Kind','Source Domain','Title','Published Time','URL','Source Title','Negative Votes','Positive Votes','Important Votes','Likes Count','Dislikes Count','Currencies']
    st.dataframe(crypto_media)
    st.markdown("***")  

    #Nasdaq Crypto index (NCID)
    st.subheader("Nasdaq Crypto index (NCID)")
    st.write("The Nasdaq Crypto Index is designed to measure the performance of a material portion of the overall digital asset market. Digital assets are eligible for inclusion in the Index if they satisfy the criteria set forth under “Index Eligibility.” The Index periodically adjusts Index Constituents and weightings to reflect changes in the digital asset market.")
    st.write("For more details, please visit https://indexes.nasdaq.com/docs/methodology_NCIS.pdf")
    nci()
    ncid = pd.read_csv("csv/ncid1.csv")
    #print(ncid.at[165, 'Low'])
    ncid.at[165, "Low"] = 3159.8
    fig = go.Figure(data=[go.Candlestick(x=ncid['Date'],
                open=ncid['Open'],
                high=ncid['High'],
                low=ncid['Low'],
                close=ncid['Close'])])
    st.plotly_chart(fig, use_container_width=True)

    nci_1h = pd.read_csv("csv/ncid_1h.csv")
    nci_1d = pd.read_csv("csv/ncid_1d.csv")
    nci_merge = pd.merge(nci_1h, nci_1d, on="indicator")
    nci_merge.columns = ['Technical Indicators','Signal (1 Hour)','Value (1 Hour)','Signal (1 Day)','Value (1 Day)']
    st.dataframe(nci_merge)
    #result = pd.merge(left, right, on="key")

    st.markdown("***")

    #Cryptocurrency ETFs
    st.subheader("Blockchain/ Cryptocurrency related ETFs Performance")
    st.write("This is a list of major Blockchain ETFs traded in the USA.(* Assets in thousands of U.S. Dollars.)(Source: etfdb.com) ")
    etf()
    crypto_etf = pd.read_csv("csv/table_data.csv")
    crypto_etf.drop(crypto_etf.tail(1).index,inplace=True)
    crypto_etf = crypto_etf.iloc[: , :-24]
    st.dataframe(crypto_etf)
    
        

