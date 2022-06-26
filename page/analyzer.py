import streamlit as st
import requests
from streamlit_lottie import st_lottie_spinner
import streamlit.components.v1 as components
from pytrends.request import TrendReq
from pytrends import dailydata
import yfinance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from PIL import Image
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates


def get_analyzer():

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # lottie_url = "https://assets7.lottiefiles.com/packages/lf20_p9ce3saq.json"
    # lottie_json = load_lottieurl(lottie_url)

    st.warning("Under Construction")


    with st.form(key='my_form'):
        ticker = st.selectbox('Select Ticker',
                              ['BTC/USD','ETH/USD','BNB/USD','SOL/USD','BNB/BTC','BNB/ETH','ETH/BTC', 'SOL/BNB', 'SOL/BTC'])
        #period = st.selectbox("Period", ('1day','1h','1min'))
        width = st.slider("Set the width of the real-time chart ", 200, 1800, 1100, 100) 
        submit_button = st.form_submit_button(label='Submit')
    height = 800
  
    if submit_button:
        if ticker == "BTC/USD":      
            get_btcusd(height, width)   
        if ticker == "ETH/USD":          
            get_ethusd(height, width)
        if ticker == "BNB/USD":
            get_bnbusd(height, width)
        if ticker == "SOL/USD":
            get_solusd(height, width)
        if ticker == "BNB/BTC":
            get_bnbbtc(height, width)
        if ticker == "BNB/ETH":
            get_bnbeth(height, width)
        if ticker == "ETH/BTC":
            get_ethbtc(height, width)        
        if ticker == "SOL/ETH":
            get_soleth(height, width)
        if ticker == "SOL/BTC":
            get_solbtc(height, width)



def get_btcusd(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:BTCUSD",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )
    st.markdown("***")

    #Candlestick Pattern Analysis
    st.subheader('Candlestick Pattern Analysis')
    st.info("Candlestick Pattern Analysis")
    from candlestick import candlestick

    # Find candles where inverted hammer is detected

    candles = requests.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=30')
    candles_dict = candles.json()

    candles_df = pd.DataFrame(candles_dict,
                           columns=['open_time', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_asset_volume', 'number_of_trades','taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'ignore'])

    candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms')
    candles_df['close_time'] = pd.to_datetime(candles_df['close_time'], unit='ms')

    candles_df = candlestick.inverted_hammer(candles_df, target='InvertedHammers')
    candles_df = candlestick.doji_star(candles_df, target='doji_star')
    candles_df = candlestick.bearish_harami(candles_df, target='bearish_harami')
    candles_df = candlestick.bullish_harami(candles_df, target='bullish_harami')
    candles_df = candlestick.dark_cloud_cover(candles_df, target='dark_cloud_cover')
    candles_df = candlestick.doji(candles_df, target='doji')
    candles_df = candlestick.dragonfly_doji(candles_df, target='dragonfly_doji')
    candles_df = candlestick.hanging_man(candles_df, target='hanging_man')
    candles_df = candlestick.gravestone_doji(candles_df, target='gravestone_doji')
    candles_df = candlestick.bearish_engulfing(candles_df, target='bearish_engulfing')
    candles_df = candlestick.bullish_engulfing(candles_df, target='bullish_engulfing')
    candles_df = candlestick.hammer(candles_df, target='hammer')
    candles_df = candlestick.morning_star(candles_df, target='morning_star')
    candles_df = candlestick.morning_star_doji(candles_df, target='morning_star_doji')
    candles_df = candlestick.piercing_pattern(candles_df, target='piercing_pattern')
    candles_df = candlestick.rain_drop(candles_df, target='rain_drop')
    candles_df = candlestick.rain_drop_doji(candles_df, target='rain_drop_doji')
    candles_df = candlestick.star(candles_df, target='star')
    candles_df = candlestick.shooting_star(candles_df, target='shooting_star')

    st.dataframe(candles_df)
    st.markdown("***")

    #Google Trend
    st.subheader('Google Trend Analysis')
    st.info("Google Trends values are calculated on a scale from 0 to 100. A higher value indicates the search item is more popular. A value of 0 indicates there was not enough data for the search items.")

    with st.expander(" 👁  Click to view the recent 7-days Google trend for Bitcoin"):  
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = ["Bitcoin","BTC"]
        pytrends.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')
        df_pytrends = pytrends.interest_over_time()
        st.dataframe(df_pytrends)


    with st.expander(" 👁  Click to view the correlation of recent 30 days price and Google trend"):
        ticker = 'BTC-USD' # which stock to search for
        kw_list = 'Bitcoin'
        t_now = datetime.now()
        t_prev= datetime.fromtimestamp(t_now.timestamp()-(3600*24*30))
        trends = dailydata.get_daily_data(kw_list,t_prev.year,t_prev.month,t_now.year,t_now.month, geo='')
        fin_data = yf.download(ticker,start=t_prev.strftime('%Y-%m-%d'),end=t_now.strftime('%Y-%m-%d'))

        fin_indx = 0 #Prices: 0=Open,1=High,2=Low,3=Close,4=Adj Close,5=Volume
        trend_indx = 3 #Popularity: 0=unscaled 0-100, 1=monthly, 2=isPartial, 3=scaled 

        plt.style.use('ggplot')
        fig,ax = plt.subplots(figsize=(14,9))

        fin_keys = fin_data.keys() # keys for naming plotted finance data
        trend_keys = trends.keys() # keys for naming plotted trends data

        fin_x = [ii.timestamp() for ii in fin_data.index] # formatting dates into timestamp for plotting
        fin_y = (fin_data.values)[:,fin_indx] # trend data to plot

        trend_x = [ii.timestamp() for ii in trends.index] # formatting dates into timestamp for plotting
        trend_y = (trends.values)[:,trend_indx] # trend data to plot

        trend_start_indx = np.argmin(np.abs(np.subtract(trend_x,fin_x[0])))
        trend_end_indx   = np.argmin(np.abs(np.subtract(trend_x,fin_x[-1])))
        trend_x = trend_x[trend_start_indx:trend_end_indx+1] # align trends + stock $
        trend_y = trend_y[trend_start_indx:trend_end_indx+1] # align trends + stock $

        scat1 = ax.scatter(trend_x,trend_y,color=plt.cm.tab20(0),s=120) # scatter trend data

        ax.set_ylabel('Trend: '+(trend_keys[trend_indx].replace('_',' ')),
                        fontsize=20,color=plt.cm.tab20(0))
        x_ticks = ax.get_xticks()
        x_str_labels = [(datetime.fromtimestamp(ii)).strftime('%m-%d-%Y') for ii in x_ticks]
        ax.set_xticklabels(x_str_labels) # format dates on x-axis
        ax.set_xlabel('Time [Month-Day-Year]',fontsize=20)

        ax2 = ax.twinx() # twin axis to plot both data on the same plot
        ax2.grid(False) # this prevents the axes from being too messy
        scat2 = ax2.scatter(fin_x,fin_y,color=plt.cm.tab20(2),s=120) # scatter finance data
        ax2.set_ylabel(fin_keys[fin_indx]+' Price [$ USD]',fontsize=20,color=plt.cm.tab20(2))

        plt.title(ticker,fontsize=20)
        plt.savefig('Images/btc_gtrend.png')
        from PIL import Image
        image_scaled = Image.open('Images/btc_gtrend.png')
        st.image(image_scaled)


    with st.expander(" 👁  Click to view Wordcloud of related Google Trend Search Key Words"):
        pytrend = TrendReq(hl='en-US', tz=-480)
        kw_list = ["bitcoin"]
        pytrend.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='US', gprop='')
        trends = pytrend.related_queries()
        df_sg = trends["bitcoin"]["top"]
        text = ' '.join(df_sg["query"].to_list())
        # Generate a word cloud image
        wordcloud = WordCloud().generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        plt.title(ticker,fontsize=20)
        plt.savefig('Images/btc_wordcloud.png')
        from PIL import Image
        image_wc = Image.open('Images/btc_wordcloud.png')
        st.image(image_wc)

    st.markdown("***")
    #Support and Resistance
    st.subheader('Support and Resistance Analysis')
    st.info("Support occurs where a downtrend is expected to pause due to a concentration of demand. Resistance occurs where an uptrend is expected to pause temporarily, due to a concentration of supply. ")   

    plt.rcParams['figure.figsize'] = [12, 7]

    plt.rc('font', size=14) 
    fin_data['Date'] = pd.to_datetime(fin_data.index)
    fin_data['Date'] = fin_data['Date'].apply(mpl_dates.date2num)

    fin_data = fin_data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

    def isSupport(fin_data,i):
        support = fin_data['Low'][i] < fin_data['Low'][i-1]  and fin_data['Low'][i] < fin_data['Low'][i+1] \
        and fin_data['Low'][i+1] < fin_data['Low'][i+2] and fin_data['Low'][i-1] < fin_data['Low'][i-2]

        return support

    def isResistance(fin_data,i):
        resistance = fin_data['High'][i] > fin_data['High'][i-1]  and fin_data['High'][i] > fin_data['High'][i+1] \
        and fin_data['High'][i+1] > fin_data['High'][i+2] and fin_data['High'][i-1] > fin_data['High'][i-2] 

        return resistance
    levels = []
    for i in range(2,fin_data.shape[0]-2):
        if isSupport(fin_data,i):
            levels.append((i,fin_data['Low'][i]))
        elif isResistance(fin_data,i):
            levels.append((i,fin_data['High'][i]))

    def plot_all():
        fig, ax = plt.subplots()

        candlestick_ohlc(ax,fin_data.values,width=0.6, \
                        colorup='green', colordown='red', alpha=0.8)

        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()

        for level in levels:
            plt.hlines(level[1],xmin=fin_data['Date'][level[0]],\
                    xmax=max(fin_data['Date']),colors='blue')
        plt.savefig('Images/support_resist_btc.png')
        from PIL import Image
        image_sr = Image.open('Images/support_resist_btc.png')
        st.image(image_sr)
    plot_all()

    #Fibonacci Retracements
    st.subheader('Fibonacci Retracements')
    st.info("Fibonacci retracement levels are horizontal lines that indicate where support and resistance are likely to occur. Each level is associated with a percentage. The percentage is how much of a prior move the price has retraced.")   
    highest_swing = -1
    lowest_swing = -1
    for i in range(1,fin_data.shape[0]-1):
        if fin_data['High'][i] > fin_data['High'][i-1] and fin_data['High'][i] > fin_data['High'][i+1] and (highest_swing == -1 or fin_data['High'][i] > fin_data['High'][highest_swing]):
            highest_swing = i

        if fin_data['Low'][i] < fin_data['Low'][i-1] and fin_data['Low'][i] < fin_data['Low'][i+1] and (lowest_swing == -1 or fin_data['Low'][i] < fin_data['Low'][lowest_swing]):
            lowest_swing = i
    ratios = [0,0.236, 0.382, 0.5 , 0.618, 0.786,1]
    colors = ["black","r","g","b","cyan","magenta","yellow"]
    levels = []

    max_level = fin_data['High'][highest_swing]
    min_level = fin_data['Low'][lowest_swing]

    for ratio in ratios:
        if highest_swing > lowest_swing: # Uptrend
            levels.append(max_level - (max_level-min_level)*ratio)
        else: # Downtrend
            levels.append(min_level + (max_level-min_level)*ratio)

    plt.rcParams['figure.figsize'] = [12, 7]

    plt.rc('font', size=14)

    plt.plot(fin_data['Close'])
    start_date = fin_data.index[min(highest_swing,lowest_swing)]
    end_date = fin_data.index[max(highest_swing,lowest_swing)]
    for i in range(len(levels)):
        plt.hlines(levels[i],start_date, end_date,label="{:.1f}%".format(ratios[i]*100),colors=colors[i], linestyles="dashed")
    plt.legend()

    plt.savefig('Images/fibonacci_btc.png')
    from PIL import Image
    image_fr = Image.open('Images/fibonacci_btc.png')
    st.image(image_fr)


def get_ethusd(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:ETHUSD",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )

def get_bnbusd(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:BNBUSD",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )

def get_solusd(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:SOLUSD",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )

def get_bnbbtc(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:BNBBTC",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )


def get_bnbeth(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:BNBETH",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )


def get_ethbtc(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:ETHBTC",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )


def get_soleth(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:SOLETH",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )

def get_solbtc(height, width):

    components.html(
        f"""
        <div class="tradingview-widget-container">
        <div id="tradingview_c2f24"></div>

        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">

        new TradingView.widget(
        {{
        "width": {width},
        "height": {height},
        "symbol": "BINANCE:SOLBTC",
        "interval": "D",
        "timezone": "Asia/Hong_Kong",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [
            "BB@tv-basicstudies",
            "MACD@tv-basicstudies",
            "RSI@tv-basicstudies",
            "WilliamR@tv-basicstudies"
        ],
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )

