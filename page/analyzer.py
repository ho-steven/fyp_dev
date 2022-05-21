import streamlit as st
import requests
from streamlit_lottie import st_lottie_spinner
import streamlit.components.v1 as components
from crypto import ada, bit, eth, xlm, ltc, usdt, dot, bch, doge, shib

def user_input_features(ticker, period):
    today = date.today()
    #ticker = st.sidebar.selectbox("Ticker", ('BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT'))
    #period = st.sidebar.selectbox("Period", ('1d','1h','1m'))
    if period == '1day': 
        start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
        end_date = st.sidebar.date_input("End Date", date.today())        
        return ticker, start_date, end_date
    elif period == '1h':
        no_of_hrs_analysis = st.sidebar.radio("Number of Hours for analysis", ["1000"])
        return ticker, no_of_hrs_analysis
    elif period == '1min':
        no_of_mins_analysis = st.sidebar.radio("Number of Minutes for analysis", ["1000"])
        return ticker, no_of_mins_analysis
        

def time_series_1d(ticker, period, start, end ):

    # Initialize client - apikey parameter is requiered
    td = TDClient(apikey="6b27384627a5423aa7b69032d14d9ca8")

    # Construct the necessary time series (interval:1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month)
    ts = td.time_series(
        symbol=ticker,
        interval=period,
        outputsize='1000',
        exchange='Binance', 
        start_date=start,
        end_date=end,
        timezone="America/New_York",
    )

    # Returns pandas.DataFrame
    time_s_1d = ts.as_pandas() 
    st.dataframe(time_s_1d)

#https://github.com/Japan745/CryptoBase
## Technical indicators to be used
# ADX, BBANDS , CCI, EMA, ICHIMOKU, MACD, MOM, PERCENT_B, RSI , SMA, STOCH, STOCHRSI, ULTOSC, VWAP, WILLR, WMA
# https://github.com/ho-steven/SP500-Analyzer/blob/main/apps/prediction.py


# Investing: RSI(14), STOCH(9,6), STOCHRSI(14), MACD(12,26), ADX(14), Williams %R, CCI(14), ATR(14), Highs/Lows(14), Ultimate Oscillator, ROC, Bull/Bear Power(13), (SMA & EMA) MA5, MA10, MA20, MA50, MA100, MA200
# https://www.investing.com/crypto/bitcoin/btc-usd-technical?cid=1035793
# https://investpy.readthedocs.io/_info/usage.html
# https://finviz.com/crypto_charts.ashx?t=BTCUSD



# Tradingview: RSI(14), Stochastic %K (14, 3, 3), CCI(20), ADX(14), Awesome Oscillator, Momentum (10), MACD Level (12, 26), Stochastic RSI Fast (3, 3, 14, 14), Williams Percent Range (14), Bull Bear Power, Ultimate Oscillator (7, 14, 28), Ichimoku Base Line (9, 26, 52, 26), Volume Weighted Moving Average (20), Hull Moving Average (9)
#https://tradingview.brianthe.dev/
#https://www.tradingview.com/symbols/BTCUSD/technicals/


def get_analyzer():

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # lottie_url = "https://assets7.lottiefiles.com/packages/lf20_p9ce3saq.json"
    # lottie_json = load_lottieurl(lottie_url)

    st.warning("Under Construction")
# https://assets10.lottiefiles.com/packages/lf20_8lcyef9f.json
# https://assets1.lottiefiles.com/packages/lf20_2xgpnwrq.json
# https://assets3.lottiefiles.com/packages/lf20_aj79ghex.json


# Forecast
# https://assets10.lottiefiles.com/packages/lf20_8lcyef9f.json

# analyser
# https://assets7.lottiefiles.com/packages/lf20_p9ce3saq.json

# market
# https://assets1.lottiefiles.com/packages/lf20_2xgpnwrq.json

# portfolio
# https://assets6.lottiefiles.com/private_files/lf30_gonpfxdh.json


    # st.warning(
        # "NOTE :-  The Predicted values of cryptocurrencies are forecasted by machine learning algorithm and are for your "
        # "reference only, it doesn't guarantee future exact values."
        # "Please do a research before taking any further decision based on this forecasted values.")

    #  Correlation analysis
    
    #  Price chart (1m, 1h, 1d)
    #  Basic infographics + intro
    #  Technical Analysis (ta-lib)
    
    # AI forecasting
    #  Trading views
    #  News Sentiment Analysis
    #  Social Media Analysis (lunarcrush)
       # Twitter Analysis
    #AI Models (logistic regression, LSTM, Random Forest)
    #BNB/ETH, BNB/USD, BNB/BTC, BTC/USD, ETH/BTC, ETH/USD, SOL/BNB, SOL/BTC, SOL/USD, SOL/ETH (Huobi)

    # st.sidebar.header('User Input Parameters')
    # ticker = st.sidebar.selectbox("Ticker", ('BTC/USD','ETH/USD','BNB/USD','SOL/USD','BNB/BTC','BNB/ETH','ETH/BTC', 'SOL/BNB', 'SOL/BTC'))
    # period = st.sidebar.selectbox("Period", ('1day','1h','1min'))
    # if period == "1day" and ticker!='SOL/ETH':
        # symbol, start, end = user_input_features(ticker, period)
        # start = pd.to_datetime(start)
        # end = pd.to_datetime(end)
        # time_series_1d(ticker, period, start, end)
        #binance = ccxt.binance()
        #btc_ticker = binance.fetchMarkOHLCV ("BTC/USDT", '1d')
        #st.dataframe(btc_ticker)
        #st.table(btc_ticker)
        #st.json(btc_ticker)

    # elif period == "1h":
        # symbol, no_of_hrs = user_input_features(ticker, period)
    # elif period == "1min":
        # symbol, no_of_mins = user_input_features(ticker, period)

    #ticker = st.sidebar.selectbox("Ticker", ('BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT'))
    #period = st.sidebar.selectbox("Period", ('1d','1h','1m'))

    with st.form(key='my_form'):
        ticker = st.selectbox('Select Ticker',
                              ['BTC/USD','ETH/USD','BNB/USD','SOL/USD','BNB/BTC','BNB/ETH','ETH/BTC', 'SOL/BNB', 'SOL/BTC'])
        period = st.selectbox("Period", ('1day','1h','1min'))
        submit_button = st.form_submit_button(label='Submit')


    height = 600
    width = st.slider("Set the width of the chart ", 200, 1800, 800, 100)

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
        "container_id": "tradingview_c2f24"
        }}
        );

        </script>
        </div>
        """,
        height=height,
        width=width,
    )



    #if submit_button:
        
        # if crypto == "Bitcoin (BTC)":
            # with st_lottie_spinner(lottie_json):
                # bit.get_bit()
        # if crypto == "Ethereum (ETH)":
            # with st_lottie_spinner(lottie_json):
                # eth.get_eth()
        # if crypto == "Stellar (XLM)":
            # with st_lottie_spinner(lottie_json):
                # xlm.get_xlm()
        # if crypto == "Tether (USDT)":
            # with st_lottie_spinner(lottie_json):
                # usdt.get_usdt()
        # if crypto == "Bitcoin Cash (BCH)":
            # with st_lottie_spinner(lottie_json):
                # bch.get_bch()
        # if crypto == "Litecoin (LTC)":
            # with st_lottie_spinner(lottie_json):
                # ltc.get_ltc()
        # if crypto == "Polkadot (DOT)":
            # with st_lottie_spinner(lottie_json):
                # dot.get_dot()
        # if crypto == "Dogecoin (DOGE)":
            # with st_lottie_spinner(lottie_json):
                # doge.get_doge()
        # if crypto == "Cardano (ADA)":
            # with st_lottie_spinner(lottie_json):
                # ada.get_ada()
        # if crypto == "Shiba Inu (SHIB)":
            # with st_lottie_spinner(lottie_json):
                # shib.get_shib()
        # To-do list: 
        #Market: Coorelation matrix
        #Analyzer: 
        #1.Price and icon
        #2.Chart and technical analysis ((ta-lib))
        #3.news
        #4.social media
        #5.comparison with benchmark (other 3 cryptocurrency/ ncid index) (if have time)
        
        
        #Forecasting and Recommendations
        #  Trading views (& summarized recommendation)
        #  News Sentiment Analysis (& summarized recommendation)
        #  Social Media Analysis (lunarcrush)
        #  Twitter Analysis
        #  AI Models (logistic regression, LSTM, Random Forest, Reinforcement Learning)