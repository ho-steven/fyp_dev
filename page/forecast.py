
import streamlit as st
import requests
from streamlit_lottie import st_lottie_spinner

from crypto import ada_trading, bit_trading, eth_trading, xlm_trading, ltc_trading, usdt_trading, dot_trading, bch_trading, doge_trading, shib_trading


def get_forecast():

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_url = "https://assets3.lottiefiles.com/private_files/lf30_h4qnjuax.json"
    lottie_json = load_lottieurl(lottie_url)

    #st.warning("NOTE :-  The Predicted values of cryptocurrencies are forecasted by machine learning algorithm and are for your "
    #           "reference only, it doesn't guarantee future exact values."
    #           "Please do a research before taking any further decision based on this forecasted values.")
    st.warning("Under Construction")

    with st.form(key='my_form'):
            crypto = st.selectbox('Select Cryptocurrency',
                              ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Binance Coin (BNB)', 'Solana (SOL)'])
            submit_button = st.form_submit_button(label='Submit')



    if submit_button:

        if crypto == "Bitcoin (BTC)":
                st.info("Info : If yellow line uppercrosses the line blue line means it's time to BUY and if yellow line lowercrosses the blue line means its time to SELL")
                with st_lottie_spinner(lottie_json):
                    bit_trading.get_bit_trading()

        if crypto == "Ethereum (ETH)":
                st.info("Info : If yellow line uppercrosses the line blue line means it's time to BUY and if yellow line lowercrosses the blue line means its time to SELL")
                with st_lottie_spinner(lottie_json):
                    eth_trading.get_eth_trading()


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