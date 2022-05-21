
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

