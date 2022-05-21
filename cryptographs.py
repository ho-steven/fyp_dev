import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import graph_color_decision



plt.style.use('ggplot')

def get_btc_graph():

    data = yf.download(tickers='BTC-USD', period = '1wk', interval = '15m')
    plt.figure().set_figheight(2)
    if graph_color_decision.bit_color > 0.0:
        plt.plot(data.index,data['Close'],color="green",linewidth=3.0)
    else:
        plt.plot(data.index, data['Close'], color="red",linewidth=3.0)
    plt.axis('off')
    st.pyplot(plt,use_container_width=True)
    plt.cla()

def get_eth_graph():
    data = yf.download(tickers='ETH-USD', period='1wk', interval='15m')
    plt.figure().set_figheight(2)
    if graph_color_decision.eth_color > 0.0:
        plt.plot(data.index,data['Close'],color="green",linewidth=3.0)
    else:
        plt.plot(data.index, data['Close'], color="red",linewidth=3.0)
    plt.axis('off')
    st.pyplot(plt,use_container_width=True)
    plt.cla()

def get_bnb_graph():
    data = yf.download(tickers='BNB-USD', period='1wk', interval='15m')
    plt.figure().set_figheight(2)
    if graph_color_decision.bnb_color > 0.0:
        plt.plot(data.index,data['Close'],color="green",linewidth=3.0)
    else:
        plt.plot(data.index, data['Close'], color="red",linewidth=3.0)
    plt.axis('off')
    st.pyplot(plt)
    plt.cla()

def get_sol_graph():
    data = yf.download(tickers='SOL-USD', period='1wk', interval='15m')
    plt.figure().set_figheight(2)
    if graph_color_decision.sol_color > 0.0:
        plt.plot(data.index,data['Close'],color="green",linewidth=3.0)
    else:
        plt.plot(data.index, data['Close'], color="red",linewidth=3.0)
    plt.axis('off')
    st.pyplot(plt)
    plt.cla()

