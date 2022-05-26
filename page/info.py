import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import base64
import requests
from requests import Request, Session
import json
import cryptographs
import graph_color_decision
import streamlit.components.v1 as components

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

#https://research.binance.com/en/projects/bnb
#https://research.binance.com/en/projects/ethereum
#https://docs.solana.com/history
#https://research.binance.com/en/projects/solana
#Reference


headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': '6d6d5a32-8c60-4040-8f72-095ea1bb5b19',   
        }
timezonehk = "Asia/Hong_Kong"


def get_info():

    with st.form(key='my_form'):
        crypto = st.selectbox('Select Cryptocurrency',
                          ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Binance Coin (BNB)', 'Solana (SOL)'])
        submit_button = st.form_submit_button(label='Submit')
          
        
    if submit_button:
        if crypto == "Bitcoin (BTC)":          
            get_btc()   
        if crypto == "Ethereum (ETH)":
            get_eth()
        if crypto == "Binance Coin (BNB)":
            get_bnb()
        if crypto == "Solana (SOL)":
            get_sol()
            

def get_btc():
    parameters = {'slug': 'bitcoin', 'convert': 'USD'}
    session = Session()
    session.headers.update(headers)
    response = session.get(url, params=parameters)
    hour24 = json.loads(response.text)['data']['1']['quote']['USD']['percent_change_24h']
    price = json.loads(response.text)['data']['1']['quote']['USD']['price']
    marketcap = json.loads(response.text)['data']['1']['quote']['USD']['market_cap']
    marketcap_ch = json.loads(response.text)['data']['1']['quote']['USD']['market_cap_dominance']
    volume = json.loads(response.text)['data']['1']['quote']['USD']['volume_24h']
    volume_ch = json.loads(response.text)['data']['1']['quote']['USD']['volume_change_24h']
    week_per_ch = json.loads(response.text)['data']['1']['quote']['USD']['percent_change_7d']
    graph_color_decision.eth_color = week_per_ch
    col1, col2, col3,col4,g1 = st.columns([1.5,2,2,2,1.5])


    with col1:
        st.image("Images/bitcoin.png")
    with col2:
        st.metric(label='Bitcoin BTC ', value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
    with col3:
        st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
    with col4:
        st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))), delta=str(round(marketcap_ch, 2)) + "%")
    with g1:
        st.write("7 days graph")
        cryptographs.get_btc_graph()

    st.header("Bitcoin")
    
    data = yf.download("BTC-USD")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.index), y=list(data['Adj Close']), line=dict(color="#fb9e18")))
    fig.update_layout(
                      xaxis_title="Date",
                      yaxis_title="Price in USD", )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="HTD", step="hour", stepmode="todate"),
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("Introduction")
    st.write("According to Bitcoin.org (2022), Bitcoin is a decentralized digital currency created by Satoshi Nakamoto in 2009, that uses peer-to-peer (P2P) technology to enable payments to anyone at a very low costs, and the transfer process would be completely operated without central authority, thus money issuance and relevant transactions could be carried out by the network, with blockchain technology – a shared public ledger to confirm all transactions among users. As of May 2022, Bitcoin has still dominated with the highest market cap at over USD$560 billion, accounted more than double compared with Ethereum.")
    st.header("Mechanism of Bitcoin")
    st.write("The Bitcoin protocol uses an SHA-256d-based Proof-of-Work (PoW) algorithm to reach network consensus. Its network has a target block time of 10 minutes and a maximum supply of 21 million tokens, with a decaying token emission rate. To prevent fluctuation of the block time, the network's block difficulty is re-adjusted through an algorithm based on the past 2016 block times.")
    st.write("With a block size limit capped at 1 megabyte, the Bitcoin Protocol has supported both the Lightning Network, a second-layer infrastructure for payment channels, and Segregated Witness, a soft-fork to increase the number of transactions on a block, as solutions to network scalability.")

    st.header("Bitcoin White Paper")

    with open("white_paper/bitcoin.pdf", "rb") as file:
        btn=st.download_button(
        label="Download Bitcoin White Paper",
        data=file,
        file_name="bitcoin.pdf",
        mime="application/octet-stream")


    def st_display_pdf(pdf_file):
        with open(pdf_file,"rb") as f:
            base64_pdf=base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display,unsafe_allow_html=True)


    #with st.expander(" 👁  Bitcoin white paper "):
    st_display_pdf("white_paper/bitcoin.pdf")
    st.write("")
    st.header("Github Open Source Code")
    st.write('[Bitcoin Open Source Code Link](https://github.com/bitcoin)')
    st.write("")
    st.write("References: Yahoo Finance, CoinMarketCap, Binance Research")

def get_eth():
    parameters = {'slug': 'ethereum', 'convert': 'USD'}
    session = Session()
    session.headers.update(headers)
    response = session.get(url, params=parameters)
    hour24 = json.loads(response.text)['data']['1027']['quote']['USD']['percent_change_24h']
    price = json.loads(response.text)['data']['1027']['quote']['USD']['price']
    marketcap = json.loads(response.text)['data']['1027']['quote']['USD']['market_cap']
    marketcap_ch = json.loads(response.text)['data']['1027']['quote']['USD']['market_cap_dominance']
    volume = json.loads(response.text)['data']['1027']['quote']['USD']['volume_24h']
    volume_ch = json.loads(response.text)['data']['1027']['quote']['USD']['volume_change_24h']
    week_per_ch = json.loads(response.text)['data']['1027']['quote']['USD']['percent_change_7d']
    graph_color_decision.eth_color = week_per_ch
    col5, col6, col7,col8,g2 = st.columns([1.5,2,2,2,1.5])

    with col5:
        st.image("Images/ethereum.png")
    with col6:
        st.metric(label="Ethereum (ETH)", value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
    with col7:
        st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
    with col8:
        st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))),delta=str(round(marketcap_ch, 2)) + "%")
    with g2:
        st.write("7 days graph")
        cryptographs.get_eth_graph()

    st.header("Ethereum")
    

    data = yf.download("ETH-USD")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.index), y=list(data['Adj Close']), line=dict(color="#fb9e18")))
    fig.update_layout(
                      xaxis_title="Date",
                      yaxis_title="Price in USD", )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="HTD", step="hour", stepmode="todate"),
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("Introduction")
    st.write("As the second largest cryptocurrency that could apply in multiple functions to perform payments as international transfer service, however it carried the largest network consisting of smart contracts, which has made it one of the most popular currencies especially in terms of functionality. As of May 2022, Ethereum contains the second highest market cap at over USD$238 trillion, which is the closest rival to Bitcoin.")

    st.header("Mechanism of Ethereum")
    st.write("The Ethereum platform supports ether in addition to a network of decentralized apps, otherwise known as "
             "dApps. Smart contracts, which originated on the Ethereum platform, are a central component of how the "
             "platform operates. Many decentralized finance (DeFi) and other applications use smart contracts in "
             "conjunction with blockchain technology.")

    st.write("The Ethereum network derives its security from the decentralized nature of blockchain technology. A vast network of computers "
             "worldwide maintains the Ethereum blockchain network, and the network requires distributed consensus—majority"
             " agreement—for any changes to be made to the blockchain. An individual or group of network participants "
             "would need to gain majority control of the Ethereum platform’s computing power—a task that would be "
             "gargantuan, if not impossible—to successfully manipulate the Ethereum blockchain.")

    st.write("The Ethereum platform can support many more applications than ETH and other cryptocurrencies. The network’s "
             "users can create, publish, monetize, and use a diverse range of applications on the Ethereum platform, and "
             "can use ETH or another cryptocurrency as payment.")

    st.write("Ethereum’s transition to the proof of stake protocol, which enables users to validate transactions and mint"
             " new ETH based on their ether holdings, is part of a major upgrade to the Ethereum platform known as Eth2."
             " The upgrade also adds capacity to the Ethereum network to support its growth, which helps to address chronic"
             " network congestion problems that have driven up gas fees.")

             
    st.header("Ethereum White Paper")

    with open("white_paper/ethereum.pdf", "rb") as file:
        btn = st.download_button(
            label="Download Ethereum White Paper",
            data=file,
            file_name="ethereum.pdf",
            mime="application/octet-stream")

    def st_display_pdf(pdf_file):
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    
    st_display_pdf("white_paper/ethereum.pdf")
    st.write("")
    st.header("Github Open Source Code")
    st.write('[Ethereum Open Source Code Link](https://github.com/ethereum)')
    st.write("")
    st.write("References: Yahoo Finance, CoinMarketCap, Binance Research")




def get_bnb():
    parameters = {'slug': 'binance-coin', 'convert': 'USD'}
    session = Session()
    session.headers.update(headers)
    response = session.get(url, params=parameters)
    hour24 = json.loads(response.text)['data']['1839']['quote']['USD']['percent_change_24h']
    price = json.loads(response.text)['data']['1839']['quote']['USD']['price']
    marketcap = json.loads(response.text)['data']['1839']['quote']['USD']['market_cap']
    marketcap_ch = json.loads(response.text)['data']['1839']['quote']['USD']['market_cap_dominance']
    volume = json.loads(response.text)['data']['1839']['quote']['USD']['volume_24h']
    volume_ch = json.loads(response.text)['data']['1839']['quote']['USD']['volume_change_24h']
    week_per_ch = json.loads(response.text)['data']['1839']['quote']['USD']['percent_change_7d']
    graph_color_decision.bnb_color = week_per_ch
    col9, col10, col11,col12,g3 = st.columns([1.5,2,2,2,1.5])

    with col9:
        st.image("Images/binance.png")
    with col10:
        st.metric(label="Binance (BNB)", value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
    with col11:
        st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
    with col12:
        st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))),delta=str(round(marketcap_ch, 2)) + "%")
    with g3:
        st.write("7 days graph")
        cryptographs.get_bnb_graph()

    st.header("Binance Coin")

    data = yf.download("BNB-USD")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.index), y=list(data['Adj Close']), line=dict(color="#fb9e18")))
    fig.update_layout(
                      xaxis_title="Date",
                      yaxis_title="Price in USD", )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="HTD", step="hour", stepmode="todate"),
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)        

    st.header("Introduction")
    st.write("As the native cryptocurrency of Binance Exchange, investors can use Binance Coin to make online and in-store purchase, as well as create smart contracts, travel bookingand offer lending, in general to utilize use cases for performing multiple functions (Moore, 2022). Created by Zhao since 2017, Binance has also proven it to be one of the best performed cryptocurrencies in recent years, as of May 2022, Binance Coin has the fifth largest market cap at over USD$52 trillion.")

    st.header("Mechanism of Binance Coin")
    st.write("Binance Coin was initially based on Ethereum blockchain network, but now the native currency has already transited to Binance’s own blockchain network – called the Binance Chain. In 2021, Binance Coin has launched an automatic burn feature that could offer a longer standing deflationary mechanism")

 
             
    st.header("Binance Coin White Paper")

    with open("white_paper/binance.pdf", "rb") as file:
        btn = st.download_button(
            label="Download Binance Coin White Paper",
            data=file,
            file_name="binance.pdf",
            mime="application/octet-stream")

    def st_display_pdf(pdf_file):
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    
    st_display_pdf("white_paper/binance.pdf")
    st.write("References: Yahoo Finance, CoinMarketCap, Binance Research")
    st.write("")
    st.header("Github Open Source Code")
    st.write('[Binance Coin Open Source Code Link](https://github.com/bnb-chain)')
    st.write("")
    st.write("References: Yahoo Finance, CoinMarketCap, Binance Research")



        
def get_sol():
    parameters = {'slug': 'solana', 'convert': 'USD'}
    session = Session()
    session.headers.update(headers)
    response = session.get(url, params=parameters)
    hour24 = json.loads(response.text)['data']['5426']['quote']['USD']['percent_change_24h']
    price = json.loads(response.text)['data']['5426']['quote']['USD']['price']
    marketcap = json.loads(response.text)['data']['5426']['quote']['USD']['market_cap']
    marketcap_ch = json.loads(response.text)['data']['5426']['quote']['USD']['market_cap_dominance']
    volume = json.loads(response.text)['data']['5426']['quote']['USD']['volume_24h']
    volume_ch = json.loads(response.text)['data']['5426']['quote']['USD']['volume_change_24h']
    week_per_ch = json.loads(response.text)['data']['5426']['quote']['USD']['percent_change_7d']
    graph_color_decision.sol_color = week_per_ch
    col13, col14, col15,col16,g4 = st.columns([1.5,2,2,2,1.5])

    with col13:
        st.image("Images/solana.png")
    with col14:
        st.metric(label="Solana (SOL)", value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
    with col15:
        st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
    with col16:
        st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))),delta=str(round(marketcap_ch, 2)) + "%")
    with g4:
        st.write("7 days graph")
        cryptographs.get_sol_graph()

    st.header("Solana")

    data = yf.download("SOL-USD")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(data.index), y=list(data['Adj Close']), line=dict(color="#fb9e18")))
    fig.update_layout(
                      xaxis_title="Date",
                      yaxis_title="Price in USD", )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="HTD", step="hour", stepmode="todate"),
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


    st.header("Introduction")
    st.write("As another popular open-source blockchains, Solana also has smart contract functionality in hosting decentralized applications. It was developed on an application called Rust that builds a wide variety of applications; thus, the system of Solana is easier and more accessible for deploying apps and further development among developers, which encourage an exponential growth of the ecosystem (Karuki, 2022).")

    st.header("Mechanism of Solana")
    st.write("Compared to Bitcoin and Ethereum, Solana is a system replying on Proof of History (POH), that involves a certain procedure of sequential computational steps, in which could make decision between time gap between events by giving it a time stamp, especially with its advantage of high-speed capabilities to offer to its investors as it can process 65,000 transactions per second, block time is just one second.")

       
    st.header("Solana White Paper")

    with open("white_paper/solana.pdf", "rb") as file:
        btn = st.download_button(
            label="Download Solana White Paper",
            data=file,
            file_name="white_paper/solana.pdf",
            mime="application/octet-stream")

    def st_display_pdf(pdf_file):
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    
    st_display_pdf("white_paper/solana.pdf")

    st.write("")
    st.header("Github Open Source Code")
    st.write('[Solana Open Source Code Link](https://github.com/solana-labs/)')
    st.write("")
    st.write("References: Yahoo Finance, CoinMarketCap, Binance Research")

#As another popular open-source blockchains, Solana also has smart contract functionality in hosting decentralized applications. It was developed on an application called Rust that builds a wide variety of applications; thus, the system of Solana is easier and more accessible for deploying apps and further development among developers, which encourage an exponential growth of the ecosystem (Karuki, 2022). As of May 2022,

#Compared to Bitcoin and Ethereum, Solana is a system replying on Proof of History (POH), that involves a certain procedure of sequential computational steps, in which could make decision between time gap between events by giving it a time stamp, especially with its advantage of high-speed capabilities to offer to its investors as it can process 65,000 transactions per second, block time is just one second.
