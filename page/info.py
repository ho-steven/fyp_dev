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
            
        
    def get_crypto_price(cryptocurrency, currency):
        return cryptocompare.get_price(cryptocurrency, currency)[cryptocurrency][currency]
        
        

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

    st.header("Background")
    st.write("Bitcoin is a digital currency which operates free of any central control or the oversight of banks "
             "or governments. Instead, it relies on peer-to-peer software and cryptography. Bitcoins can currently"
             " be subdivided by seven decimal places: a thousandth of a bitcoin is known as a milli and a hundred "
             "millionth of a bitcoin is known as a satoshi. In truth there is no such thing as a bitcoin or a wallet, "
             "just agreement among the network about ownership of a coin. A private key is used to prove ownership of "
             "funds to the network when making a transaction. A person could simply memorise their private key and need"
             " nothing else to retrieve or spend their virtual cash, a concept which is known as a “brain wallet”.")
    st.header("History")
    st.write("The word bitcoin was defined in a white paper published on 31 October 2008. On 31 October 2008, "
             "a link to a paper authored by Satoshi Nakamoto titled Bitcoin: A Peer-to-Peer Electronic Cash System"
             " was posted to a cryptography mailing list. Nakamoto implemented the bitcoin software as open-source"
             " code and released it in January 2009.  Nakamoto's identity remains unknown. On 3 January 2009, the"
             " bitcoin network was created when Nakamoto mined the starting block of the chain, known as the "
             "genesis block")
    st.header("Block Chain")
    st.write("Bitcoin is a network that runs on a protocol known as the blockchain. While it does not mention "
             "the word blockchain, a 2008 paper by a person or people calling themselves Satoshi Nakamoto first "
             "described the use of a chain of blocks to verify transactions and engender trust in a network.The "
             "bitcoin blockchain is a public ledger that records bitcoin transactions. It is implemented as a "
             "chain of blocks, each block containing a hash of the previous block up to the genesis blockin the "
             "chain. A network of communicating nodes running bitcoin software maintains the blockchain. "
             "Transactions of the form payer X sends Y bitcoins to payee Z are broadcast to this network using "
             "readily available software applications. In the blockchain, bitcoins are registered to bitcoin "
             "addresses. Creating a bitcoin address requires nothing more than picking a random valid private "
             "key and computing the corresponding bitcoin address. Blockchain's versatility has caught the eye "
             "of governments and private corporations; indeed, some analysts believe that blockchain technology "
             "will ultimately be the most impactful aspect of the cryptocurrency craze.")
    st.header("Software Implementation")
    st.write("Bitcoin Core is free and open-source software that serves as a bitcoin node (the set of which form"
             " the bitcoin network) and provides a bitcoin wallet which fully verifies payments. It is considered"
             " to be bitcoin's reference implementation. Initially, the software was published by Satoshi Nakamoto"
             " under the name Bitcoin, and later renamed to Bitcoin Core to distinguish it from the network."
             " It is also known as the Satoshi client. Bitcoin Core includes a transaction verification engine and "
             "connects to the bitcoin network as a full node. Moreover, a cryptocurrency wallet, which can be used to "
             "transfer funds, is included by default. The wallet allows for the sending and receiving of bitcoins. "
             "It does not facilitate the buying or selling of bitcoin. It allows users to generate QR codes to receive "
             "payment. The software validates the entire blockchain, which includes all bitcoin transactions ever. "
             "This distributed ledger which has reached more than 235 gigabytes in size as of Jan 2019, must be"
             " downloaded or synchronized before full participation of the client may occur. Although the complete "
             "blockchain is not needed all at once since it is possible to run in pruning mode.")


    st.header("Bitcoin White Paper")

    with open("white_paper/bitcoin.pdf", "rb") as file:
        btn=st.download_button(
        label="Click me! to download white paper",
        data=file,
        file_name="bitcoin.pdf",
        mime="application/octet-stream")


    def st_display_pdf(pdf_file):
        with open(pdf_file,"rb") as f:
            base64_pdf=base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display,unsafe_allow_html=True)


    with st.expander(" 👁  Bitcoin white paper "):
        st_display_pdf("white_paper/bitcoin.pdf")







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

        # url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        # headers = {
                # 'Accepts': 'application/json',
                # 'X-CMC_PRO_API_KEY': 'f196187f-d576-4c6c-8d4c-d35a4bab8511',
        #6d6d5a32-8c60-4040-8f72-095ea1bb5b19
        # }
        # def get_btc():
                # parameters = {'slug': 'bitcoin', 'convert': 'CAD'}
                # session = Session()
                # session.headers.update(headers)
                # response = session.get(url, params=parameters)
                # hour24 = json.loads(response.text)['data']['1']['quote']['CAD']['percent_change_24h']
                # price = json.loads(response.text)['data']['1']['quote']['CAD']['price']
                # marketcap = json.loads(response.text)['data']['1']['quote']['CAD']['market_cap']
                # marketcap_ch = json.loads(response.text)['data']['1']['quote']['CAD']['market_cap_dominance']
                # volume = json.loads(response.text)['data']['1']['quote']['CAD']['volume_24h']
                # volume_ch = json.loads(response.text)['data']['1']['quote']['CAD']['volume_change_24h']
                # week_per_ch = json.loads(response.text)['data']['1']['quote']['CAD']['percent_change_7d']
                # graph_color_decision.bit_color = week_per_ch
                # cl1,col1, col2, col3,col4,g1 = st.columns([0.5,1.5,2,2,2,1.5])

                # with cl1:
                        # st.write(1)
                # with col1:
                        # st.image("Images/bitcoin.png")
                # with col2:
                        # st.metric(label='Bitcoin BTC ', value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
                # with col3:
                        # st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
                # with col4:
                        # st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))), delta=str(round(marketcap_ch, 2)) + "%")
                # with g1:
                        # st.write("7 days graph")
                        # cryptographs.get_btc_graph()

        # def get_eth():
                # parameters = {'slug': 'ethereum', 'convert': 'CAD'}
                # session = Session()
                # session.headers.update(headers)
                # response = session.get(url, params=parameters)
                # hour24 = json.loads(response.text)['data']['1027']['quote']['CAD']['percent_change_24h']
                # price = json.loads(response.text)['data']['1027']['quote']['CAD']['price']
                # marketcap = json.loads(response.text)['data']['1027']['quote']['CAD']['market_cap']
                # marketcap_ch = json.loads(response.text)['data']['1027']['quote']['CAD']['market_cap_dominance']
                # volume = json.loads(response.text)['data']['1027']['quote']['CAD']['volume_24h']
                # volume_ch = json.loads(response.text)['data']['1027']['quote']['CAD']['volume_change_24h']
                # week_per_ch = json.loads(response.text)['data']['1027']['quote']['CAD']['percent_change_7d']
                # graph_color_decision.eth_color = week_per_ch
                # cl2,col5, col6, col7,col8,g2 = st.columns([0.5,1.5,2,2,2,1.5])

                # with cl2:
                        # st.write(2)
                # with col5:
                        # st.image("Images/ethereum.png")
                # with col6:
                        # st.metric(label="Ethereum (ETH)", value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
                # with col7:
                        # st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
                # with col8:
                        # st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))),delta=str(round(marketcap_ch, 2)) + "%")
                # with g2:
                        # st.write("7 days graph")
                        # cryptographs.get_eth_graph()

        # def get_xlm():
                # parameters = {'slug': 'stellar', 'convert': 'CAD'}
                # session = Session()
                # session.headers.update(headers)
                # response = session.get(url, params=parameters)
                # hour24 = json.loads(response.text)['data']['512']['quote']['CAD']['percent_change_24h']
                # price = json.loads(response.text)['data']['512']['quote']['CAD']['price']
                # marketcap = json.loads(response.text)['data']['512']['quote']['CAD']['market_cap']
                # marketcap_ch = json.loads(response.text)['data']['512']['quote']['CAD']['market_cap_dominance']
                # volume = json.loads(response.text)['data']['512']['quote']['CAD']['volume_24h']
                # volume_ch = json.loads(response.text)['data']['512']['quote']['CAD']['volume_change_24h']
                # week_per_ch = json.loads(response.text)['data']['512']['quote']['CAD']['percent_change_7d']
                # graph_color_decision.xlm_color = week_per_ch
                # cl3,col9, col10, col11,col12,g3 = st.columns([0.5,1.5,2,2,2,1.5])

                # with cl3:
                        # st.write(3)
                # with col9:
                        # st.image("Images/xlmcoin.png")
                # with col10:
                        # st.metric(label="Stellar (XLM)", value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
                # with col11:
                        # st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
                # with col12:
                        # st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))),delta=str(round(marketcap_ch, 2)) + "%")
                # with g3:
                        # st.write("7 days graph")
                        # cryptographs.get_xlm_graph()

        # def get_usdt():
                # parameters = {'slug': 'tether', 'convert': 'CAD'}
                # session = Session()
                # session.headers.update(headers)
                # response = session.get(url, params=parameters)
                # hour24 = json.loads(response.text)['data']['825']['quote']['CAD']['percent_change_24h']
                # price = json.loads(response.text)['data']['825']['quote']['CAD']['price']
                # marketcap = json.loads(response.text)['data']['825']['quote']['CAD']['market_cap']
                # marketcap_ch = json.loads(response.text)['data']['825']['quote']['CAD']['market_cap_dominance']
                # volume = json.loads(response.text)['data']['825']['quote']['CAD']['volume_24h']
                # volume_ch = json.loads(response.text)['data']['825']['quote']['CAD']['volume_change_24h']
                # week_per_ch = json.loads(response.text)['data']['825']['quote']['CAD']['percent_change_7d']
                # graph_color_decision.usdt_color = week_per_ch
                # cl4,col13, col14, col15,col16,g4 = st.columns([0.5,1.5,2,2,2,1.5])

                # with cl4:
                        # st.write(4)
                # with col13:
                        # st.image("Images/usdtcoin.png")
                # with col14:
                        # st.metric(label="Tether (USDT)", value=("$"+str(round(price, 2))), delta=str(round(hour24, 2)) + "%")
                # with col15:
                        # st.metric(label="Volume (24h)", value=("$"+str(round(volume, 2))), delta=str(round(volume_ch, 2)) + "%")
                # with col16:
                        # st.metric(label="Market Cap", value=("$"+str(round(marketcap, 2))),delta=str(round(marketcap_ch, 2)) + "%")
                # with g4:
                        # st.write("7 days graph")
                        # cryptographs.get_usdt_graph()



# def live_price():

    # timezoneca = "Canada/Eastern"
    # Asia/Hong_Kong
    # def load_lottieurl(url: str):
        # r = requests.get(url)
        # if r.status_code != 200:
            # return None
        # return r.json()

    # lottie_url = "https://assets3.lottiefiles.com/private_files/lf30_h4qnjuax.json"
    # lottie_json = load_lottieurl(lottie_url)

    # def get_crypto_price(cryptocurrency, currency):
        # return cryptocompare.get_price(cryptocurrency, currency)[cryptocurrency][currency]

    # with st.form(key='my_form'):
        # crypto = st.selectbox('Select Cryptocurrency',
                              # ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Stellar (XLM)', 'Tether (USDT)', 'Bitcoin Cash (BCH)',
                               # 'Litecoin (LTC)', 'Polkadot (DOT)', 'Dogecoin (DOGE)', 'Cardano (ADA)', 'Shiba Inu (SHIB)'])
        # submit_button = st.form_submit_button(label='Submit')

    # if submit_button:
        # if crypto == "Bitcoin (BTC)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                # fig = go.FigureWidget()
                # fig.add_scatter(line=dict(color="#76D714"))
                # fig.update_layout(title="Bitcoin live price", xaxis_title="Time", yaxis_title= crypto +" Price in CAD ")
                # values = [get_crypto_price('BTC', 'CAD')]
                # times = []
            # while True:
                # with plot_spot:
                        # values.append(get_crypto_price('BTC', 'CAD'))
                        # times.append(datetime.now(timezone(timezoneca)))
                        # fig.data[0].x = times
                        # fig.data[0].y = values
                        # st.plotly_chart(fig, use_container_width=True)
                        # time.sleep(3)

        # if crypto == "Ethereum (ETH)":
                # plot_spot = st.empty()
                # with st_lottie_spinner(lottie_json):
                    # fig = go.FigureWidget()
                    # fig.add_scatter(line=dict(color="#76D714"))
                    # fig.update_layout(title="ETH" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                    # values = [get_crypto_price('ETH', 'CAD')]
                    # times = []
                # while True:
                    # with plot_spot:
                        # values.append(get_crypto_price('ETH', 'CAD'))
                        # times.append(datetime.now(timezone(timezoneca)))
                        # fig.data[0].x = times
                        # fig.data[0].y = values
                        # st.plotly_chart(fig, use_container_width=True)
                        # time.sleep(3)

        # if crypto == "Stellar (XLM)":
                # plot_spot = st.empty()
                # with st_lottie_spinner(lottie_json):
                    # fig = go.FigureWidget()
                    # fig.add_scatter(line=dict(color="#76D714"))
                    # fig.update_layout(title="XLM" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                    # values = [get_crypto_price('XLM', 'CAD')]
                    # times = []
                # while True:
                    # with plot_spot:
                        # values.append(get_crypto_price('XLM', 'CAD'))
                        # times.append(datetime.now(timezone(timezoneca)))
                        # fig.data[0].x = times
                        # fig.data[0].y = values
                        # st.plotly_chart(fig, use_container_width=True)
                        # time.sleep(3)

        # if crypto == "Tether (USDT)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                # fig = go.FigureWidget()
                # fig.add_scatter(line=dict(color="#76D714"))
                # fig.update_layout(title="USDT" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                # values = [get_crypto_price('USDT', 'CAD')]
                # times = []
            # while True:
                # with plot_spot:
                    # values.append(get_crypto_price('USDT', 'CAD'))
                    # times.append(datetime.now(timezone(timezoneca)))
                    # fig.data[0].x = times
                    # fig.data[0].y = values
                    # st.plotly_chart(fig, use_container_width=True)
                    # time.sleep(3)

        # if crypto == "Bitcoin Cash (BCH)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                # fig = go.FigureWidget()
                # fig.add_scatter(line=dict(color="#76D714"))
                # fig.update_layout(title="BCH" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                # values = [get_crypto_price('BCH', 'CAD')]
                # times = []
            # while True:
                # with plot_spot:
                    # values.append(get_crypto_price('BCH', 'CAD'))
                    # times.append(datetime.now(timezone(timezoneca)))
                    # fig.data[0].x = times
                    # fig.data[0].y = values
                    # st.plotly_chart(fig, use_container_width=True)
                    # time.sleep(3)

        # if crypto == "Litecoin (LTC)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                # fig = go.FigureWidget()
                # fig.add_scatter(line=dict(color="#76D714"))
                # fig.update_layout(title="LTC" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                # values = [get_crypto_price('LTC', 'CAD')]
                # times = []
            # while True:
                # with plot_spot:
                    # values.append(get_crypto_price('LTC', 'CAD'))
                    # times.append(datetime.now(timezone(timezoneca)))
                    # fig.data[0].x = times
                    # fig.data[0].y = values
                    # st.plotly_chart(fig, use_container_width=True)
                    # time.sleep(3)

        # if crypto == "Polkadot (DOT)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                # fig = go.FigureWidget()
                # fig.add_scatter(line=dict(color="#76D714"))
                # fig.update_layout(title="DOT" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                # values = [get_crypto_price('DOT', 'CAD')]
                # times = []
            # while True:
                # with plot_spot:
                    # values.append(get_crypto_price('DOT', 'CAD'))
                    # times.append(datetime.now(timezone(timezoneca)))
                    # fig.data[0].x = times
                    # fig.data[0].y = values
                    # st.plotly_chart(fig, use_container_width=True)
                    # time.sleep(3)

        # if crypto == "Dogecoin (DOGE)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                # fig = go.FigureWidget()
                # fig.add_scatter(line=dict(color="#76D714"))
                # fig.update_layout(title="DOGE" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                # values = [get_crypto_price('DOGE', 'CAD')]
                # times = []
            # while True:
                # with plot_spot:
                    # values.append(get_crypto_price('DOGE', 'CAD'))
                    # times.append(datetime.now(timezone(timezoneca)))
                    # fig.data[0].x = times
                    # fig.data[0].y = values
                    # st.plotly_chart(fig, use_container_width=True)
                    # time.sleep(3)

        # if crypto == "Cardano (ADA)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                # fig = go.FigureWidget()
                # fig.add_scatter(line=dict(color="#76D714"))
                # fig.update_layout(title="ADA" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                # values = [get_crypto_price('ADA', 'CAD')]
                # times = []
            # while True:
                # with plot_spot:
                    # values.append(get_crypto_price('ADA', 'CAD'))
                    # times.append(datetime.now(timezone(timezoneca)))
                    # fig.data[0].x = times
                    # fig.data[0].y = values
                    # st.plotly_chart(fig, use_container_width=True)
                    # time.sleep(3)

        # if crypto == "Shiba Inu (SHIB)":
            # plot_spot = st.empty()
            # with st_lottie_spinner(lottie_json):
                    # fig = go.FigureWidget()
                    # fig.add_scatter(line=dict(color="#76D714"))
                    # fig.update_layout(title="SHIB" + " live price", xaxis_title="Time", yaxis_title=crypto + " Price in CAD ")
                    # values = [get_crypto_price("SHIB", 'CAD')]
                    # times = []
            # while True:
                # with plot_spot:
                    # values.append(get_crypto_price("SHIB", 'CAD'))
                    # times.append(datetime.now(timezone(timezoneca)))
                    # fig.data[0].x = times
                    # fig.data[0].y = values
                    # st.plotly_chart(fig, use_container_width=True)
                    # time.sleep(3)
