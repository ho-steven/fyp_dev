import streamlit as st
from requests import Request, Session
import json


import cryptographs



def get_home():
        # with open('homeview') as f:
                # st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        
        
        st.subheader("Introduction")
        st.write("To develop an AI investment advisor in Cryptocurrency that contains a prediction model to forecast multiple cryptocurrency prices including Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB) and Solana (SOL) with upward/downward trend and pair coin trading, additionally to develop a centric-design portfolio management tool on a web-based platform, contains a variety of customized data analytic dashboard and charting tools to provide a comprehensive trading overview and prediction with frequently-updated processed data. The analyser could fulfil an increasing demand from investors who look for advanced tools to overview their trading strategy and portfolio of cryptocurrencies, by bringing them a hybrid financial advisor targeting various levels of investors with a nimble and informative platform for accessing latest information and market trend, price prediction with personalized needs on portfolio advice.")
        st.markdown("***")
        
        st.subheader("Project Objectives")
        st.write("Cryptocurrency has been a popular topic in the financial industry recent years with its rapid growth of penetration into the societies and influencing investment mindset of the young generations and the entire ecosystem. Although a dramatic expansion of the emerging crypto industry has drawn public attention and impacts to the traditional financial industry, there is still a lot of room to improve the existing financial analyser platforms with sufficient research and technical investigation to the cryptocurrency market. Such a speedy pace of industry nature has also made it more challenging to build an enhanced prediction model to forecast the market trend and provide investment advice by building a personalized portfolio of cryptocurrencies especially in coin swapping with multiple factor analysis than traditional stocks.")

        col10, col11, col12 = st.columns([3, 5, 3])

        with col10:
            st.write(' ')

        with col11:
            from PIL import Image
            image4 = Image.open("Images/problem_statement.jpg")
            st.image(image4)

        with col12:
            st.write(' ')


        st.write("Meanwhile, the experiment design could solve some practical problems by improving current machine learning techniques, thus making a contribution to the financial industry by solving problems for investors in the real world. Three main machine learning model techniques – Random Forest, Long Short-term Memory (LSTM), and News Sentiment Analysis would be explored and applied as key components throughout the experiment design in a comprehensive approach, combining trained algorithms and artificial intelligence techniques to forecast cryptocurrency prices based on prices, volume, market capitalization, volatility, additionally to provide prediction with insightful predictions and non-technical factors considered from news sentiment analysis.")
        st.markdown("**Advanced portfolio management and investment advice in coin swapping**")
        st.write("For developing a financial screener with allocate cryptocurrencies (Bitcoin, Ethereum, Solana and Binance Coin) individually and coin-pairing to maximize the return of the portfolio within the risk threshold of the user by trending analysis, trading and swapping coin signals, order to provide an investor centric-design web application that allows users to specify their risk threshold interact with the system and see the performance of their portfolio.")
        st.write("Key mechanism of portfolio management aims to maximize profit and minimize risk by coin trading among the four selected cryptocurrencies instead of buying or selling with stable coins. Thus, to bring academic contribution to the research area in swapping cryptocurrencies using Machine Learning techniques on top of prediction on traditional trading techniques.")
        st.markdown("**Hybrid financial analyser with combination of financial and non-financial factors**")
        st.write("Another essential contribution from this project to the research area of price predictions is regarding the enhanced system flow design by combining machine learning models with both financial and non-financial factors. The cause was due to previous research results are lacking stock and cryptocurrencies price prediction by considering multiple factors in timely manner. Further mechanism details and future work would be explained in later sessions.")
        st.markdown("***")
        
        st.subheader("System Flow Diagram")
        col1, col2, col3 = st.columns([3, 5, 3])

        with col1:
            st.write(' ')

        with col2:
            
            image = Image.open("Images/intro.jpg")
            st.image(image)

        with col3:
            st.write(' ')
        st.markdown("***")

        
        st.subheader("Implementation Schedule and Milestones")       

        col4, col5, col6 = st.columns([3, 5, 3])

        with col4:
            st.write(' ')

        with col5:
            image1 = Image.open("Images/milestones.jpg")
            st.image(image1)

        with col6:
            st.write(' ')
            
        st.markdown("***")
        
        
        
        st.subheader("Team")
        col7, col8, col9 = st.columns([3, 5, 3])

        with col7:
            st.write(' ')

        with col8:
            image2 = Image.open("Images/member.jpg")
            st.image(image2)

        with col9:
            st.write(' ')
            

        st.markdown("***")
        
        st.subheader("Disclaimer") 
        st.write("This website and the information provided on this website has been prepared solely for informational and educational purposes and should not be construed as an offer to buy or sell or a solicitation of an offer to buy or sell any crypto asssets or to participate in any transaction or trading activities. Before making any investment decisions, you should consider your own financial situation, investment objectives and experiences, risk acceptance and ability to understand the nature and risks of the relevant product. The website owner shall not be liable to any loss or damage incurred by any person caused by direct or indirect usage of the information or its content stated herein")     

