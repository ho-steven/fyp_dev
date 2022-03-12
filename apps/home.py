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


def app():
    st.balloons()
    st.title('HKU MSc(CompSc) Project 2022')
    st.header('**AI Investment Advisor in Cryptocurrency**')
    st.markdown("***")
    
    st.subheader("Background")
    st.write("Our project aims to develop an AI investment advisor in Cryptocurrency that is supported by three AI models as fundamental analysis. The advisor will perform as a financial indicator to forecast the upward and downward trend of prices in multiple cryptocurrency as well as an advanced price prediction among the selected cryptocurrency by swapping/ trading in coin pairs. The model will be developed as a professional portfolio management tool on a web platform to bring out insightful data in visualized analysis for better UX/UI, thus to bring the audience a holistic view and easy-understanding forecast.")
    st.markdown("***")
    
    st.subheader("System Flow Diagram")
    from PIL import Image
    image = Image.open('intro.jpg')
    st.image(image)
    st.markdown("***")
    
    st.subheader("Project Timeline")
    image1 = Image.open('timeline.jpg')
    st.image(image1)
    st.markdown("***")
    
    st.subheader("Team")
    image2 = Image.open('team.jpg')
    st.image(image2)
    st.markdown("***")
    
    st.subheader("Disclaimer") 
    st.caption("This website and the information provided on this website has been prepared solely for informational and educational purposes and should not be construed as an offer to buy or sell or a solicitation of an offer to buy or sell any crypto asssets or to participate in any transaction or trading activities. Before making any investment decisions, you should consider your own financial situation, investment objectives and experiences, risk acceptance and ability to understand the nature and risks of the relevant product. The website owner shall not be liable to any loss or damage incurred by any person caused by direct or indirect usage of the information or its content stated herein")     
    




