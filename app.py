from page import home, market, info, analyzer, forecast, portfolio, team

import streamlit as st
from streamlit_option_menu import option_menu
from  PIL import Image
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI Investment Advisor in Cryptocurrency",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",)


st.subheader('HKU MSc(CompSc) Project 2022')
st.markdown("<center><h1 Style=\"overflow: visible; padding-bottom: 50px; padding-top: 0px;\">AI Investment Advisor in Cryptocurrency</h1></center>", unsafe_allow_html=True)
components.html(
    """
    <script type="text/javascript" src="https://files.coinmarketcap.com/static/widget/coinMarquee.js"></script><div id="coinmarketcap-widget-marquee" coins="1,1027,5426,1839" currency="USD" theme="light" transparent="false" show-symbol-logo="true"></div> 
    """,height= 70
)

# Get icons from https://icons.getbootstrap.com/
#2ECC71
selected = option_menu(
        menu_title=None,
        options=["Home", "Market", "Info", "Analyzer", "Forecast","Portfolio","Team"],
        icons=['bank', 'cash-coin', 'info-square', 'magic','activity','clipboard-data','person-bounding-box'],
        default_index=0,
        orientation = "horizontal",
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "#F88017 ", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "light-grey"},
        "nav-link-selected": {"background-color": "#F88017 "},
}
)

if selected == "Home":
    home.get_home()
elif selected == "Market":
    market.get_market()
elif selected == "Info":
    info.get_info()
elif selected == "Analyzer":
    analyzer.get_analyzer()
elif selected == "Forecast":
    forecast.get_forecast()
elif selected == "Portfolio":
    portfolio.get_portfolio()
elif selected == "Team":
    team.aus()

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)

# Home: Done
# Market: Add back NCID, correlation matrix (Top 10) (day, week, year)
# Info: image price volumn bar + overall price chart + introduction + whitepaper
# Analyzer: Price Chart (1m, 1h, 1d), Technical Analysis, News, Social Media, Twitter
# Forecast: Trading Views, News Sentiment Analysis, Social Media Analysis, Twitter Analysis, AI models Predictions (Logistic regression, LSTM, Random Forest), (1m, 1h, 1d)
# Portfolio: calculate the investment proportion of the cryptos (Next day)
# Team: Add description and photos