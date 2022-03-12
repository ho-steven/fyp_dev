import streamlit as st
from multiapp import MultiApp
from apps import home, market, crypto, strategy # import your app modules here


app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Crypto Market", market.app)
app.add_app("Crypto Analyzer", crypto.app)
app.add_app("AI prediction models and Portfolio Management", strategy.app)


# The main app
app.run()
