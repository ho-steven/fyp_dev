
import streamlit as st
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from datetime import date
import pypfopt
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import discrete_allocation
from pypfopt.cla import CLA
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import ffn
import plotly.express as px
import seaborn as sns
from pandas_datareader import data as web


yf.pdr_override()

def get_portfolio():

    start_date = st.date_input("Input start date of your investment", date(2021, 1, 1))
    portfolio_action = st.selectbox('Select your objective:',
                              ['Show Detailed Comparison Report','Calculate the current Optimal Portfolio','Maximise Return with Given Risk', 'Minimize Risk with Given Return'])



    if portfolio_action == "Show Detailed Comparison Report":
        portfolio_analysis(start_date)
            #yfPrice = yf.download(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"], start = start_date)
            #prices = yfPrice["Close"].dropna(how="all")
            # to get the initial weights
            #S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            #ef = EfficientFrontier(None, S, weight_bounds=(0, 1))
            #ef.min_volatility()
            #weights = ef.clean_weights()
            #weighting = pd.DataFrame(list(weights.items()),
            #          columns=['Ticker','Weighting'])

            #st.dataframe(weighting)
            
    if portfolio_action == "Calculate the current Optimal Portfolio":
        #start_date2 = st.date_input("Input start date for calculation", date(2021, 1, 1))
        btc_weight = st.slider('Select the weighting for BTC-USD:',0.0, 1.0, 0.25)
        eth_weight = st.slider('Select the weighting for ETH-USD:',0.0, 1.0, 0.25)
        bnb_weight = st.slider('Select the weighting for BNB-USD:',0.0, 1.0, 0.25)
        sol_weight = st.slider('Select the weighting for SOL-USD:',0.0, 1.0, 0.25)
        #value = st.number_input('Investment Amount', value=1000000)
        

        if st.button("Submit for calculation"):
            if (btc_weight + eth_weight + bnb_weight + sol_weight) == 1.0:
                # portfolio weights
                prices = ffn.get('BTC-USD,ETH-USD,BNB-USD,SOL-USD', start=start_date)
                weights = np.asarray([btc_weight,eth_weight,bnb_weight,sol_weight])              
                returns = prices.pct_change()            
                # mean daily return and covariance of daily returns
                mean_daily_returns = returns.mean()
                cov_matrix = returns.cov()      
                # portfolio return and volatility
                pf_return = round(np.sum(mean_daily_returns * weights) * 252, 3)
                pf_std_dev = round(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252), 3)
                st.markdown("***")
                st.info("The expected annualized return and volatility calculated with the given weightings:")
                st.write("Expected annualized return: " + "{:.1%}".format(pf_return))
                st.write("Volatility: " + "{:.1%}".format(pf_std_dev))
                st.write("")
                # Expected returns and sample covariance
                exp_returns = expected_returns.mean_historical_return(prices)
                covar = risk_models.sample_cov(prices)
                st.info("Calculate portfolio weights that maximize Sharpe ratio (No short):")
                # Optimise portfolio for maximum Sharpe Ratio
                ef = EfficientFrontier(exp_returns, covar)
                raw_weights = ef.max_sharpe()
                pf = ef.clean_weights()
                weighting = pd.DataFrame(list(pf.items()),columns=['Ticker','Weighting'])
                st.dataframe(weighting)
                #pd.Series(pf).plot.pie(figsize=(10,10))
                #plt.savefig('Images/pie1.png')
                #from PIL import Image
                #image_price_compare = Image.open('Images/pie1.png')
                #cola, colb, colc = st.columns([1, 5, 1])
                #with cola:
                #    st.write(' ')
                #with colb:
                #    st.image(image_price_compare, caption='Calculate portfolio weights that maximize Sharpe ratio')
                #with colc:
                #    st.write(' ')
                perf = ef.portfolio_performance(verbose=True)
                st.write("Expected annual return: ", '{:.1%}'.format(perf[0]))
                st.write("Annual volatility: ", '{:.1%}'.format(perf[1]))
                st.write("Sharpe Ratio: ", perf[2])
                from pypfopt import plotting
                from matplotlib.ticker import FuncFormatter
                cla = CLA(exp_returns, covar)
                ax = plotting.plot_efficient_frontier(cla, showfig = True)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                plt.savefig('Images/ef.png')
                from PIL import Image
                image_ef = Image.open('Images/ef.png')
                cola, colb, colc = st.columns([1, 5, 1])
                with cola:
                    st.write(' ')
                with colb:
                    st.image(image_ef, caption='Efficient Frontier')
                with colc:
                    st.write(' ')
                st.write("")
                st.info("Calculate portfolio weights that maximize Sharpe ratio (Allow short):")
                ef_short = EfficientFrontier(exp_returns, covar, weight_bounds=(-1, 1))
                pf2 = ef_short.efficient_return(target_return=perf[0])
                weighting2 = pd.DataFrame(list(pf2.items()),columns=['Ticker','Weighting'])
                st.dataframe(weighting2)
                perf2 = ef_short.portfolio_performance(verbose=True)
                st.write("Expected annual return: ", '{:.1%}'.format(perf2[0]))
                st.write("Annual volatility: ", '{:.1%}'.format(perf2[1]))
                st.write("Sharpe Ratio: ", perf2[2])
                st.write("")


                #st.info("Calculate portfolio weights breakdown")
                #latest_prices = discrete_allocation.get_latest_prices(prices)
                #allocation, leftover = discrete_allocation.DiscreteAllocation(pf, latest_prices, total_portfolio_value=value).lp_portfolio()
                #btcusd_amount = allocation['btcusd']
                #bnbusd_amount = allocation['bnbusd']
                #solusd_amount = allocation['solusd']
                #ethusd_amount = allocation['ethusd'] 
                #st.write('Allocation Result is BTC-USD: ',btcusd_amount,'BNB-USD: ', bnbusd_amount,'SOL-USD: ', solusd_amount,'ETH-USD: ', ethusd_amount )               
                #st.write("Funds remaining: ${:.2f}".format(leftover))
            else:
                st.error("The sum of the weightings must be equal to 1.0. Please submit again.")

    if portfolio_action == "Maximise Return with Given Risk":
        volatility = st.slider('Select your target volatility:',0.0, 1.0, 0.8)
        if st.button("Get result"):
            try:
                st.info("Portfolio that Maximise Return with Given Risk:")
                prices = ffn.get('BTC-USD,ETH-USD,BNB-USD,SOL-USD', start=start_date)
                exp_returns = expected_returns.mean_historical_return(prices)
                covar = risk_models.sample_cov(prices)
                ef = EfficientFrontier(exp_returns, covar)
                ef.efficient_risk(target_volatility=volatility)
                weights = ef.clean_weights()
                weighting = pd.DataFrame(list(weights.items()),columns=['Ticker','Weighting'])
                st.dataframe(weighting)

                perf = ef.portfolio_performance(verbose=True)
                st.write("Expected annual return: ", '{:.1%}'.format(perf[0]))
                st.write("Annual volatility: ", '{:.1%}'.format(perf[1]))
                st.write("Sharpe Ratio: ", perf[2])

                pd.Series(weights).plot.pie(figsize=(10,10))
                plt.savefig('Images/port_given_risk.png')
                from PIL import Image
                image_ef = Image.open('Images/port_given_risk.png')
                cola, colb, colc = st.columns([1, 5, 1])
                with cola:
                    st.write(' ')
                with colb:
                    st.image(image_ef, caption='The Optimal Portfolio with given risk')
                with colc:
                    st.write(' ')              

            except:
                st.error("Please adjust your volatility level to get a valid result")

    if portfolio_action == "Minimize Risk with Given Return":
        tar_return = st.slider('Select your desired return:',0.0, 1.0, 0.6)
        if st.button("Get portfolio"):
            try:
                st.info("Portfolio that Minimize Risk with Given Return:")
                prices = ffn.get('BTC-USD,ETH-USD,BNB-USD,SOL-USD', start=start_date)
                exp_returns = expected_returns.mean_historical_return(prices)
                covar = risk_models.sample_cov(prices)
                ef = EfficientFrontier(exp_returns, covar)
                ef.efficient_return(target_return=tar_return)
                weights = ef.clean_weights()
                weighting = pd.DataFrame(list(weights.items()),columns=['Ticker','Weighting'])
                st.dataframe(weighting)

                perf = ef.portfolio_performance(verbose=True)
                st.write("Expected annual return: ", '{:.1%}'.format(perf[0]))
                st.write("Annual volatility: ", '{:.1%}'.format(perf[1]))
                st.write("Sharpe Ratio: ", perf[2])
                pd.Series(weights).plot.pie(figsize=(10,10))
                plt.savefig('Images/port_given_return.png')
                from PIL import Image
                image_ef = Image.open('Images/port_given_return.png')
                cola, colb, colc = st.columns([1, 5, 1])
                with cola:
                    st.write(' ')
                with colb:
                    st.image(image_ef, caption='The Optimal Portfolio with given return')
                with colc:
                    st.write(' ')   

            except:
                st.error("Please adjust your return level to get a valid result")

def portfolio_analysis(start_date):

    st.subheader("Portfolio Management and Comparison on Selected Cryptocurrencies")
    st.markdown("***")
    st.info("Price Comparison from the chosen start date")
    #prices = yf.download(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"], start = '2021-01-01')
    prices = ffn.get('BTC-USD,ETH-USD,BNB-USD,SOL-USD', start = start_date)
    st.line_chart(prices)
    st.info("Price Comparison (Set the start date as the comparison base)")
    price_compare = prices.rebase().plot()
    price_compare.figure.savefig('Images/price_compare.png')
    from PIL import Image
    image_price_compare = Image.open('Images/price_compare.png')
    

    col1, col2, col3 = st.columns([1, 5, 1])

    with col1:
        st.write(' ')

    with col2:
        st.image(image_price_compare, caption='Price Comparison of the four selected cryptocurrencies')

    with col3:
        st.write(' ')

    st.info("Price Drawdown of the four selected cryptocurrencies")
    price_drawdown = prices.to_drawdown_series().plot()
    price_drawdown.figure.savefig('Images/price_drawdown.png')
    image_price_drawdown = Image.open('Images/price_drawdown.png')
    

    col4, col5, col6 = st.columns([1, 5, 1])

    with col4:
        st.write(' ')

    with col5:
        st.image(image_price_drawdown, caption='Price Drawdown')

    with col6:
        st.write(' ')

    st.info("Daily Returns Breakdown")
    daily_returns = prices.pct_change().dropna()
    st.line_chart(daily_returns)
    #fig = px.line(daily_returns[['btcusd', 'ethusd', 'bnbusd', 'solusd']])
    #fig.write_image(file='Images/daily_return.png', format='.png')
    #image_daily_returns = Image.open('Images/daily_returns.png')
    #st.image(image_daily_returns, caption='daily_returns')

    st.info("Normal Distribution of daily return")
    sns.displot(data=daily_returns[['btcusd', 'ethusd','bnbusd','solusd']], kind = 'kde', aspect = 2.5)
    plt.xlim(-0.1, 0.1)
    plt.savefig('Images/normal_distribute_daily_return.png')
    image_nd_daily_return = Image.open('Images/normal_distribute_daily_return.png')
    st.image(image_nd_daily_return, caption='Normal Distribution of daily return')

    st.info("Daily Return distribution breakdown")
    returns = prices.to_returns().dropna()
    returns.hist(figsize=(10, 5))
    plt.savefig('Images/price_nd_breakdown.png')
    image_price_breakdown = Image.open('Images/price_nd_breakdown.png')
    st.image(image_price_breakdown, caption='Daily Return distribution breakdown')

    st.info("Detailed Comparison of selected cryptocurrencies")
    stats = prices.calc_stats()
    stats.to_csv(sep=',', path='csv/compare_stats.csv')
    df_stats = pd.read_csv('csv/compare_stats.csv',usecols = ['Stat','btcusd','ethusd','bnbusd','solusd'])
    df_stats = df_stats.dropna(how='all')
    #df_stats.dropna(how="all", inplace=True)
    #df_stats.to_csv('csv/compare_stats_new.csv', header=True)
    #df_stats_new = pd.read_csv('csv/compare_stats_new.csv')

    col7, col8, col9 = st.columns([1, 5, 1])

    with col7:
        st.write(' ')

    with col8:
        st.dataframe(df_stats)

    with col9:
        st.write(' ')


    
    