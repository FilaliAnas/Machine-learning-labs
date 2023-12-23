import yfinance as yf
import streamlit as st

# Define the stock symbol and download historical data

st.header("Stock Price Prediction")

stock_symbol = st.sidebar.selectbox('Select a stock symbol',['AAPL','GOOGL','TSLAs'])
start_date = st.sidebar.date_input('train start time')
end_date = st.sidebar.date_input('train end time')

data = yf.download(stock_symbol, start=start_date, end=end_date)
st.write(data)
print(data)