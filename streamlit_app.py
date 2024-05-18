# streamlit_app.py
import streamlit as st
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.title("Cryptocurrency Price Prediction")

tickers = ['BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'LINK/USDT']
timeframes = ['1w', '1d', '4h', '1h', '30m', '15m', '5m', '3m', '1m']

ticker = st.selectbox('Select Ticker', tickers)
timeframe = st.selectbox('Select Timeframe', timeframes)

if st.button('Train and Predict'):
    response = requests.post('https://render-crypto.onrender.com/predict', json={'ticker': ticker, 'timeframe': timeframe})
    data = response.json()
    
    st.write("### Latest Candle Predictions")
    for model, prediction in data['latest_candle_prediction'].items():
        direction = 'Bullish' if prediction == 1 else 'Bearish'
        st.write(f"{model}: {direction}")
    
    # Fetch and display candlestick plot
    price_data = data['price_data']
    df = pd.DataFrame(price_data)
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.write("### Model Evaluation Metrics")
    
    for model, metrics in data['results'].items():
        st.write(f"#### {model}")
        st.write(f"Accuracy: {metrics['accuracy']:.2f}")
        st.write(f"Precision: {metrics['precision']:.2f}")
        st.write(f"Recall: {metrics['recall']:.2f}")
        st.write(f"F1 Score: {metrics['f1_score']:.2f}")
        
        st.write("Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
