import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Elite Binance-style Dashboard for Live BTC Trading
st.set_page_config(page_title="🚀 Elite BTC Trading Dashboard", layout="wide")
st.markdown("""
<style>
    .main {background-color: #0a0a0a; color: #f2f2f2;}
    .st-bb {background: #181818;}
    .st-cg {color: #00ff99;}
    .st-cy {color: #ffcc00;}
    .st-cb {color: #00bfff;}
    .st-cw {color: #fff;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Elite BTC/USDT Live Trading Dashboard")
st.markdown("""
A professional, real-time dashboard inspired by Binance. Live price, order book, and interactive chart for Bitcoin (BTC/USDT).
""")

# --- Sidebar for controls ---
timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
timeframe = st.sidebar.selectbox('Chart Timeframe', timeframes, index=4)
limit = st.sidebar.slider('Candles to show', min_value=50, max_value=1000, value=200, step=50)

# --- Fetch live OHLCV data ---
def fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=2000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

df = fetch_ohlcv('BTC/USDT', timeframe, limit)

# --- Live price ticker ---
latest = df.iloc[-1]
st.markdown(f"""
<div style='font-size:2.5em; color:#00ff99; font-weight:bold;'>
    BTC/USDT: ${latest['close']:.2f}
    <span style='font-size:0.5em; color:#ffcc00;'>({timeframe} candle)</span>
</div>
""", unsafe_allow_html=True)

# --- Candlestick chart with volume ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"BTC/USDT Candlestick Chart ({timeframe})", "Volume"))

fig.add_trace(
    go.Candlestick(x=df['timestamp'],
                   open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                   name='Candlesticks',
                   increasing_line_color='#00ff99', decreasing_line_color='#ff3333'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='rgba(0, 150, 255, 0.3)'),
    row=2, col=1
)

fig.update_layout(
    xaxis_rangeslider_visible=True,
    xaxis=dict(type='date',
               rangeselector=dict(
                   buttons=list([
                       dict(count=1, label='1d', step='day', stepmode='backward'),
                       dict(count=7, label='1w', step='day', stepmode='backward'),
                       dict(count=1, label='1m', step='month', stepmode='backward'),
                       dict(step='all')
                   ])
               ),
               showspikes=True,
               spikemode='across',
               spikesnap='cursor',
               showline=True,
               showgrid=True,
               gridcolor='grey',
               ),
    yaxis_title='Price (USDT)',
    yaxis2_title='Volume',
    showlegend=False,
    hovermode='x unified',
    margin=dict(l=20, r=20, t=40, b=20),
    plot_bgcolor='#181818',
    dragmode='pan',
)

st.plotly_chart(fig, use_container_width=True)

# --- Order Book (Top 10 Bids/Asks) ---
def fetch_order_book(symbol='BTC/USDT', limit=10):
    exchange = ccxt.binance()
    ob = exchange.fetch_order_book(symbol, limit=limit)
    bids = pd.DataFrame(ob['bids'], columns=['Price', 'Amount'])
    asks = pd.DataFrame(ob['asks'], columns=['Price', 'Amount'])
    return bids, asks

bids, asks = fetch_order_book('BTC/USDT', 10)
col1, col2 = st.columns(2)
col1.markdown("<b style='color:#00ff99;'>Top 10 Bids</b>", unsafe_allow_html=True)
col1.dataframe(bids, use_container_width=True, height=300)
col2.markdown("<b style='color:#ff3333;'>Top 10 Asks</b>", unsafe_allow_html=True)
col2.dataframe(asks, use_container_width=True, height=300)

# --- Refresh button ---
st.button('🔄 Refresh Data')

st.caption('Elite dashboard by GitHub Copilot | Data: Binance API | For educational use only.')
