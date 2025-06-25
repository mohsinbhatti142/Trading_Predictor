import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# ---------------- Load trained model ----------------
@st.cache_resource
def load_model_cached():
    try:
        return load_model("your_model.h5", compile=False)  # Fix: don't compile, just load for prediction
    except Exception as e:
        st.error(f"❌ your_model.h5 not found or invalid. Error: {e}")
        return None

model = load_model_cached()

# ---------------- Fetch data from Binance ----------------
timeframes = ['5m', '30m', '1h', '4h', '1d', '1w']
timeframe = st.selectbox('Select Timeframe', timeframes, index=2)

@st.cache_data(show_spinner=False)
def fetch_data(timeframe):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('METIS/USDT', timeframe, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Add indicators
    import ta
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df.dropna()

def load_model_and_scaler(timeframe):
    try:
        model = load_model(f"model_{timeframe}.h5", compile=False)
        import joblib
        scaler = joblib.load(f"scaler_{timeframe}.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Model or scaler for {timeframe} not found or invalid. Error: {e}")
        return None, None

# ---------------- Dashboard ----------------
st.set_page_config(page_title="📈 METIS Trading Dashboard", layout="wide")
st.title("📊 Real-Time METIS/USDT Price Prediction (4H)")
st.markdown("Uses ML model to predict next 1h price and gives trading signal.")

df = fetch_data(timeframe)
model, scaler = load_model_and_scaler(timeframe)

# Plot candlestick chart with volume and interactive features
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"METIS/USDT Candlestick Chart ({timeframe})", "Volume"))

fig.add_trace(
    go.Candlestick(x=df['timestamp'],
                   open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                   name='Candlesticks'),
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
               gridcolor='lightgrey',
               ),
    yaxis_title='Price (USDT)',
    yaxis2_title='Volume',
    showlegend=False,
    hovermode='x unified',
    margin=dict(l=20, r=20, t=40, b=20),
    plot_bgcolor='white',
    dragmode='pan',
)

st.plotly_chart(fig, use_container_width=True)

# Prediction and signal
if model and scaler and len(df) >= 10:
    features = df[['close', 'rsi', 'macd', 'macd_signal']].values[-10:]  # shape (10, 4)
    scaled = scaler.transform(features)
    X = scaled.reshape(1, 10, 4)
    prediction_scaled = model.predict(X)
    close_index = 0
    close_min = scaler.data_min_[close_index]
    close_max = scaler.data_max_[close_index]
    prediction = prediction_scaled[0][0] * (close_max - close_min) + close_min
    current_price = df['close'].iloc[-1]
    delta = prediction - current_price
    signal = "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD"

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Current Price", f"${current_price:.2f}")
    col2.metric(f"🔮 Predicted (Next {timeframe})", f"${prediction:.2f}", delta=f"{delta:.2f}")
    if signal == "BUY":
        col3.success("✅ Signal: BUY")
    elif signal == "SELL":
        col3.error("🔻 Signal: SELL")
    else:
        col3.warning("⏸️ Signal: HOLD")

