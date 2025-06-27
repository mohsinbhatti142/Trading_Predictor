import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import streamlit.components.v1 as components

# --- Dashboard UI ---
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

st.title("🚀 Elite BTC/USDT Live Trading Dashboard + ML Prediction")
st.markdown("""
A professional, real-time dashboard inspired by Binance. Live price, order book, and ML-based prediction for Bitcoin (BTC/USDT).
""")

# --- Sidebar for controls ---
timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
timeframe = st.sidebar.selectbox('Chart Timeframe', timeframes, index=3)
limit = st.sidebar.slider('Candles to show', min_value=50, max_value=1000, value=200, step=50)

# --- Fetch live OHLCV data ---
def fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=200):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    import ta
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df.dropna()

df = fetch_ohlcv('BTC/USDT', timeframe, limit)
df = add_indicators(df)

# --- Live price ticker ---
latest = df.iloc[-1]
st.markdown(f"""
<div style='font-size:2.5em; color:#00ff99; font-weight:bold;'>
    BTC/USDT: ${latest['close']:.2f}
    <span style='font-size:0.5em; color:#ffcc00;'>({timeframe} candle)</span>
</div>
""", unsafe_allow_html=True)

# --- TradingView Chart ---
st.header("📈 Live TradingView Chart - BTC/USDT")
tradingview_html = """
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({
      "width": "100%",
      "height": 600,
      "symbol": "BINANCE:BTCUSDT",
      "interval": "1",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#181818",
      "enable_publishing": false,
      "hide_top_toolbar": false,
      "container_id": "tradingview_chart"
    });
  </script>
</div>
<!-- TradingView Widget END -->
"""
components.html(tradingview_html, height=620)

# --- ML Prediction (LSTM) ---
# Use optimal lookback for each timeframe
lookback_dict = {
    '5m': 50,
    '15m': 50,
    '30m': 30,
    '1h': 30,
    '4h': 20,
    '1d': 10
}
lookback = lookback_dict.get(timeframe, 10)

def load_model_and_scaler(timeframe):
    try:
        model = load_model(f"model_{timeframe}.h5", compile=False)
        scaler = joblib.load(f"scaler_{timeframe}.pkl")
        return model, scaler
    except Exception as e:
        st.warning(f"ML model/scaler not found for {timeframe}. Showing chart only.\n{e}")
        return None, None

model, scaler = load_model_and_scaler(timeframe)

if model and scaler and len(df) >= lookback:
    features = df[['close', 'rsi', 'macd', 'macd_signal']].values[-lookback:]
    scaled = scaler.transform(features)
    X = scaled.reshape(1, lookback, 4)
    prediction_scaled = model.predict(X)
    close_index = 0
    close_min = scaler.data_min_[close_index]
    close_max = scaler.data_max_[close_index]
    prediction = prediction_scaled[0][0] * (close_max - close_min) + close_min
    current_price = df['close'].iloc[-1]
    delta = prediction - current_price
    signal = "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD"

    st.markdown(f"""
    <div style='font-size:1.5em; color:#00bfff; font-weight:bold;'>
        ML Prediction (Next {timeframe}): <span style='color:#ffcc00;'>${prediction:.2f}</span>
        <span style='font-size:1em; color:#fff;'>({signal})</span>
    </div>
    """, unsafe_allow_html=True)
    if signal == "BUY":
        st.success("✅ Signal: BUY - Predicted price is higher than current price.")
    elif signal == "SELL":
        st.error("🔻 Signal: SELL - Predicted price is lower than current price.")
    else:
        st.info("⏸️ Signal: HOLD - No significant change predicted.")

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
