# 📈 Real-Time Crypto Trading Algorithm with LSTM & Streamlit

This project is a real-time trading dashboard that uses machine learning to predict the next 4-hour price movement for METIS/USDT using Binance data. It includes technical indicators (RSI, MACD), an LSTM model trained on past price movements, and an interactive dashboard built with Streamlit.

---

## 🚀 Features

- ✅ Fetches live METIS/USDT OHLCV data from Binance (4-hour intervals)
- 📉 Calculates RSI and MACD using `ta` indicators
- 🧠 Predicts the next close price using an LSTM model
- 🔍 Displays a BUY / SELL / HOLD signal based on prediction
- 📊 Live candlestick chart using Plotly
- 📺 Embedded TradingView widget for real-time charting
- ♻️ Auto-refreshes every 5 minutes

---

## 🧰 Technologies Used

- Python 3.10+
- [ccxt](https://github.com/ccxt/ccxt)
- [pandas](https://pandas.pydata.org/)
- [ta](https://technical-analysis-library-in-python.readthedocs.io/en/latest/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)

---

## 📁 Folder Structure

```
trading-lstm-dashboard/
├── app.py                      # Streamlit Dashboard
├── train_all_timeframs.py      # LSTM model training for all timeframes
├── model_5m.h5                 # Trained LSTM model for 5m timeframe
├── scaler_5m.pkl               # Scaler for 5m timeframe
├── ...                         # Models and scalers for other timeframes
├── requirements.txt            # Python dependencies
├── README.md                   # This file
```

---

## 🧪 LSTM Model Training (`train_all_timeframs.py`)

```python
import ccxt
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
import joblib

symbol = 'METIS/USDT'
timeframes = ['5m', '30m', '1h', '4h', '1d', '1w']

def fetch_ohlcv(symbol, timeframe, limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df.dropna()

def create_dataset(df, lookback=10):
    features = df[['close', 'rsi', 'macd', 'macd_signal']].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(lookback, len(scaled)-1):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i+1][0])  # predict next close price (scaled)
    return np.array(X), np.array(y), scaler

def train_and_save_for_timeframe(timeframe):
    print(f"Training for {timeframe}...")
    df = fetch_ohlcv(symbol, timeframe)
    df = add_indicators(df)
    X, y, scaler = create_dataset(df)
    if len(X) < 10:
        print(f"Not enough data for {timeframe}, skipping.")
        return
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=16, validation_split=0.1, verbose=1,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    model.save(f"model_{timeframe}.h5")
    joblib.dump(scaler, f"scaler_{timeframe}.pkl")
    print(f"✅ Saved model_{timeframe}.h5 and scaler_{timeframe}.pkl")

if __name__ == "__main__":
    for tf in timeframes:
        train_and_save_for_timeframe(tf)
```

---

## 📊 Streamlit Dashboard (`app.py`)

```python
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=300000, key="refresh")

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
```

---

## 🔗 Live Chart with TradingView (Optional)
Use the following code in `st.components.v1.html()` to embed a chart:

```html
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({
      "width": "100%",
      "height": 620,
      "symbol": "BINANCE:BTCUSDT",
      "interval": "240",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "container_id": "tradingview_chart"
    });
  </script>
</div>
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Then run the app:
```bash
streamlit run app.py
```

---

## 📌 Notes
- This project is for **educational/demo purposes**. Do not use for real trading without risk management and regulatory compliance.
- You can extend this to other coins (ETH, BNB) and intervals (1h, 1d).
- Add backtesting for better strategy evaluation.

---

## 🤝 Contribute
Pull requests and feedback are welcome!

---

## 📧 Contact
If you found this useful or want to collaborate, connect on [LinkedIn](linkedin.com/in/mohsin-sheraz-142nb/) or drop a message!
