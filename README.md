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
├── app.py              # Streamlit Dashboard
├── train_model.py      # LSTM model training
├── your_model.h5       # Trained LSTM model
├── scaler.pkl          # Scaler used to normalize inputs
├── README.md           # This file
```

---

## 🧪 LSTM Model Training (`train_model.py`)

```python
import pandas as pd
import numpy as np
import ccxt
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Fetch data from Binance
exchange = ccxt.binance()
df = exchange.fetch_ohlcv('METIS/USDT', '4h', limit=300)
df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Feature engineering
indicators = ta.momentum.RSIIndicator(df['close'])
df['rsi'] = indicators.rsi()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df = df.dropna()

features = df[['close', 'rsi', 'macd', 'macd_signal']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

# Create dataset
lookback = 10
X, y = [], []
for i in range(lookback, len(scaled)-1):
    X.append(scaled[i-lookback:i])
    y.append(scaled[i+1][0])

X, y = np.array(X), np.array(y)

# Train LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.1)

# Save model and scaler
model.save("your_model.h5")
joblib.dump(scaler, "scaler.pkl")
```

---

## 📊 Streamlit Dashboard (`app.py`)

```python
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import joblib
import ta
from datetime import datetime
from tensorflow.keras.models import load_model
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

st_autorefresh(interval=300000, key="refresh")  # 5 minutes

model = load_model("your_model.h5")
scaler = joblib.load("scaler.pkl")

# Fetch data
df = ccxt.binance().fetch_ohlcv('BTC/USDT', '4h', limit=150)
df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df = df.dropna()

# Prepare input
lookback = 10
recent = df[['close', 'rsi', 'macd', 'macd_signal']].tail(lookback)
scaled_input = scaler.transform(recent).reshape(1, lookback, 4)
predicted_scaled = model.predict(scaled_input)[0][0]
predicted_price = scaler.inverse_transform([[predicted_scaled, 0, 0, 0]])[0][0]

# Display dashboard
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📈 METIS/USDT Prediction (4H) - LSTM")

fig = go.Figure(data=[
    go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'])
])
st.plotly_chart(fig, use_container_width=True)

curr = df['close'].iloc[-1]
delta = predicted_price - curr
signal = "BUY" if predicted_price > curr else "SELL" if predicted_price < curr else "HOLD"

st.metric("Current Price", f"${curr:.2f}")
st.metric("Predicted Price (Next 4H)", f"${predicted_price:.2f}", delta=f"{delta:.2f}")
st.markdown(f"### 🚀 Signal: **{signal}**")
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
pip install streamlit ccxt pandas numpy ta tensorflow joblib plotly streamlit-autorefresh
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
