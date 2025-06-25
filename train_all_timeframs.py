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