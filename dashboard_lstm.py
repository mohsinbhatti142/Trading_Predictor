import pandas as pd
import numpy as np
import ccxt
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Fetch 4h BTC/USDT data
def fetch_ohlcv(symbol='METIS/USDT', timeframe='1h', limit=10000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Add RSI and MACD
def add_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df.dropna()

# Prepare dataset
def create_dataset(df, lookback=10):
    features = df[['close', 'rsi', 'macd', 'macd_signal']].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(lookback, len(scaled)-1):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i+1][0])  # predict next close price (scaled)
    return np.array(X), np.array(y), scaler

# Train and save LSTM model
def train_lstm():
    df = fetch_ohlcv()
    df = add_indicators(df)
    X, y, scaler = create_dataset(df)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=50, batch_size=16, validation_split=0.1, verbose=1,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    model.save("your_model.h5")
    print("✅ LSTM model trained and saved as your_model.h5")

    # Save the scaler
    import joblib
    joblib.dump(scaler, "scaler.pkl")

if __name__ == "__main__":
    train_lstm()
