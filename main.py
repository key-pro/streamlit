import yfinance as yf
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pandas as pd
import streamlit as st # type: ignore

# RSIを計算する関数
def compute_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = 0 * diff
    loss = 0 * diff
    gain[diff > 0] = diff[diff > 0]
    loss[diff < 0] = -diff[diff < 0]
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACDを計算する関数
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# ボリンジャーバンドを計算する関数
def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

# Appleの株価データ取得
apple_data = yf.download('AAPL', start='2021-01-01', end='2023-11-02')

# 移動平均線を計算
ma_windows = [5, 21, 25, 50, 75, 100, 200]
for window in ma_windows:
    apple_data[f'{window}_MA'] = apple_data['Close'].rolling(window=window).mean()

# RSIを計算
apple_data['RSI'] = compute_rsi(apple_data['Close'])

# MACDを計算
apple_data['MACD'], apple_data['Signal'] = compute_macd(apple_data['Close'])

# ボリンジャーバンドを計算
apple_data['Upper_Band'], apple_data['Middle_Band'], apple_data['Lower_Band'] = compute_bollinger_bands(apple_data['Close'])

# Streamlitのタイトル
st.title('Apple (AAPL) Stock Analysis')

# 株価の終値、移動平均線、ボリンジャーバンドをプロット
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
ax1.plot(apple_data['Close'], label='AAPL Close Price', color='blue')
for window in ma_windows:
    ax1.plot(apple_data[f'{window}_MA'], label=f'{window}-Day MA')
ax1.plot(apple_data['Upper_Band'], label='Upper Bollinger Band', color='orange')
ax1.plot(apple_data['Middle_Band'], label='Middle Bollinger Band', color='gray')
ax1.plot(apple_data['Lower_Band'], label='Lower Bollinger Band', color='purple')
ax1.set_title('Apple (AAPL) Stock Price with Moving Averages and Bollinger Bands')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True)

# RSIをプロット
ax2.plot(apple_data['RSI'], label='RSI', color='purple')
ax2.axhline(70, color='red', linestyle='--')
ax2.axhline(30, color='green', linestyle='--')
ax2.set_title('Apple (AAPL) Relative Strength Index (RSI)')
ax2.set_ylabel('RSI')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True)

# MACDをプロット
ax3.plot(apple_data['MACD'], label='MACD', color='blue')
ax3.plot(apple_data['Signal'], label='Signal', color='red')
ax3.set_title('Apple (AAPL) MACD')
ax3.set_ylabel('MACD')
ax3.set_xlabel('Date')
ax3.legend()
ax3.grid(True)

# Streamlitでプロットを表示
st.pyplot(fig)