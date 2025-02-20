import yfinance as yf
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pandas as pd
import streamlit as st # type: ignore

# RSIを計算する関数
# RSI (Relative Strength Index) は、価格の上昇・下落の強さを0-100の範囲で示す指標
# window: RSIの計算期間（デフォルト14日）
def compute_rsi(data, window=14):
    # 1日ごとの価格変動を計算
    diff = data.diff(1).dropna()
    # 上昇幅と下落幅を別々に計算
    gain = 0 * diff
    loss = 0 * diff
    gain[diff > 0] = diff[diff > 0]  # 価格上昇分
    loss[diff < 0] = -diff[diff < 0]  # 価格下落分
    # 上昇平均と下落平均を計算
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    # RSの計算（Relative Strength）
    rs = avg_gain / avg_loss
    # RSIの計算（0-100の範囲に変換）
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACDを計算する関数
# MACD (Moving Average Convergence Divergence) は、短期と長期の指数移動平均線の差を示す指標
# short_window: 短期EMAの期間（デフォルト12日）
# long_window: 長期EMAの期間（デフォルト26日）
# signal_window: シグナル線の期間（デフォルト9日）
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    # 短期と長期のEMAを計算
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    # MACDライン（短期EMA - 長期EMA）を計算
    macd = short_ema - long_ema
    # シグナルライン（MACDのEMA）を計算
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# ボリンジャーバンドを計算する関数
# ボリンジャーバンドは、移動平均線を中心に標準偏差の幅で上下のバンドを設定する指標
# window: 移動平均の期間（デフォルト20日）
# num_std: 標準偏差の倍率（デフォルト2）
def compute_bollinger_bands(data, window=20, num_std=2):
    # 移動平均（中央線）を計算
    rolling_mean = data.rolling(window=window).mean()
    # 標準偏差を計算
    rolling_std = data.rolling(window=window).std()
    # 上部バンド（移動平均 + 標準偏差×倍率）を計算
    upper_band = rolling_mean + (rolling_std * num_std)
    # 下部バンド（移動平均 - 標準偏差×倍率）を計算
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

# Appleの株価データを取得（2021年1月1日から2023年11月2日まで）
apple_data = yf.download('AAPL', start='2021-01-01', end='2023-11-02')

# 複数の期間の移動平均線を計算
# 5日、21日、25日、50日、75日、100日、200日の移動平均を計算
ma_windows = [5, 21, 25, 50, 75, 100, 200]
for window in ma_windows:
    apple_data[f'{window}_MA'] = apple_data['Close'].rolling(window=window).mean()

# RSI指標を計算（14日期間）
apple_data['RSI'] = compute_rsi(apple_data['Close'])

# MACD指標とシグナルラインを計算
apple_data['MACD'], apple_data['Signal'] = compute_macd(apple_data['Close'])

# ボリンジャーバンドを計算（20日期間、標準偏差2倍）
apple_data['Upper_Band'], apple_data['Middle_Band'], apple_data['Lower_Band'] = compute_bollinger_bands(apple_data['Close'])

# Streamlitアプリケーションのタイトルを設定
st.title('Apple (AAPL) Stock Analysis')

# グラフを3つのサブプロットで構成（株価チャート、RSI、MACD）
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

# 1番目のサブプロット：株価チャートと各種指標
ax1.plot(apple_data['Close'], label='AAPL Close Price', color='blue')
# 各期間の移動平均線をプロット
for window in ma_windows:
    ax1.plot(apple_data[f'{window}_MA'], label=f'{window}-Day MA')
# ボリンジャーバンドをプロット
ax1.plot(apple_data['Upper_Band'], label='Upper Bollinger Band', color='orange')
ax1.plot(apple_data['Middle_Band'], label='Middle Bollinger Band', color='gray')
ax1.plot(apple_data['Lower_Band'], label='Lower Bollinger Band', color='purple')
ax1.set_title('Apple (AAPL) Stock Price with Moving Averages and Bollinger Bands')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True)

# 2番目のサブプロット：RSIチャート
ax2.plot(apple_data['RSI'], label='RSI', color='purple')
# RSIの売られすぎ（30）と買われすぎ（70）のラインを表示
ax2.axhline(70, color='red', linestyle='--')
ax2.axhline(30, color='green', linestyle='--')
ax2.set_title('Apple (AAPL) Relative Strength Index (RSI)')
ax2.set_ylabel('RSI')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True)

# 3番目のサブプロット：MACDチャート
ax3.plot(apple_data['MACD'], label='MACD', color='blue')
ax3.plot(apple_data['Signal'], label='Signal', color='red')
ax3.set_title('Apple (AAPL) MACD')
ax3.set_ylabel('MACD')
ax3.set_xlabel('Date')
ax3.legend()
ax3.grid(True)

# Streamlitでグラフを表示
st.pyplot(fig)