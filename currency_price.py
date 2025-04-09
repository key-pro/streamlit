import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import talib
import numpy as np
from plotly.subplots import make_subplots

# ページ設定
st.set_page_config(
    page_title="為替分析ダッシュボード",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="collapsed"  # サイドバーを初期で折りたたむ
)

# スタイル設定
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1600px;  # 最大幅を広げる
        margin: 0 auto;
    }
    .currency-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #333333;
    }
    .indicator-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #333333;
    }
    .indicator-card h4 {
        color: #1f77b4;
        margin-bottom: 15px;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .indicator-card h3 {
        color: #2c3e50;
        font-size: 2rem;
        margin: 15px 0;
        font-weight: bold;
    }
    .indicator-card p {
        color: #666666;
        margin: 10px 0;
        font-size: 1.2rem;
    }
    .stMarkdown {
        color: #333333;
    }
    h3 {
        color: #2c3e50;
        margin-top: 30px;
        font-size: 1.8rem;
        font-weight: bold;
    }
    h4 {
        color: #2c3e50;
        margin-top: 20px;
        font-size: 1.5rem;
        font-weight: bold;
    }
    # チャートコンテナのスタイルを追加
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ページの背景色を設定
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# ヘッダー
st.title("💹 為替分析ダッシュボード")
st.markdown("---")

# サイドバー設定
st.sidebar.title("設定")
currency_pair = st.sidebar.selectbox(
    "通貨ペアを選択",
    ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"]
)

timeframe = st.sidebar.selectbox(
    "時間枠を選択",
    ["1日", "1週間", "1ヶ月", "3ヶ月", "6ヶ月", "1年"]
)

# データ取得関数
def get_currency_data(pair, period):
    symbol = pair.replace("/", "")
    if period == "1日":
        interval = "1m"
        period = "1d"
    elif period == "1週間":
        interval = "15m"
        period = "5d"
    elif period == "1ヶ月":
        interval = "1h"
        period = "1mo"
    else:
        interval = "1d"
        period = period.replace("ヶ月", "mo").replace("年", "y")
    
    data = yf.download(f"{symbol}=X", period=period, interval=interval)
    return data

# テクニカル指標計算関数
def calculate_indicators(data):
    # データを1次元のnumpy配列に変換
    close = np.array(data['Close'], dtype=np.float64)
    high = np.array(data['High'], dtype=np.float64)
    low = np.array(data['Low'], dtype=np.float64)
    
    # NaN値を除去
    close = close[~np.isnan(close)]
    high = high[~np.isnan(high)]
    low = low[~np.isnan(low)]
    
    # RSI
    rsi = talib.RSI(close, timeperiod=14)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close)
    
    # ボリンジャーバンド
    upper_band, middle_band, lower_band = talib.BBANDS(close, timeperiod=20)
    
    # ストキャスティクス
    slowk, slowd = talib.STOCH(high, low, close)
    
    # 移動平均
    sma_20 = talib.SMA(close, timeperiod=20)
    sma_50 = talib.SMA(close, timeperiod=50)
    sma_200 = talib.SMA(close, timeperiod=200)
    
    # ADX
    adx = talib.ADX(high, low, close, timeperiod=14)
    
    # ATR
    atr = talib.ATR(high, low, close, timeperiod=14)
    
    # 結果をPandas Seriesに変換
    return {
        'rsi': pd.Series(rsi, index=data.index[-len(rsi):]),
        'macd': pd.Series(macd, index=data.index[-len(macd):]),
        'macd_signal': pd.Series(macd_signal, index=data.index[-len(macd_signal):]),
        'macd_hist': pd.Series(macd_hist, index=data.index[-len(macd_hist):]),
        'upper_band': pd.Series(upper_band, index=data.index[-len(upper_band):]),
        'middle_band': pd.Series(middle_band, index=data.index[-len(middle_band):]),
        'lower_band': pd.Series(lower_band, index=data.index[-len(lower_band):]),
        'slowk': pd.Series(slowk, index=data.index[-len(slowk):]),
        'slowd': pd.Series(slowd, index=data.index[-len(slowd):]),
        'sma_20': pd.Series(sma_20, index=data.index[-len(sma_20):]),
        'sma_50': pd.Series(sma_50, index=data.index[-len(sma_50):]),
        'sma_200': pd.Series(sma_200, index=data.index[-len(sma_200):]),
        'adx': pd.Series(adx, index=data.index[-len(adx):]),
        'atr': pd.Series(atr, index=data.index[-len(atr):])
    }

# メインコンテンツ
col1, col2 = st.columns([1, 4])  # 比率を1:4に変更

with col1:
    st.markdown("### 為替レート")
    data = get_currency_data(currency_pair, timeframe)
    indicators = calculate_indicators(data)
    
    # 現在のレートカード
    current_rate = float(data['Close'].iloc[-1])
    previous_rate = float(data['Close'].iloc[-2])
    change = ((current_rate - previous_rate) / previous_rate) * 100
    
    st.markdown(f"""
    <div class="currency-card">
        <h3>{currency_pair}</h3>
        <h2 style="font-size: 2.5rem;">{current_rate:.2f}</h2>
        <p style="color: {'red' if change < 0 else 'green'}; font-size: 1.2rem;">
            {change:+.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### チャート")
    # メインチャートとテクニカル指標を縦に並べて表示
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,  # スペースをより詰める
        row_heights=[0.6, 0.15, 0.15, 0.15],  # メインチャートの比率をさらに増加
        subplot_titles=(
            f"{currency_pair} チャート",
            "RSI (相対力指数)",
            "MACD (移動平均収束拡散指標)",
            "ストキャスティクス"
        )
    )
    
    # メインチャート（ローソク足）
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='ローソク足',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # 移動平均線
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['sma_20'],
            name='SMA20 (20日)',
            line=dict(color='#1976D2', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['sma_50'],
            name='SMA50 (50日)',
            line=dict(color='#FB8C00', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['sma_200'],
            name='SMA200 (200日)',
            line=dict(color='#7B1FA2', width=1)
        ),
        row=1, col=1
    )
    
    # ボリンジャーバンド
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['upper_band'],
            name='ボリンジャー上限',
            line=dict(color='rgba(128, 128, 128, 0.6)', width=1, dash='dash'),
            fill=None
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['lower_band'],
            name='ボリンジャー下限',
            line=dict(color='rgba(128, 128, 128, 0.6)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['rsi'],
            name='RSI (14)',
            line=dict(color='#2196F3', width=1.5)
        ),
        row=2, col=1
    )
    
    # RSIのオーバーボート/アンダーボートライン
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 82, 82, 0.8)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(76, 175, 80, 0.8)", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['macd'],
            name='MACD',
            line=dict(color='#E91E63', width=1.5)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['macd_signal'],
            name='シグナル',
            line=dict(color='#FF9800', width=1.5)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=indicators['macd_hist'],
            name='ヒストグラム',
            marker_color=np.where(indicators['macd_hist'] >= 0, '#26a69a', '#ef5350'),
            marker_line_width=0
        ),
        row=3, col=1
    )
    
    # ストキャスティクス
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['slowk'],
            name='%K',
            line=dict(color='#4CAF50', width=1.5)
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['slowd'],
            name='%D',
            line=dict(color='#9C27B0', width=1.5)
        ),
        row=4, col=1
    )
    
    # ストキャスティクスのオーバーボート/アンダーボートライン
    fig.add_hline(y=80, line_dash="dash", line_color="rgba(255, 82, 82, 0.8)", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="rgba(76, 175, 80, 0.8)", row=4, col=1)
    
    # レイアウトの更新
    fig.update_layout(
        height=1000,  # 高さを1000pxに増加
        showlegend=True,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=12)  # 凡例のフォントサイズを大きく
        ),
        margin=dict(r=150, l=50, t=50, b=50),
    )
    
    # フォントサイズの調整
    fig.update_xaxes(
        tickfont=dict(size=12),  # フォントサイズを大きく
        titlefont=dict(size=14)
    )
    
    fig.update_yaxes(
        tickfont=dict(size=12),  # フォントサイズを大きく
        titlefont=dict(size=14)
    )
    
    # サブプロットのタイトルスタイルを更新
    fig.update_annotations(font_size=16)  # タイトルのフォントサイズを大きく
    
    # チャートの表示
    st.plotly_chart(fig, use_container_width=True, height=1000)
    st.markdown('</div>', unsafe_allow_html=True)

# テクニカル指標
st.markdown("### テクニカル指標")
col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("#### トレンド系指標")
    st.markdown(f"""
    <div class="indicator-card">
        <h4>ADX (14日)</h4>
        <h3>{float(indicators['adx'].iloc[-1]):.2f}</h3>
        <p>トレンド強度: <span style="color: {'#2196F3' if float(indicators['adx'].iloc[-1]) > 25 else '#666666'}">{'強い' if float(indicators['adx'].iloc[-1]) > 25 else '弱い'}</span></p>
    </div>
    
    <div class="indicator-card">
        <h4>ATR (14日)</h4>
        <h3>{float(indicators['atr'].iloc[-1]):.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("#### オシレーター系指標")
    st.markdown(f"""
    <div class="indicator-card">
        <h4>RSI (14日)</h4>
        <h3>{float(indicators['rsi'].iloc[-1]):.2f}</h3>
        <p>シグナル: <span style="color: {'red' if float(indicators['rsi'].iloc[-1]) > 70 else 'green' if float(indicators['rsi'].iloc[-1]) < 30 else '#666666'}">
            {'売られすぎ' if float(indicators['rsi'].iloc[-1]) < 30 else '買われすぎ' if float(indicators['rsi'].iloc[-1]) > 70 else '中立'}
        </span></p>
    </div>
    
    <div class="indicator-card">
        <h4>ストキャスティクス</h4>
        <p style="font-size: 1.4rem;">%K: {float(indicators['slowk'].iloc[-1]):.2f}</p>
        <p style="font-size: 1.4rem;">%D: {float(indicators['slowd'].iloc[-1]):.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("#### トレンドフォロー系指標")
    st.markdown(f"""
    <div class="indicator-card">
        <h4>MACD</h4>
        <p style="font-size: 1.4rem;">MACD: {float(indicators['macd'].iloc[-1]):.3f}</p>
        <p style="font-size: 1.4rem;">シグナル: {float(indicators['macd_signal'].iloc[-1]):.3f}</p>
        <p style="font-size: 1.4rem;">ヒストグラム: <span style="color: {'#26a69a' if float(indicators['macd_hist'].iloc[-1]) >= 0 else '#ef5350'}">{float(indicators['macd_hist'].iloc[-1]):.3f}</span></p>
    </div>
    
    <div class="indicator-card">
        <h4>ボリンジャーバンド</h4>
        <p style="font-size: 1.4rem;">上限: {float(indicators['upper_band'].iloc[-1]):.2f}</p>
        <p style="font-size: 1.4rem;">中央: {float(indicators['middle_band'].iloc[-1]):.2f}</p>
        <p style="font-size: 1.4rem;">下限: {float(indicators['lower_band'].iloc[-1]):.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# フッター
st.markdown("---")
st.markdown("© 2025 kamiyuki key-project")