import yfinance as yf  # Yahoo Financeから株価データを取得するためのライブラリ
import matplotlib.pyplot as plt # type: ignore  # グラフ描画用ライブラリ
import numpy as np  # 数値計算用ライブラリ
import pandas as pd  # データ操作・分析用ライブラリ
import streamlit as st # type: ignore  # WebアプリケーションフレームワークStreamlit
from sklearn.linear_model import LinearRegression # type: ignore  # 線形回帰モデル用ライブラリ

# RSI(Relative Strength Index:相対力指数)を計算する関数
# RSIは、一定期間における値上がり幅の平均と値下がり幅の平均の比から算出される指標
def compute_rsi(data, window=14):
    diff = data.diff(1).dropna()  # 前日比を計算し、欠損値を除去
    gain = np.where(diff > 0, diff, 0).flatten()  # 値上がり幅を抽出
    loss = np.where(diff < 0, -diff, 0).flatten()  # 値下がり幅を抽出
    gain_series = pd.Series(gain, index=diff.index)  # 値上がり幅をSeriesに変換
    loss_series = pd.Series(loss, index=diff.index)  # 値下がり幅をSeriesに変換
    avg_gain = gain_series.rolling(window=window, min_periods=1).mean()  # 値上がり幅の移動平均
    avg_loss = loss_series.rolling(window=window, min_periods=1).mean()  # 値下がり幅の移動平均
    rs = avg_gain / avg_loss  # RSの計算（Relative Strength）
    rsi = 100 - (100 / (1 + rs))  # RSIの計算（0-100の範囲で表示）
    return rsi

# MACD(Moving Average Convergence Divergence:移動平均収束拡散法)を計算する関数
# MACDは、短期と長期の指数移動平均線の差を表す指標
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()  # 短期の指数移動平均
    long_ema = data.ewm(span=long_window, adjust=False).mean()  # 長期の指数移動平均
    macd = short_ema - long_ema  # MACDラインの計算
    signal = macd.ewm(span=signal_window, adjust=False).mean()  # シグナルラインの計算
    return macd, signal

# ボリンジャーバンドを計算する関数
# ボリンジャーバンドは、移動平均線を中心に標準偏差の幅で上下のバンドを設定する指標
def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()  # 移動平均の計算
    rolling_std = data['Close'].rolling(window=window).std()  # 標準偏差の計算
    upper_band = rolling_mean + (rolling_std * num_std)  # 上部バンドの計算
    lower_band = rolling_mean - (rolling_std * num_std)  # 下部バンドの計算
    return upper_band, lower_band

# Yahoo Financeから株価データを取得する関数
def get_stock_data(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)  # 指定期間の株価データを取得
    if data.empty:  # データが取得できなかった場合のエラーハンドリング
        st.error('⚠️ データが取得できません。銘柄コードを確認してください。')
        st.stop()
    return data

# RSIとMACDの将来値を予測する関数
def predict_future(data, column, days=5):
    data = data.dropna()  # 欠損値を除去
    X = np.arange(len(data)).reshape(-1, 1)  # 説明変数（時系列）
    y = data[column].values.reshape(-1, 1)  # 目的変数（指標値）
    model = LinearRegression()  # 線形回帰モデルのインスタンス化
    model.fit(X, y)  # モデルの学習
    future_X = np.arange(len(data), len(data) + days).reshape(-1, 1)  # 予測用の説明変数
    future_y = model.predict(future_X)  # 将来値の予測
    return future_y.flatten()

# 三角持ち合いを検出する関数を修正
def detect_triangle_pattern(data):
    try:
        # データの準備
        target_data = data.copy()
        highs = target_data['High'].values
        lows = target_data['Low'].values
        dates = target_data.index
        
        # 初期幅（H）の計算
        initial_height = max(highs[:20]) - min(lows[:20])
        
        # ピーク検出のパラメータ
        window = 5
        min_points = 4
        
        # 高値と安値のピークを検出
        high_peaks = []
        low_peaks = []
        high_dates = []
        low_dates = []
        
        # 高値のピークを検出
        for i in range(window, len(highs)-window):
            if highs[i] == max(highs[i-window:i+window+1]):
                if not high_peaks or (highs[i] < high_peaks[-1]):  # 下降トレンドの確認
                    high_peaks.append(highs[i])
                    high_dates.append(dates[i])
        
        # 安値のピークを検出
        for i in range(window, len(lows)-window):
            if lows[i] == min(lows[i-window:i+window+1]):
                if not low_peaks or (lows[i] > low_peaks[-1]):  # 上昇トレンドの確認
                    low_peaks.append(lows[i])
                    low_dates.append(dates[i])
        
        if len(high_peaks) < min_points or len(low_peaks) < min_points:
            return "データ不足", None, None, None, None, None
        
        # トレンドラインの計算
        high_x = np.arange(len(high_peaks))
        low_x = np.arange(len(low_peaks))
        
        # y = ax + b の係数を計算
        high_coeffs = np.polyfit(high_x, high_peaks, 1)  # a2, b2
        low_coeffs = np.polyfit(low_x, low_peaks, 1)    # a1, b1
        
        a1, b1 = low_coeffs   # 上昇トレンドライン
        a2, b2 = high_coeffs  # 下降トレンドライン
        
        # 収束点の計算
        if abs(a1 - a2) > 0.0001:  # ゼロ除算を防ぐ
            x_c = (b2 - b1) / (a1 - a2)
            y_c = a1 * x_c + b1
            
            # 収束点の日付を推定
            days_to_convergence = int(x_c - len(high_x) + 1)
            if days_to_convergence > 0:
                convergence_date = dates[-1] + pd.Timedelta(days=days_to_convergence)
            else:
                convergence_date = dates[-1]
        else:
            convergence_date = dates[-1]
            x_c = len(high_x)
            y_c = a1 * x_c + b1
        
        # パターン判定
        # 対称三角形: 上下のトレンドラインの傾きの絶対値がほぼ等しい
        is_symmetrical = (abs(abs(a1) - abs(a2)) < 0.1 and a1 > 0 and a2 < 0)
        
        # 上昇三角形: 上のトレンドラインがほぼ水平で、下のトレンドラインが上昇
        is_ascending = (abs(a2) < 0.05 and a1 > 0.05)
        
        # 下降三角形: 下のトレンドラインがほぼ水平で、上のトレンドラインが下降
        is_descending = (abs(a1) < 0.05 and a2 < -0.05)
        
        # 目標価格の計算
        target_prices = {
            "上方ブレイク": y_c + initial_height,
            "下方ブレイク": y_c - initial_height
        }
        
        # パターンの種類を判定
        pattern_info = {
            "パターン": "パターンなし",
            "収束予想日": convergence_date,
            "目標価格": target_prices
        }
        
        if is_symmetrical:
            pattern_info["パターン"] = "対称三角形"
            pattern_info["説明"] = "上下どちらのブレイクも同確率。ブレイク方向に大きな値動きの可能性。"
        elif is_ascending:
            pattern_info["パターン"] = "上昇三角形"
            pattern_info["説明"] = "上方ブレイクの可能性が高く、強気相場の継続を示唆。"
        elif is_descending:
            pattern_info["パターン"] = "下降三角形"
            pattern_info["説明"] = "下方ブレイクの可能性が高く、弱気相場の継続を示唆。"
        
        return pattern_info, high_coeffs, low_coeffs, high_dates, low_dates, target_prices
        
    except Exception as e:
        print(f"Error in detect_triangle_pattern: {str(e)}")
        return "パターンなし", None, None, None, None, None

# Streamlit UIの構築
st.title('📈 株価分析アプリ')  # アプリケーションのタイトル
stock_name = st.text_input('🔍 銘柄 (例: AAPL)', value='AAPL')  # 銘柄入力フィールド
start_date = st.date_input('📅 開始日', value=pd.Timestamp('2000-01-01'), min_value=pd.Timestamp('1900-01-01'), max_value=pd.Timestamp.today())  # 開始日選択
end_date = st.date_input('📅 終了日', value=pd.Timestamp.today(), min_value=pd.Timestamp('1900-01-01'), max_value=pd.Timestamp.today())  # 終了日選択

# 銘柄コードの入力チェック
if not stock_name:
    st.error('⚠️ 銘柄コードは必須です。')
    st.stop()

# 分析実行ボタンが押された時の処理
if st.button('📊 分析実行'):
    data = get_stock_data(stock_name, start_date, end_date)  # 株価データの取得
    data['RSI'] = compute_rsi(data['Close'])  # RSIの計算
    data['MACD'], data['Signal'] = compute_macd(data['Close'])  # MACDの計算
    
    # ボリンジャーバンドの計算
    data['Upper Band'], data['Lower Band'] = compute_bollinger_bands(data)

    # 未来のRSI予測値と日付の計算
    future_rsi = predict_future(data, 'RSI')  # RSIの予測
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=5)  # 予測日付の設定
    
    # 未来のMACD予測値と日付の計算
    future_macd = predict_future(data, 'MACD')  # MACDの予測
    future_macd_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=5)  # 予測日付の設定
    
    # グラフ表示用のサブプロット作成
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))  # 3つのグラフを縦に配置
    
    # ボリンジャーバンドのプロット
    ax1.plot(data['Close'], label='Close Price', color='black', linewidth=2, linestyle='-')  # 株価の推移
    ax1.plot(data['Upper Band'], label='Upper Band', color='orange', linewidth=2, linestyle='--', alpha=0.8)  # 上部バンド
    ax1.plot(data['Lower Band'], label='Lower Band', color='orange', linewidth=2, linestyle='--', alpha=0.8)  # 下部バンド
    ax1.set_title('Bollinger Bands')  # グラフタイトル
    ax1.set_xlabel('Date')  # X軸ラベル
    ax1.set_ylabel('Price')  # Y軸ラベル
    ax1.grid(True)  # グリッド表示
    ax1.legend()  # 凡例表示
    
    # RSIのプロット
    ax2.plot(data['RSI'], label='RSI', color='purple', linewidth=2, linestyle='-')  # RSIの推移
    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)', linewidth=2)  # 売られすぎライン
    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)', linewidth=2)  # 買われすぎライン
    ax2.set_title('RSI')  # グラフタイトル
    ax2.set_xlabel('Date')  # X軸ラベル
    ax2.set_ylabel('RSI Value')  # Y軸ラベル
    ax2.set_ylim(0, 100)  # Y軸の範囲設定
    ax2.grid(True)  # グリッド表示
    ax2.legend()  # 凡例表示
    
    # MACDのプロット
    ax3.plot(data['MACD'], label='MACD', color='blue', linewidth=2, linestyle='-')  # MACDライン
    ax3.plot(data['Signal'], label='Signal', color='red', linewidth=2, linestyle='--', alpha=0.8)  # シグナルライン
    ax3.set_title('MACD')  # グラフタイトル
    ax3.set_xlabel('Date')  # X軸ラベル
    ax3.set_ylabel('MACD Value')  # Y軸ラベル
    ax3.grid(True)  # グリッド表示
    ax3.legend()  # 凡例表示
    
    # 三角持ち合いパターンの検出と描画
    pattern_info, high_coeffs, low_coeffs, high_dates, low_dates, target_prices = detect_triangle_pattern(data)
    
    st.write('📐 三角持ち合いパターン分析:')
    if isinstance(pattern_info, dict):
        st.write(f'検出されたパターン: {pattern_info["パターン"]}')
        
        # パターンに応じた説明を表示
        if pattern_info["パターン"] == "対称三角形":
            st.write('説明: 上下どちらのブレイクも同確率。ブレイク方向に大きな値動きの可能性。')
        elif pattern_info["パターン"] == "上昇三角形":
            st.write('説明: 上方ブレイクの可能性が高く、強気相場の継続を示唆。')
        elif pattern_info["パターン"] == "下降三角形":
            st.write('説明: 下方ブレイクの可能性が高く、弱気相場の継続を示唆。')
        
        st.write(f'収束予想日: {pattern_info["収束予想日"].strftime("%Y-%m-%d")}')
        
        if target_prices:
            st.write('予想目標価格:')
            # numpy.float64をfloatに変換してからフォーマット
            upper_target = float(target_prices["上方ブレイク"])
            lower_target = float(target_prices["下方ブレイク"])
            st.write(f'上方ブレイク時: {upper_target:.2f}')
            st.write(f'下方ブレイク時: {lower_target:.2f}')
    else:
        st.write(f'検出されたパターン: {pattern_info}')
    
    # 三角持ち合いのプロット（パターンが検出された場合のみ）
    if all(v is not None for v in [high_coeffs, low_coeffs, high_dates, low_dates]):
        # 高値のトレンドライン
        high_x = np.arange(len(high_dates))
        high_trend = high_coeffs[0] * high_x + high_coeffs[1]
        
        # 安値のトレンドライン
        low_x = np.arange(len(low_dates))
        low_trend = low_coeffs[0] * low_x + low_coeffs[1]
        
        # トレンドラインをプロット（線を細く、透明度を調整）
        ax1.plot(high_dates, high_trend, 'r--', label='Upper trend line', linewidth=1, alpha=0.7)
        ax1.plot(low_dates, low_trend, 'g--', label='Lower trend line', linewidth=1, alpha=0.7)
        
        # ピークポイントをプロット
        ax1.scatter(high_dates, high_trend, color='red', s=50)
        ax1.scatter(low_dates, low_trend, color='green', s=50)
        
        ax1.legend()
    
    # グラフの表示
    st.pyplot(fig)
    
    # 予測結果のデータフレーム作成
    rsi_df = pd.DataFrame({'Date': future_dates.date, 'RSI': future_rsi})  # RSI予測値のデータフレーム
    macd_df = pd.DataFrame({'Date': future_macd_dates.date, 'MACD': future_macd})  # MACD予測値のデータフレーム
    
    # 予測結果の表示
    st.write('📈 未来の RSI 予測:')
    st.table(rsi_df)  # RSI予測値をテーブル形式で表示
    
    st.write('📉 未来の MACD 予測:')
    st.table(macd_df)  # MACD予測値をテーブル形式で表示