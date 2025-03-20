import yfinance as yf  # Yahoo Financeから株価データを取得するためのライブラリ
import matplotlib.pyplot as plt # type: ignore  # グラフ描画用ライブラリ
import numpy as np  # 数値計算用ライブラリ
import pandas as pd  # データ操作・分析用ライブラリ
import streamlit as st # type: ignore  # WebアプリケーションフレームワークStreamlit
from sklearn.linear_model import LinearRegression # type: ignore  # 線形回帰モデル用ライブラリ
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import talib as ta  # 追加
import matplotlib.dates as mdates

# インポート文の後に追加
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォントの設定
# または
# plt.rcParams['font.family'] = 'IPAGothic'  # 別の日本語フォント

# ページ設定を最初に行う
st.set_page_config(layout="wide")

# テクニカル指標の説明文を定義
indicator_descriptions = {
    'SMA': '単純移動平均線（Simple Moving Average）は、指定期間の終値の単純平均を計算します。トレンドの方向性や支持/抵抗レベルを判断するのに使用されます。',
    'EMA': '指数平滑移動平均線（Exponential Moving Average）は、直近のデータにより重みを置いた移動平均線です。市場の変化により敏感に反応します。',
    'DEMA': '二重指数移動平均線（Double Exponential Moving Average）は、EMAを2回適用することでラグを減らした指標です。より早いトレンド転換のシグナルを提供します。',
    'WMA': '加重移動平均線（Weighted Moving Average）は、新しいデータにより大きな重みを付けて計算される移動平均線です。',
    'BBANDS': 'ボリンジャーバンド（Bollinger Bands）は、移動平均線を中心に標準偏差で上下のバンドを設定します。価格のボラティリティと相対的な価格水準を示します。',
    'SAR': 'パラボリックSAR（Parabolic Stop and Reverse）は、トレンドの転換点を特定し、潜在的な売買ポイントを示す指標です。',
    'MACD': 'MACD（Moving Average Convergence Divergence）は、2つの移動平均線の差を利用してトレンドの方向性とモメンタムを判断する指標です。',
    'RSI': '相対力指数（Relative Strength Index）は、価格の上昇・下降の強さを0-100の範囲で示すモメンタム指標です。',
    'ATR': '平均真実範囲（Average True Range）は、価格のボラティリティを測定する指標です。',
    'ADX': '平均方向性指数（Average Directional Index）は、トレンドの強さを測定する指標です。',
    'DI': '方向性指数（Directional Indicator）は、価格移動の方向性を示す指標です。+DIと-DIがあります。',
    'MOM': 'モメンタム（Momentum）は、現在の価格と一定期間前の価格の差を計算し、価格変動の勢いを測定します。',
    'STDDEV': '標準偏差（Standard Deviation）は、価格のボラティリティを統計的に測定する指標です。',
    'MAX/MIN': '指定期間における価格の最高値と最安値を示します。',
    'CCI': '商品チャネル指数（Commodity Channel Index）は、価格が平均的な動きからどれだけ乖離しているかを測定する指標です。'
}

# インポート文の直後に配置
if 'show_sma' not in st.session_state:
    st.session_state.show_sma = True
if 'show_ema' not in st.session_state:
    st.session_state.show_ema = False
if 'show_macd' not in st.session_state:
    st.session_state.show_macd = False
if 'show_rsi' not in st.session_state:
    st.session_state.show_rsi = True
if 'show_adx' not in st.session_state:
    st.session_state.show_adx = False
if 'show_bbands' not in st.session_state:
    st.session_state.show_bbands = False

# パラメータの初期値設定
if 'sma_period' not in st.session_state:
    st.session_state.sma_period = 20
if 'ema_period' not in st.session_state:
    st.session_state.ema_period = 20
if 'bb_period' not in st.session_state:
    st.session_state.bb_period = 20
if 'bb_std' not in st.session_state:
    st.session_state.bb_std = 2.0
if 'macd_fast' not in st.session_state:
    st.session_state.macd_fast = 12
if 'macd_slow' not in st.session_state:
    st.session_state.macd_slow = 26
if 'macd_signal' not in st.session_state:
    st.session_state.macd_signal = 9
if 'rsi_period' not in st.session_state:
    st.session_state.rsi_period = 14
if 'adx_period' not in st.session_state:
    st.session_state.adx_period = 14

# RSI(Relative Strength Index:相対力指数)を計算する関数
# RSIは、一定期間における値上がり幅の平均と値下がり幅の平均の比から算出される指標
def compute_rsi(data, period=14):
    """
    TA-Libを使用してRSIを計算
    """
    try:
        rsi = ta.RSI(data['Close'].values, timeperiod=period)
        return pd.Series(rsi, index=data.index)
    except Exception as e:
        print(f"RSI計算エラー: {str(e)}")
        return pd.Series(np.nan, index=data.index)


# MACD(Moving Average Convergence Divergence:移動平均収束拡散法)を計算する関数
# MACDは、短期と長期の指数移動平均線の差を表す指標
def compute_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    TA-Libを使用してMACDを計算
    """
    try:
        macd, signal, hist = ta.MACD(
            data['Close'].values,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        return pd.Series(macd, index=data.index), pd.Series(signal, index=data.index)
    except Exception as e:
        print(f"MACD計算エラー: {str(e)}")
        return pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)


# ボリンジャーバンドを計算する関数
# ボリンジャーバンドは、移動平均線を中心に標準偏差の幅で上下のバンドを設定する指標
def compute_bollinger_bands(data, period=20, nbdevup=2, nbdevdn=2):
    """
    TA-Libを使用してボリンジャーバンドを計算
    """
    try:
        upper, middle, lower = ta.BBANDS(
            data['Close'].values,
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=ta.MA_Type.SMA
        )
        return pd.Series(upper, index=data.index), pd.Series(lower, index=data.index)
    except Exception as e:
        print(f"ボリンジャーバンド計算エラー: {str(e)}")
        return pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)


# Yahoo Financeから株価データを取得する関数を修正
def get_stock_data(stock_name, start_date, end_date):
    """
    株価データを取得する関数
    """
    try:
        # ティッカーシンボルが空でないことを確認
        if not stock_name:
            return None
            
        # ティッカーオブジェクトの作成
        ticker = yf.Ticker(stock_name)
        
        # 銘柄の基本情報を確認
        info = ticker.info
        if not info:
            return None
            
        # データ取得
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval='1d'
        )
        
        if data.empty:
            return None
            
        return data
        
    except Exception as e:
        print(f"データ取得エラー: {str(e)}")
        return None


# RSIとMACDの将来値を予測する関数
def predict_future(data, column, days=5):
    try:
        # 欠損値を含まない列のみを取得
        valid_data = data[column].dropna()
        
        # データが空でないことを確認
        if len(valid_data) < 2:
            print(f"警告: {column}の有効なデータが不足しています")
            return np.array([np.nan] * days)
            
        X = np.arange(len(valid_data)).reshape(-1, 1)
        y = valid_data.values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(valid_data), len(valid_data) + days).reshape(-1, 1)
        future_y = model.predict(future_X)
        
        return future_y.flatten()
        
    except Exception as e:
        print(f"予測エラー ({column}): {str(e)}")
        return np.array([np.nan] * days)


# 三角持ち合いを検出する関数
# 株価チャートから三角持ち合いパターン(対称三角形、上昇三角形、下降三角形)を検出し、
# パターンの種類、収束予想日、目標価格などの情報を返す
def detect_triangle_pattern(data):
    try:
        # データの準備
        highs = data['High'].values
        lows = data['Low'].values
        dates = data.index
        
        # より単純なピーク検出方法に変更
        peaksH = []
        peaksL = []
        
        # 3日間の移動平均を使用して、ノイズを減らす
        highs_smooth = pd.Series(highs).rolling(window=3).mean()
        lows_smooth = pd.Series(lows).rolling(window=3).mean()
        
        # 単純な方法でピークを検出
        for i in range(1, len(data)-1):
            # 高値のピーク：前後より高ければピークとする
            if highs_smooth[i] > highs_smooth[i-1] and highs_smooth[i] > highs_smooth[i+1]:
                peaksH.append(i)
            
            # 安値のピーク：前後より低ければピークとする
            if lows_smooth[i] < lows_smooth[i-1] and lows_smooth[i] < lows_smooth[i+1]:
                peaksL.append(i)
        
        # 最低2つのピークがあれば分析を続行
        if len(peaksH) < 2 or len(peaksL) < 2:
            return "データ不足", None, None, None, None, None
        
        # 最新の10個のピークを使用
        peaksH = peaksH[-30:]
        peaksL = peaksL[-30:]
        
        # トレンドラインの計算を単純化
        high_x = np.array(peaksH)
        low_x = np.array(peaksL)
        high_y = highs[peaksH]
        low_y = lows[peaksL]
        
        # 直線フィッティング
        high_coeffs = np.polyfit(high_x, high_y, 1)
        low_coeffs = np.polyfit(low_x, low_y, 1)
        
        a1, b1 = low_coeffs   # 下側のトレンドライン
        a2, b2 = high_coeffs  # 上側のトレンドライン
        
        # パターン判定の条件を大幅に緩和
        is_symmetrical = abs(abs(a1) - abs(a2)) < 2.0  # かなり広い範囲で対称と判定
        is_ascending = (abs(a2) < 1.0 and a1 > 0.0)    # ほぼ水平でもOK
        is_descending = (abs(a1) < 1.0 and a2 < 0.0)   # ほぼ水平でもOK
        
        # 収束点の計算を単純化
        x_c = (b2 - b1) / (a1 - a2) if abs(a1 - a2) > 0.00001 else len(data)
        y_c = a1 * x_c + b1
        
        # 収束までの日数計算
        days_to_convergence = int(x_c - len(data) + 1)
        
        # 収束期間の制限を緩和（180日まで）
        if 0 < days_to_convergence < 180:
            convergence_date = dates[-1] + pd.Timedelta(days=days_to_convergence)
        else:
            convergence_date = dates[-1] + pd.Timedelta(days=30)  # デフォルトで30日後に設定
        
        # 目標価格の計算を単純化
        price_range = np.max(highs) - np.min(lows)
        target_prices = {
            "上方ブレイク": y_c + price_range * 0.3,  # より控えめな目標値
            "下方ブレイク": y_c - price_range * 0.3
        }
        
        # パターン情報の設定
        # パターンの種類、収束予想日、目標価格を辞書形式で格納
        pattern_info = {
            "パターン": "パターンなし",
            "収束予想日": convergence_date,
            "目標価格": target_prices
        }
        
        # パターンに応じた情報を設定
        if is_symmetrical:
            pattern_info["パターン"] = "対称三角形"
            pattern_info["説明"] = "上下どちらのブレイクも同確率。ブレイク方向に大きな値動きの可能性。"
        elif is_ascending:
            pattern_info["パターン"] = "上昇三角形"
            pattern_info["説明"] = "上方ブレイクの可能性が高く、強気相場の継続を示唆。"
        elif is_descending:
            pattern_info["パターン"] = "下降三角形"
            pattern_info["説明"] = "下方ブレイクの可能性が高く、弱気相場の継続を示唆。"
        
        # ピーク日付の設定
        high_dates = dates[peaksH]  # 高値のピーク日付
        low_dates = dates[peaksL]   # 安値のピーク日付
        
        # 分析結果を返す
        return pattern_info, high_coeffs, low_coeffs, high_dates, low_dates, target_prices
        
    except Exception as e:
        # エラー発生時はエラーメッセージを出力し、デフォルト値を返す
        print(f"Error in detect_triangle_pattern: {str(e)}")
        return "パターンなし", None, None, None, None, None


def detect_patterns(data):
    try:
        # データの準備
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        dates = data.index
        
        # TA-Libでパターン検出
        # ヘッドアンドショルダー（上向き）
        h_and_s = ta.CDLHSPATTERN(open=data['Open'].values, 
                                 high=high,
                                 low=low, 
                                 close=close)
        
        # 逆ヘッドアンドショルダー（下向き）
        inv_h_and_s = ta.CDLINVERTEDHSPATTERN(open=data['Open'].values,
                                             high=high,
                                             low=low,
                                             close=close)
        
        # パターンが検出された位置を特定
        h_and_s_idx = np.where(h_and_s != 0)[0]
        inv_h_and_s_idx = np.where(inv_h_and_s != 0)[0]
        
        patterns = []
        
        # 通常のヘッドアンドショルダーパターン
        for idx in h_and_s_idx:
            if idx >= 5:  # パターンの開始位置を確保
                pattern_start = idx - 5
                pattern_end = idx
                
                pattern_height = np.max(high[pattern_start:pattern_end+1]) - np.min(low[pattern_start:pattern_end+1])
                neckline = np.min(low[pattern_start:pattern_end+1])
                target_price = neckline - pattern_height
                
                patterns.append({
                    "パターン": "ヘッドアンドショルダー（天井）",
                    "説明": "天井圏形成を示す反転パターン。ネックラインを下抜けると下落トレンドの開始を示唆。",
                    "検出日": dates[idx],
                    "ネックライン": neckline,
                    "目標価格": target_price,
                    "信頼度": "高" if h_and_s[idx] > 0 else "中"
                })
        
        # 逆ヘッドアンドショルダーパターン
        for idx in inv_h_and_s_idx:
            if idx >= 5:
                pattern_start = idx - 5
                pattern_end = idx
                
                pattern_height = np.max(high[pattern_start:pattern_end+1]) - np.min(low[pattern_start:pattern_end+1])
                neckline = np.max(high[pattern_start:pattern_end+1])
                target_price = neckline + pattern_height
                
                patterns.append({
                    "パターン": "逆ヘッドアンドショルダー（底）",
                    "説明": "底値圏形成を示す反転パターン。ネックラインを上抜けると上昇トレンドの開始を示唆。",
                    "検出日": dates[idx],
                    "ネックライン": neckline,
                    "目標価格": target_price,
                    "信頼度": "高" if inv_h_and_s[idx] > 0 else "中"
                })
        
        return patterns
        
    except Exception as e:
        print(f"Error in detect_patterns: {str(e)}")
        return []


# カスタムCSSを修正
st.markdown("""
<style>
    /* 全体のテーマカラー */
    :root {
        --primary: #1E3D59;
        --accent: #17A2B8;
        --success: #28A745;
        --danger: #DC3545;
        --dark: #343A40;
        --light: #F8F9FA;
    }

    /* メインヘッダー */
    .main-header {
        font-size: 2.2rem;
        color: white;
        text-align: left;
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, #1E3D59 0%, #17A2B8 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* メトリックカード */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #17A2B8;
        margin: 0.5rem 1rem;  /* 上下左右のマージンを追加 */
        min-width: 130px;  /* 最小幅を少し広げる */
        height: 85px;  /* 高さを固定 */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-label {
        color: #6c757d;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;  /* ラベルと値の間隔を調整 */
    }

    .metric-value {
        color: #343a40;
        font-size: 1.4rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* インジケーター */
    .indicator-positive {
        color: #28A745;
        font-size: 1.4rem;  /* 値のサイズに合わせる */
    }

    .indicator-negative {
        color: #DC3545;
        font-size: 1.4rem;  /* 値のサイズに合わせる */
    }

    /* チャートコンテナ */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 2rem;
    }

    /* サイドバーカスタマイズ */
    .sidebar-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* 検索ボックス */
    .search-box {
        background: white;
        padding: 2rem;  /* パディングを増やす */
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;  /* 下部の余白を追加 */
    }

    /* ボタン */
    .stButton>button {
        background: linear-gradient(135deg, #17A2B8 0%, #1E3D59 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        width: 100%;
    }

    /* データテーブル */
    .dataframe {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# メインコンテンツエリアの配置の比率を修正
main_col1, main_col2 = st.columns([4, 1])  # 比率を4:1に変更（現在は2:1）

# メトリクスとグラフのプレースホルダーを作成
with main_col1:
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()  # グラフ用のプレースホルダーを追加

# 検索パネル
with main_col2:
    st.markdown("""
    <div class="search-box">
        <h4 style='
            color: #1E3D59; 
            margin: 1.5rem;  /* 上下左右の余白を1.5remに */
            padding: 1rem;   /* 内側の余白を1remに */
            background-color: #f8f9fa;  /* 背景色を追加 */
            border-radius: 8px;         /* 角を丸く */
            text-align: center;         /* テキストを中央揃え */
        '>🔍 銘柄検索</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # 入力フォーム
    with st.container():
        stock_name = st.text_input('銘柄コード', value='AAPL', help='例: AAPL, GOOGL, MSFT').strip().upper()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                '開始日',
                value=pd.Timestamp('2000-01-01'),
                min_value=pd.Timestamp('1990-01-01'),
                max_value=pd.Timestamp.today()
            )
        with col2:
            end_date = st.date_input(
                '終了日',
                value=pd.Timestamp.today(),
                min_value=pd.Timestamp('1990-01-01'),
                max_value=pd.Timestamp.today()
            )
        
        if st.button('分析開始', use_container_width=True):
            try:
                # 入力値の検証
                if not stock_name:
                    st.error('銘柄コードを入力してください。')
                    st.stop()
                
                if start_date >= end_date:
                    st.error('開始日は終了日より前の日付を選択してください。')
                    st.stop()
                
                # ローディング表示
                with st.spinner(f'{stock_name}のデータを取得中...'):
                    # データ取得を試行
                    data = get_stock_data(stock_name, start_date, end_date)
                    
                    if data is None or data.empty:
                        st.error(f'銘柄コード "{stock_name}" のデータを取得できませんでした。\n'
                                '以下を確認してください：\n'
                                '- 銘柄コードが正しいか\n'
                                '- インターネット接続が安定しているか\n'
                                '- 指定した期間にデータが存在するか')
                        st.stop()
                    
                    # データ取得成功のメッセージ
                    st.success(f'{stock_name}のデータを取得しました！')
                    
                    # 最新のデータを取得
                    latest_data = data.iloc[-1]
                    previous_data = data.iloc[-2]
                    
                    # メトリクスの更新
                    with metrics_placeholder:
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                        with metrics_col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">現在値</div>
                                <div class="metric-value">${latest_data['Close']:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 前日比の計算と表示
                        price_change = ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100
                        price_change_class = 'indicator-positive' if price_change >= 0 else 'indicator-negative'
                        price_change_sign = '+' if price_change >= 0 else ''
                        
                        with metrics_col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">前日比</div>
                                <div class="metric-value {price_change_class}">{price_change_sign}{price_change:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 出来高の表示
                        volume = latest_data['Volume']
                        volume_display = f"{volume/1000000:.1f}M" if volume >= 1000000 else f"{volume/1000:.1f}K"
                        
                        with metrics_col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Price</div>
                                <div class="metric-value">{volume_display}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # RSIの計算と表示
                        rsi = compute_rsi(data).iloc[-1]
                        
                        with metrics_col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">RSI</div>
                                <div class="metric-value">{rsi:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # グラフの描画
                with chart_placeholder:
                    try:
                        # 表示するグラフの数を計算
                        active_indicators = []
                        if st.session_state.show_macd: active_indicators.append('MACD')
                        if st.session_state.show_rsi: active_indicators.append('RSI')
                        if st.session_state.show_adx: active_indicators.append('ADX')
                        
                        # サブプロットの数を設定
                        n_plots = 1 + len(active_indicators)
                        
                        # フィギュアサイズを設定
                        fig = plt.figure(figsize=(72, 24 + 12 * len(active_indicators)))
                        
                        # グリッドのスペースを調整（余白を大きく）
                        spec = fig.add_gridspec(
                            n_plots, 
                            1, 
                            height_ratios=[3] + [1] * len(active_indicators),
                            hspace=0.4,  # サブプロット間の縦方向の間隔
                        )
                        
                        # 余白を設定
                        plt.subplots_adjust(
                            left=0.1,    # 左余白を10%に
                            right=0.95,  # 右余白を5%に
                            top=0.95,    # 上余白を5%に
                            bottom=0.1   # 下余白を10%に
                        )
                        
                        # メインチャート（株価）
                        ax_main = fig.add_subplot(spec[0])
                        ax_main.plot(data.index, data['Close'], label='StockPrice', color='black', alpha=0.8, linewidth=4)
                        
                        # タイトルの位置を調整（上方向に余白を追加）
                        ax_main.set_title(f'{stock_name}', 
                                         fontsize=72, 
                                         pad=50,  # タイトルとグラフの間隔を増やす
                                         y=1.02   # タイトルの縦位置を少し上に
                        )
                        
                        # 移動平均線の追加
                        if st.session_state.show_sma:
                            sma = ta.SMA(data['Close'].values, timeperiod=st.session_state.sma_period)
                            ax_main.plot(data.index, sma, label=f'SMA({st.session_state.sma_period})', alpha=0.7)
                        
                        if st.session_state.show_ema:
                            ema = ta.EMA(data['Close'].values, timeperiod=st.session_state.ema_period)
                            ax_main.plot(data.index, ema, label=f'EMA({st.session_state.ema_period})', alpha=0.7)
                        
                        if st.session_state.show_bbands:
                            upper, middle, lower = ta.BBANDS(data['Close'].values, 
                                                           timeperiod=st.session_state.bb_period,
                                                           nbdevup=st.session_state.bb_std,
                                                           nbdevdn=st.session_state.bb_std)
                            ax_main.fill_between(data.index, upper, lower, alpha=0.1, color='gray')
                            ax_main.plot(data.index, upper, '--', color='gray', alpha=0.7, label='BB Upper')
                            ax_main.plot(data.index, lower, '--', color='gray', alpha=0.7, label='BB Lower')
                        
                        # インジケーターの追加
                        current_plot = 1
                        
                        # MACD
                        if st.session_state.show_macd:
                            ax_macd = fig.add_subplot(spec[current_plot], sharex=ax_main)
                            macd, signal, hist = ta.MACD(data['Close'].values,
                                                        fastperiod=st.session_state.macd_fast,
                                                        slowperiod=st.session_state.macd_slow,
                                                        signalperiod=st.session_state.macd_signal)
                            ax_macd.plot(data.index, macd, label='MACD', color='blue')
                            ax_macd.plot(data.index, signal, label='Signal', color='red')
                            ax_macd.bar(data.index, hist, label='Histogram', color='gray', alpha=0.3)
                            ax_macd.grid(True, alpha=0.3)
                            ax_macd.legend(loc='upper left', fontsize=48)  # 凡例を48ptに
                            ax_macd.set_title('MACD', fontsize=56)  # タイトルを56ptに
                            ax_macd.tick_params(axis='both', labelsize=48)  # 軸ラベルを48ptに
                            current_plot += 1
                        
                        # RSI
                        if st.session_state.show_rsi:
                            ax_rsi = fig.add_subplot(spec[current_plot], sharex=ax_main)
                            rsi = ta.RSI(data['Close'].values, timeperiod=st.session_state.rsi_period)
                            ax_rsi.plot(data.index, rsi, label='RSI', color='purple')
                            ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.3)
                            ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.3)
                            ax_rsi.set_ylim(0, 100)
                            ax_rsi.grid(True, alpha=0.3)
                            ax_rsi.legend(loc='upper left', fontsize=48)  # 凡例を48ptに
                            ax_rsi.set_title('RSI', fontsize=56)  # タイトルを56ptに
                            ax_rsi.tick_params(axis='both', labelsize=48)  # 軸ラベルを48ptに
                            current_plot += 1
                        
                        # ADX
                        if st.session_state.show_adx:
                            ax_adx = fig.add_subplot(spec[current_plot], sharex=ax_main)
                            adx = ta.ADX(data['High'].values, data['Low'].values, data['Close'].values, 
                                        timeperiod=st.session_state.adx_period)
                            plus_di = ta.PLUS_DI(data['High'].values, data['Low'].values, data['Close'].values, 
                                                timeperiod=st.session_state.adx_period)
                            minus_di = ta.MINUS_DI(data['High'].values, data['Low'].values, data['Close'].values, 
                                                 timeperiod=st.session_state.adx_period)
                            ax_adx.plot(data.index, adx, label='ADX', color='black')
                            ax_adx.plot(data.index, plus_di, label='+DI', color='green')
                            ax_adx.plot(data.index, minus_di, label='-DI', color='red')
                            ax_adx.grid(True, alpha=0.3)
                            ax_adx.legend(loc='upper left', fontsize=48)  # 凡例を48ptに
                            ax_adx.set_title('ADX/DI', fontsize=56)  # タイトルを56ptに
                            ax_adx.tick_params(axis='both', labelsize=48)  # 軸ラベルを48ptに
                        
                        # 全てのサブプロットに共通の設定
                        for ax in fig.axes:
                            ax.grid(True, alpha=0.3, linestyle='--', linewidth=3)  # グリッド線も太く
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=48)  # 日付ラベルを48ptに
                            ax.tick_params(axis='both', labelsize=48)  # 全ての軸ラベルを48ptに
                        
                        # Y軸のラベル間隔を調整（ラベルが重ならないように）
                        for ax in fig.axes:
                            ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # Y軸の目盛り数を制限
                        
                        # レイアウトの自動調整の前に余白を確保
                        plt.tight_layout(pad=3.0)  # 全体の余白を増やす
                        
                        # グラフの表示
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f'グラフの描画中にエラーが発生しました: {str(e)}')

            except Exception as e:
                st.error(f'予期せぬエラーが発生しました: {str(e)}')
                st.stop()

# セッションステートの初期化を拡張
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.update({
        # 移動平均線系
        'show_sma': True,
        'show_ema': False,
        'show_dema': False,
        'show_wma': False,
        'show_bbands': False,
        'show_sar': False,
        
        # モメンタム系
        'show_macd': False,
        'show_rsi': True,
        'show_mom': False,
        'show_cci': False,
        
        # トレンド系
        'show_adx': False,
        'show_di': False,
        
        # ボラティリティ系
        'show_atr': False,
        'show_stddev': False,
        'show_maxmin': False,
        
        # パラメータ設定
        'sma_period': 20,
        'ema_period': 20,
        'dema_period': 20,
        'wma_period': 20,
        'bb_period': 20,
        'bb_std': 2.0,
        'sar_acceleration': 0.02,
        'sar_maximum': 0.2,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'mom_period': 10,
        'cci_period': 20,
        'adx_period': 14,
        'atr_period': 14,
        'stddev_period': 20
    })

# サイドバーの設定
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 テクニカル指標設定</div>', unsafe_allow_html=True)
    
    # 移動平均線系
    st.markdown('<div class="sidebar-subheader">移動平均線系指標</div>', unsafe_allow_html=True)
    
    for indicator in ['SMA', 'EMA', 'DEMA', 'WMA', 'BBANDS', 'SAR']:
        col1, col2 = st.columns([3, 1])
        with col1:
            show_key = f'show_{indicator.lower()}'
            st.session_state[show_key] = st.checkbox(
                indicator, 
                value=st.session_state[show_key],
                key=f'{indicator.lower()}_check'
            )
        with col2:
            st.button('ℹ️', help=indicator_descriptions[indicator], key=f'{indicator.lower()}_help')
    
    # モメンタム系
    st.markdown('<div class="sidebar-subheader">モメンタム系指標</div>', unsafe_allow_html=True)
    
    for indicator in ['MACD', 'RSI', 'MOM', 'CCI']:
        col1, col2 = st.columns([3, 1])
        with col1:
            show_key = f'show_{indicator.lower()}'
            st.session_state[show_key] = st.checkbox(
                indicator, 
                value=st.session_state[show_key],
                key=f'{indicator.lower()}_check'
            )
        with col2:
            st.button('ℹ️', help=indicator_descriptions[indicator], key=f'{indicator.lower()}_help')
    
    # トレンド系
    st.markdown('<div class="sidebar-subheader">トレンド系指標</div>', unsafe_allow_html=True)
    
    for indicator in ['ADX', 'DI']:
        col1, col2 = st.columns([3, 1])
        with col1:
            show_key = f'show_{indicator.lower()}'
            st.session_state[show_key] = st.checkbox(
                indicator, 
                value=st.session_state[show_key],
                key=f'{indicator.lower()}_check'
            )
        with col2:
            st.button('ℹ️', help=indicator_descriptions[indicator], key=f'{indicator.lower()}_help')
    
    # ボラティリティ系
    st.markdown('<div class="sidebar-subheader">ボラティリティ系指標</div>', unsafe_allow_html=True)
    
    for indicator in ['ATR', 'STDDEV', 'MAX/MIN']:
        col1, col2 = st.columns([3, 1])
        with col1:
            show_key = f'show_{indicator.lower().replace("/", "")}'
            st.session_state[show_key] = st.checkbox(
                indicator, 
                value=st.session_state[show_key],
                key=f'{indicator.lower().replace("/", "")}_check'
            )
        with col2:
            st.button('ℹ️', help=indicator_descriptions[indicator], key=f'{indicator.lower().replace("/", "")}_help')

    # パラメータ設定
    with st.expander('📊 パラメータ設定'):
        # 移動平均線系のパラメータ
        if any([st.session_state.show_sma, st.session_state.show_ema, 
                st.session_state.show_dema, st.session_state.show_wma]):
            st.markdown('##### 移動平均線パラメータ')
            if st.session_state.show_sma:
                st.session_state.sma_period = st.slider('SMA期間', 5, 200, st.session_state.sma_period, key='sma_period_slider')
            if st.session_state.show_ema:
                st.session_state.ema_period = st.slider('EMA期間', 5, 200, st.session_state.ema_period, key='ema_period_slider')
            if st.session_state.show_dema:
                st.session_state.dema_period = st.slider('DEMA期間', 5, 200, st.session_state.dema_period, key='dema_period_slider')
            if st.session_state.show_wma:
                st.session_state.wma_period = st.slider('WMA期間', 5, 200, st.session_state.wma_period, key='wma_period_slider')

        # ボリンジャーバンドのパラメータ
        if st.session_state.show_bbands:
            st.markdown('##### ボリンジャーバンドパラメータ')
            st.session_state.bb_period = st.slider('期間', 5, 200, st.session_state.bb_period, key='bb_period_slider')
            st.session_state.bb_std = st.slider('標準偏差', 1.0, 3.0, st.session_state.bb_std, 0.1, key='bb_std_slider')

        # パラボリックSARのパラメータ
        if st.session_state.show_sar:
            st.markdown('##### パラボリックSARパラメータ')
            st.session_state.sar_acceleration = st.slider('加速係数', 0.01, 0.2, st.session_state.sar_acceleration, 0.01, key='sar_acceleration_slider')
            st.session_state.sar_maximum = st.slider('最大値', 0.1, 0.5, st.session_state.sar_maximum, 0.1, key='sar_maximum_slider')

        # MACDのパラメータ
        if st.session_state.show_macd:
            st.markdown('##### MACDパラメータ')
            st.session_state.macd_fast = st.slider('短期期間', 5, 50, st.session_state.macd_fast, key='macd_fast_slider')
            st.session_state.macd_slow = st.slider('長期期間', 10, 100, st.session_state.macd_slow, key='macd_slow_slider')
            st.session_state.macd_signal = st.slider('シグナル期間', 5, 50, st.session_state.macd_signal, key='macd_signal_slider')

        # その他のインジケーターのパラメータ
        for indicator, period_key, label in [
            ('RSI', 'rsi_period', 'RSI期間'),
            ('MOM', 'mom_period', 'モメンタム期間'),
            ('CCI', 'cci_period', 'CCI期間'),
            ('ADX', 'adx_period', 'ADX期間'),
            ('ATR', 'atr_period', 'ATR期間'),
            ('STDDEV', 'stddev_period', '標準偏差期間')
        ]:
            if st.session_state[f'show_{indicator.lower()}']:
                st.markdown(f'##### {indicator}パラメータ')
                st.session_state[period_key] = st.slider(
                    label, 5, 50, st.session_state[period_key], 
                    key=f'{indicator.lower()}_period_slider'
                )

# StockPrice表示部分の修正
if 'df' in st.session_state and not st.session_state.df.empty:
    metrics_placeholder = st.empty()
    with metrics_placeholder.container():
        try:
            latest_data = st.session_state.df.iloc[-1]
            prev_data = st.session_state.df.iloc[-2]
            
            # 現在値
            current_price = latest_data['Close']
            
            # 前日比の計算
            price_change = current_price - prev_data['Close']
            price_change_percent = (price_change / prev_data['Close']) * 100
            
            # 出来高
            volume = latest_data['Volume']
            
            # RSIの計算（既に計算済みの場合）
            rsi = latest_data.get('RSI', None)
            
            # メトリクスの表示
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("現在値", f"¥{current_price:,.0f}")
            
            with col2:
                st.metric("前日比", 
                         f"¥{price_change:,.0f} ({price_change_percent:.2f}%)",
                         delta_color="normal" if price_change >= 0 else "inverse")
            
            with col3:
                st.metric("出来高", f"{volume:,.0f}")
            
            with col4:
                if rsi is not None:
                    st.metric("RSI", f"{rsi:.1f}")
                
        except Exception as e:
            st.error(f"メトリクスの表示中にエラーが発生しました: {str(e)}")

# セッションステートの初期化（ファイルの先頭付近に配置）
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if 'stock_price_displayed' not in st.session_state:
    st.session_state.stock_price_displayed = False