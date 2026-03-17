import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# ===========================
# トレンド指標
# ===========================

def add_sma(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """DataFrameに単純移動平均を追加する。"""
    indicator = SMAIndicator(close=df[column], window=window, fillna=True)
    df[f'SMA_{window}'] = indicator.sma_indicator()
    return df

def add_ema(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """DataFrameに指数移動平均を追加する。"""
    indicator = EMAIndicator(close=df[column], window=window, fillna=True)
    df[f'EMA_{window}'] = indicator.ema_indicator()
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, window_dev: int = 2, column: str = "Close") -> pd.DataFrame:
    """DataFrameにボリンジャーバンド（上限、下限、中央）を追加する。"""
    indicator = BollingerBands(close=df[column], window=window, window_dev=window_dev, fillna=True)
    df[f'BB_High_{window}'] = indicator.bollinger_hband()
    df[f'BB_Low_{window}'] = indicator.bollinger_lband()
    df[f'BB_Mid_{window}'] = indicator.bollinger_mavg()
    return df

def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Ichimoku Kinko Hyo (一目均衡表) to DataFrame.
    - Tenkan-sen (転換線): 9-period midpoint
    - Kijun-sen (基準線): 26-period midpoint
    - Senkou Span A (先行スパンA): Average of Tenkan/Kijun, shifted 26 forward
    - Senkou Span B (先行スパンB): 52-period midpoint, shifted 26 forward
    - Chikou Span (遅行スパン): Close shifted 26 back
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # 転換線（Tenkan-sen）: 9期間
    tenkan_high = high.rolling(9).max()
    tenkan_low = low.rolling(9).min()
    df['Ichimoku_Tenkan'] = (tenkan_high + tenkan_low) / 2

    # 基準線（Kijun-sen）: 26期間
    kijun_high = high.rolling(26).max()
    kijun_low = low.rolling(26).min()
    df['Ichimoku_Kijun'] = (kijun_high + kijun_low) / 2

    # 先行スパンA（Senkou Span A）: 転換線と基準線の平均を26期間先にシフト
    df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)

    # 先行スパンB（Senkou Span B）: 52期間の中点を26期間先にシフト
    span_b_high = high.rolling(52).max()
    span_b_low = low.rolling(52).min()
    df['Ichimoku_SpanB'] = ((span_b_high + span_b_low) / 2).shift(26)

    # 遅行スパン（Chikou Span）: 終値を26期間後ろにシフト
    df['Ichimoku_Chikou'] = close.shift(-26)

    return df

def add_dmi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """DMI（方向性指数）とADXを追加する。"""
    indicator = ADXIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], window=window, fillna=True
    )
    df['ADX'] = indicator.adx()
    df['DI_Plus'] = indicator.adx_pos()
    df['DI_Minus'] = indicator.adx_neg()
    return df

def add_parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    """
    DataFrameにパラボリックSARを追加する（手動実装）。
    追加する列: 'PSAR', 'PSAR_bull'（強気シグナル）, 'PSAR_bear'（弱気シグナル）
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(close)

    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 = 強気, -1 = 弱気
    ep = np.zeros(n)     # 極値ポイント
    af = np.zeros(n)     # 加速係数

    # 初期値
    trend[0] = 1
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = step

    for i in range(1, n):
        prev_trend = trend[i - 1]
        prev_sar = sar[i - 1]
        prev_ep = ep[i - 1]
        prev_af = af[i - 1]

        if prev_trend == 1:  # 強気
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            new_sar = min(new_sar, low[i - 1], low[max(0, i - 2)])
            if low[i] < new_sar:
                # 弱気への転換
                trend[i] = -1
                sar[i] = prev_ep
                ep[i] = low[i]
                af[i] = step
            else:
                trend[i] = 1
                sar[i] = new_sar
                if high[i] > prev_ep:
                    ep[i] = high[i]
                    af[i] = min(prev_af + step, max_step)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:  # 弱気
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            new_sar = max(new_sar, high[i - 1], high[max(0, i - 2)])
            if high[i] > new_sar:
                # 強気への転換
                trend[i] = 1
                sar[i] = prev_ep
                ep[i] = high[i]
                af[i] = step
            else:
                trend[i] = -1
                sar[i] = new_sar
                if low[i] < prev_ep:
                    ep[i] = low[i]
                    af[i] = min(prev_af + step, max_step)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af

    df['PSAR'] = sar
    df['PSAR_Bull'] = np.where(trend == 1, sar, np.nan)
    df['PSAR_Bear'] = np.where(trend == -1, sar, np.nan)
    return df

def add_envelope(df: pd.DataFrame, window: int = 20, pct: float = 2.0, column: str = "Close") -> pd.DataFrame:
    """
    エンベロープ指標（移動平均 ± pct%）を追加する。
    """
    sma = df[column].rolling(window).mean()
    df[f'ENV_Upper_{window}'] = sma * (1 + pct / 100)
    df[f'ENV_Lower_{window}'] = sma * (1 - pct / 100)
    df[f'ENV_Mid_{window}'] = sma
    return df


# ===========================
# オシレーター指標
# ===========================

def add_rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.DataFrame:
    """DataFrameに相対力指数（RSI）を追加する。"""
    indicator = RSIIndicator(close=df[column], window=window, fillna=True)
    df[f'RSI_{window}'] = indicator.rsi()
    return df

def add_macd(df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9, column: str = "Close") -> pd.DataFrame:
    """DataFrameにMACDライン、シグナル、ヒストグラムを追加する。"""
    indicator = MACD(close=df[column], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=True)
    df['MACD'] = indicator.macd()
    df['MACD_Signal'] = indicator.macd_signal()
    df['MACD_Hist'] = indicator.macd_diff()
    return df

def add_stochastics(df: pd.DataFrame, window: int = 14, smooth_window: int = 3) -> pd.DataFrame:
    """ストキャスティクスオシレーター（%Kと%D）を追加する。"""
    indicator = StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=window, smooth_window=smooth_window, fillna=True
    )
    df['Stoch_K'] = indicator.stoch()
    df['Stoch_D'] = indicator.stoch_signal()
    return df

def add_psychological_line(df: pd.DataFrame, window: int = 12, column: str = "Close") -> pd.DataFrame:
    """
    サイコロジカルラインを追加する。
    = (期間内の上昇日数 / 期間) × 100
    """
    rising = (df[column].diff() > 0).astype(int)
    df[f'PsychLine_{window}'] = rising.rolling(window).mean() * 100
    return df

def add_rci(df: pd.DataFrame, window: int = 9, column: str = "Close") -> pd.DataFrame:
    """
    RCI（順位相関指数）を追加する。
    日付の順位と価格の順位の相関を測定する。
    範囲: -100 から +100。
    """
    def _rci(s):
        n = len(s)
        date_ranks = np.arange(1, n + 1)
        price_ranks = s.rank(ascending=False).values
        d_sq = ((date_ranks - price_ranks) ** 2).sum()
        return (1 - 6 * d_sq / (n * (n**2 - 1))) * 100

    df[f'RCI_{window}'] = df[column].rolling(window).apply(_rci, raw=False)
    return df

def add_ma_deviation(df: pd.DataFrame, window: int = 25, column: str = "Close") -> pd.DataFrame:
    """
    移動平均乖離率を追加する。
    = (終値 - 移動平均) / 移動平均 × 100 (%)
    """
    ma = df[column].rolling(window).mean()
    df[f'MA_Dev_{window}'] = (df[column] - ma) / ma * 100
    return df


# ===========================
# その他の分析
# ===========================

def add_historical_volatility(df: pd.DataFrame, window: int = 21, column: str = "Close") -> pd.DataFrame:
    """
    年率換算ヒストリカルボラティリティを追加する。
    = 対数リターンの標準偏差 × sqrt(252) × 100 (%)
    """
    log_returns = np.log(df[column] / df[column].shift(1))
    df[f'HV_{window}'] = log_returns.rolling(window).std() * np.sqrt(252) * 100
    return df

def add_fibonacci(df: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    """
    最近の高値/安値に基づいてフィボナッチリトレースメントレベルを追加する。
    レベル（23.6%, 38.2%, 50%, 61.8%, 78.6%）の定数列を追加する。
    """
    recent = df.tail(lookback)
    high = recent['High'].max()
    low = recent['Low'].min()
    diff = high - low

    df['Fib_0'] = low
    df['Fib_236'] = low + diff * 0.236
    df['Fib_382'] = low + diff * 0.382
    df['Fib_500'] = low + diff * 0.500
    df['Fib_618'] = low + diff * 0.618
    df['Fib_786'] = low + diff * 0.786
    df['Fib_100'] = high
    return df
