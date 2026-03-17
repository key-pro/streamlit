import sys

import pandas as pd
import streamlit as st
import yfinance as yf


# ─────────────────────────────────────────────────────────────
# FXデータ取得モジュール
# ※ yfinance は約15〜20分の遅延データを返します（リアルタイムではありません）
# ─────────────────────────────────────────────────────────────


def _safe_float(value) -> float:
    """pandas Series/Scalar を安全に float に変換する。"""
    if isinstance(value, pd.Series):
        if value.empty:
            raise ValueError("empty series")
        return float(value.iloc[0])
    return float(value)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance の戻り値を OHLCV の単純な DataFrame に正規化する。"""
    if df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close"]
    if not all(col in normalized.columns for col in required_cols):
        return pd.DataFrame()

    if "Volume" not in normalized.columns:
        normalized["Volume"] = 0

    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return normalized


def _download_ohlcv(ticker_symbol: str, period: str, interval: str) -> pd.DataFrame:
    """指定ティッカーのOHLCVを取得する。"""
    try:
        df = yf.download(tickers=ticker_symbol, period=period, interval=interval, progress=False)
    except Exception:
        return pd.DataFrame()
    return _normalize_ohlcv(df)


@st.cache_data(ttl=3600)
def fetch_fx_data(
    ticker_symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    show_warnings: bool = False,
    pair_name: str = None,
) -> pd.DataFrame:
    """
    Yahoo Finance から過去のFXデータを取得する（シンプル版）。
    - ティッカーシンボルに対応するレートをそのまま使用する
    - MultiIndex や欠損値だけを正規化し、逆数変換やクロス計算は行わない
    """
    try:
        df = _download_ohlcv(ticker_symbol, period, interval)
        if df.empty and show_warnings:
            st.warning(f"{ticker_symbol} のデータが見つかりません。")

        # デバッグ用ログ（任意の pair_name があれば併記）
        if not df.empty and pair_name:
            try:
                latest = df["Close"].iloc[-1]
                latest_rate = _safe_float(latest)
                print(
                    f"[データ取得] {pair_name}: ticker={ticker_symbol}, レート={latest_rate:.6f}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass

        return df

    except Exception as e:
        if show_warnings:
            st.error(f"{ticker_symbol} のデータ取得に失敗しました: {e}")
        return pd.DataFrame()


def get_available_pairs() -> dict:
    """
    対応通貨ペアと Yahoo Finance ティッカーの辞書を返す。
    主要通貨＋一部マイナー通貨のペアのみを自動生成。
    """
    import utils.pair_generator as pg
    return pg.generate_all_pairs()


def get_available_pairs_grouped() -> dict:
    """
    通貨ペアをカテゴリ別にグループ化した辞書を返す。
    フィルタリング機能などで使用する。
    """
    import utils.pair_generator as pg
    return pg.get_available_pairs_grouped()


def get_timeframes() -> dict:
    """
    時間足の表示名と yfinance の (period, interval) タプルのマッピングを返す。
    """
    return {
        "1日 (5分足)":    ("1d",  "5m"),
        "1週間 (15分足)": ("5d",  "15m"),
        "1ヶ月 (1時間足)": ("1mo", "1h"),
        "3ヶ月 (日足)":   ("3mo", "1d"),
        "6ヶ月 (日足)":   ("6mo", "1d"),
        "1年 (日足)":     ("1y",  "1d"),
        "5年 (週足)":     ("5y",  "1wk"),
    }
