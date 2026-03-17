import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from utils.data_fetcher import fetch_fx_data, get_timeframes
from components.charts import render_google_candlestick_chart
from utils.indicators import (
    add_sma,
    add_ema,
    add_bollinger_bands,
    add_macd,
    add_rsi,
    add_ichimoku,
    add_dmi,
    add_parabolic_sar,
    add_envelope,
    add_stochastics,
    add_psychological_line,
    add_rci,
    add_ma_deviation,
    add_historical_volatility,
    add_fibonacci,
)


st.set_page_config(
    page_title="米国株マーケットダッシュボード",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 米国株の代表的な銘柄リスト（セクター別）
US_STOCKS = {
    # ── インデックス・ETF ──
    "SPY (S&P500 ETF)": "SPY",
    "QQQ (NASDAQ100 ETF)": "QQQ",
    "DIA (Dow30 ETF)": "DIA",
    "IWM (Russell2000 ETF)": "IWM",
    "VOO (Vanguard S&P500)": "VOO",
    "VTI (Vanguard Total Market)": "VTI",
    "ARKK (ARK Innovation)": "ARKK",
    "XLF (金融セクター)": "XLF",
    "XLK (テクノロジー)": "XLK",
    "XLE (エネルギー)": "XLE",
    "XLV (ヘルスケア)": "XLV",
    "XLY (消費者裁量)": "XLY",
    # ── メガテック・ハイテク ──
    "AAPL (Apple)": "AAPL",
    "MSFT (Microsoft)": "MSFT",
    "GOOGL (Alphabet C)": "GOOGL",
    "GOOG (Alphabet A)": "GOOG",
    "AMZN (Amazon)": "AMZN",
    "META (Meta Platforms)": "META",
    "TSLA (Tesla)": "TSLA",
    "NVDA (NVIDIA)": "NVDA",
    "AMD (AMD)": "AMD",
    "INTC (Intel)": "INTC",
    "QCOM (Qualcomm)": "QCOM",
    "AVGO (Broadcom)": "AVGO",
    "CRM (Salesforce)": "CRM",
    "ORCL (Oracle)": "ORCL",
    "ADBE (Adobe)": "ADBE",
    "NFLX (Netflix)": "NFLX",
    "UBER (Uber)": "UBER",
    "PYPL (PayPal)": "PYPL",
    "SHOP (Shopify)": "SHOP",
    "SQ (Block)": "SQ",
    "SNOW (Snowflake)": "SNOW",
    "PLTR (Palantir)": "PLTR",
    "COIN (Coinbase)": "COIN",
    "MU (Micron)": "MU",
    "AMAT (Applied Materials)": "AMAT",
    "ASML (ASML)": "ASML",
    "IBM (IBM)": "IBM",
    "CSCO (Cisco)": "CSCO",
    # ── 金融 ──
    "BRK.B (Berkshire Hathaway)": "BRK-B",
    "JPM (JPMorgan Chase)": "JPM",
    "V (Visa)": "V",
    "MA (Mastercard)": "MA",
    "BAC (Bank of America)": "BAC",
    "WFC (Wells Fargo)": "WFC",
    "GS (Goldman Sachs)": "GS",
    "MS (Morgan Stanley)": "MS",
    "C (Citigroup)": "C",
    "AXP (American Express)": "AXP",
    "BLK (BlackRock)": "BLK",
    "SCHW (Charles Schwab)": "SCHW",
    # ── ヘルスケア・医薬 ──
    "UNH (UnitedHealth)": "UNH",
    "JNJ (Johnson & Johnson)": "JNJ",
    "PFE (Pfizer)": "PFE",
    "ABBV (AbbVie)": "ABBV",
    "MRK (Merck)": "MRK",
    "LLY (Eli Lilly)": "LLY",
    "TMO (Thermo Fisher)": "TMO",
    "ABT (Abbott)": "ABT",
    "DHR (Danaher)": "DHR",
    "BMY (Bristol-Myers)": "BMY",
    "AMGN (Amgen)": "AMGN",
    "GILD (Gilead)": "GILD",
    "MRNA (Moderna)": "MRNA",
    "REGN (Regeneron)": "REGN",
    "CVS (CVS Health)": "CVS",
    # ── 消費・小売 ──
    "HD (Home Depot)": "HD",
    "PG (Procter & Gamble)": "PG",
    "KO (Coca-Cola)": "KO",
    "PEP (PepsiCo)": "PEP",
    "MCD (McDonald's)": "MCD",
    "DIS (Disney)": "DIS",
    "NKE (Nike)": "NKE",
    "SBUX (Starbucks)": "SBUX",
    "WMT (Walmart)": "WMT",
    "COST (Costco)": "COST",
    "TGT (Target)": "TGT",
    "LOW (Lowe's)": "LOW",
    "TJX (TJX Companies)": "TJX",
    "PM (Philip Morris)": "PM",
    "MO (Altria)": "MO",
    "NEE (NextEra Energy)": "NEE",
    # ── 工業・製造・運輸 ──
    "CAT (Caterpillar)": "CAT",
    "BA (Boeing)": "BA",
    "HON (Honeywell)": "HON",
    "UPS (UPS)": "UPS",
    "FDX (FedEx)": "FDX",
    "DE (Deere)": "DE",
    "GE (General Electric)": "GE",
    "RTX (Raytheon)": "RTX",
    "LMT (Lockheed Martin)": "LMT",
    "UNP (Union Pacific)": "UNP",
    # ── エネルギー・素材 ──
    "XOM (Exxon Mobil)": "XOM",
    "CVX (Chevron)": "CVX",
    "COP (ConocoPhillips)": "COP",
    "EOG (EOG Resources)": "EOG",
    "SLB (Schlumberger)": "SLB",
}

# ティッカー → 表示名（サジェスト・完全一致用）
TICKER_TO_LABEL = {ticker: label for label, ticker in US_STOCKS.items()}


def get_symbol_suggestions(query: str, max_results: int = 12) -> list[tuple[str, str]]:
    """検索クエリに一致する (表示名, ティッカー) のリストを返す。"""
    if not query or len(query.strip()) < 1:
        return []
    q = query.strip().upper()
    results = []
    for label, ticker in US_STOCKS.items():
        if q in label.upper() or q in ticker.upper():
            results.append((label, ticker))
        if len(results) >= max_results:
            break
    return results


def apply_basic_css():
    """米国株版用のシンプルなテーマ（既存のFX版より軽量）。"""
    st.markdown(
        """
<style>
.stApp {
    background-color: #050816;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
}
.main .block-container {
    max-width: 1400px;
}
div[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #30363d;
}
.metric-card {
    background: #0d1117;
    border-radius: 12px;
    border: 1px solid #30363d;
    padding: 14px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
</style>
""",
        unsafe_allow_html=True,
    )


def header():
    st.markdown(
        """
<div style="
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:18px 0 12px;
  border-bottom:1px solid rgba(48,54,61,0.8);
  margin-bottom:18px;
">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="
      width:44px;height:44px;border-radius:14px;
      background:linear-gradient(135deg,#1f6feb,#58a6ff);
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 8px 30px rgba(31,111,235,0.6);
      font-size:24px;
    ">📈</div>
    <div>
      <div style="font-size:1.5rem;font-weight:800;letter-spacing:-0.03em;">
        米国株マーケットダッシュボード
      </div>
      <div style="font-size:0.8rem;color:#8b949e;margin-top:2px;">
        代表的な米国株・ETFをテクニカル指標で一括分析
      </div>
    </div>
  </div>
  <div style="text-align:right;font-size:0.8rem;color:#8b949e;">
    データソース: Yahoo Finance (約15〜20分遅延)
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_overview_metrics(df: pd.DataFrame, symbol_label: str):
    if df.empty or len(df) < 2:
        st.info("データが不足しているため、メトリクスを表示できません。")
        return

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["Close"])
    change = float(latest["Close"] - prev["Close"])
    pct = (change / float(prev["Close"])) * 100 if prev["Close"] != 0 else 0.0

    high_1d = float(latest["High"])
    low_1d = float(latest["Low"])
    hl_range = high_1d - low_1d

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{symbol_label} 現在値", f"{price:,.2f} USD", f"{change:+.2f} ({pct:+.2f}%)")
    c2.metric("当日高値", f"{high_1d:,.2f} USD")
    c3.metric("当日安値", f"{low_1d:,.2f} USD")
    c4.metric("当日レンジ (H-L)", f"{hl_range:,.2f} USD")


def render_search_screen():
    """銘柄検索画面（メインエリア＋サジェストタイル）。"""
    st.markdown("### 🔍 銘柄を検索")
    st.caption("ティッカーまたは会社名を入力すると、候補が表示されます。詳細を見る銘柄を選んでください。")

    if "symbol_search" not in st.session_state:
        st.session_state.symbol_search = ""

    search_query = st.text_input(
        "検索",
        value=st.session_state.symbol_search,
        placeholder="例: AAPL, Apple, Tesla, QQQ...",
        key="search_screen_input",
        label_visibility="collapsed",
    )
    st.session_state.symbol_search = search_query

    # 表示する銘柄リスト: 検索あり→サジェスト、検索なし→人気銘柄（先頭24件）
    if search_query and len(search_query.strip()) >= 1:
        suggestions = get_symbol_suggestions(search_query, max_results=24)
    else:
        suggestions = list(US_STOCKS.items())[:24]  # (label, ticker) のリスト

    if not suggestions:
        st.info("該当する銘柄がありません。別のキーワードで検索してください。")
        return

    st.markdown("---")
    st.caption("💡 銘柄をクリックして詳細画面へ")

    # タイル表示（4列）
    items_per_row = 4
    for start in range(0, len(suggestions), items_per_row):
        chunk = suggestions[start : start + items_per_row]
        cols = st.columns(items_per_row)
        for i, item in enumerate(chunk):
            if len(item) == 2:
                label, ticker = item
            else:
                label, ticker = item[0], item[1]
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{ticker}**")
                    st.caption(label)
                    if st.button("詳細を見る →", key=f"detail_btn_{ticker}_{start}_{i}", use_container_width=True):
                        st.session_state.detail_symbol_ticker = ticker
                        st.session_state.detail_symbol_label = label
                        st.rerun()


def render_sidebar_detail():
    """詳細画面用サイドバー（期間・テクニカル指標のみ）。"""
    st.sidebar.markdown("### 🧭 期間・指標")
    timeframes = get_timeframes()
    tf_label = st.sidebar.selectbox("時間足（期間）", list(timeframes.keys()), index=4)
    period, interval = timeframes[tf_label]
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 テクニカル指標")
    with st.sidebar.expander("トレンド系", expanded=True):
        show_sma20 = st.checkbox("SMA (20)", value=True, key="d_sma20")
        show_sma50 = st.checkbox("SMA (50)", key="d_sma50")
        show_ema20 = st.checkbox("EMA (20)", key="d_ema20")
        show_bb = st.checkbox("ボリンジャーバンド (20,2)", key="d_bb")
        show_ichimoku = st.checkbox("一目均衡表", key="d_ichimoku")
        show_dmi = st.checkbox("DMI / ADX (14)", key="d_dmi")
        show_psar = st.checkbox("パラボリック SAR", key="d_psar")
        show_env = st.checkbox("エンベロープ (20,±2%)", key="d_env")
    with st.sidebar.expander("オシレーター系", expanded=False):
        show_rsi = st.checkbox("RSI (14)", key="d_rsi")
        show_macd = st.checkbox("MACD", key="d_macd")
        show_stoch = st.checkbox("ストキャスティクス (14,3)", key="d_stoch")
        show_psych = st.checkbox("サイコロジカルライン (12)", key="d_psych")
        show_rci = st.checkbox("RCI (9)", key="d_rci")
        show_madev = st.checkbox("移動平均乖離率 (25)", key="d_madev")
    with st.sidebar.expander("ボラティリティ・その他", expanded=False):
        show_hv = st.checkbox("ヒストリカル VoL (21)", key="d_hv")
        show_fib = st.checkbox("フィボナッチ", key="d_fib")
    return period, interval, {
        "sma20": show_sma20, "sma50": show_sma50, "ema20": show_ema20,
        "bb": show_bb, "ichimoku": show_ichimoku, "dmi": show_dmi,
        "psar": show_psar, "env": show_env,
        "rsi": show_rsi, "macd": show_macd, "stoch": show_stoch,
        "psych": show_psych, "rci": show_rci, "madev": show_madev,
        "hv": show_hv, "fib": show_fib,
    }


def render_detail_screen(ticker: str, label: str):
    """該当銘柄の詳細画面（チャート・メトリクス・指標）。"""
    col_back, _ = st.columns([1, 10])
    with col_back:
        if st.button("← 検索画面に戻る", key="back_to_search"):
            st.session_state.detail_symbol_ticker = None
            st.session_state.detail_symbol_label = None
            st.session_state.symbol_search = ""
            if "search_screen_input" in st.session_state:
                st.session_state.search_screen_input = ""
            st.rerun()
            return

    st.markdown(f"#### 📊 {label}  —  テクニカル分析")

    period, interval, indicator_flags = render_sidebar_detail()

    with st.spinner(f"{ticker} の株価データを取得中..."):
        df = fetch_fx_data(ticker, period=period, interval=interval, show_warnings=True, pair_name=ticker)

    if df.empty:
        st.warning(f"{ticker} のデータが見つかりませんでした。ティッカーが存在しないか、市場が休場中の可能性があります。")
        return

    df, overlays, oscillators = apply_indicators(df, indicator_flags)
    render_overview_metrics(df, label)

    st.markdown("### インタラクティブ・ローソク足チャート")
    html = render_google_candlestick_chart(df, ticker, overlays=overlays)
    components.html(html, height=640, scrolling=False)

    with st.expander("📊 生データ（最新100本）", expanded=False):
        st.dataframe(df.tail(100).sort_index(ascending=False), use_container_width=True)


def apply_indicators(df: pd.DataFrame, flags: dict):
    overlays = []
    oscillators = []

    if flags["sma20"]:
        df = add_sma(df, window=20)
        overlays.append("SMA_20")
    if flags["sma50"]:
        df = add_sma(df, window=50)
        overlays.append("SMA_50")
    if flags["ema20"]:
        df = add_ema(df, window=20)
        overlays.append("EMA_20")
    if flags["bb"]:
        df = add_bollinger_bands(df, window=20)
        overlays += ["BB_High_20", "BB_Low_20", "BB_Mid_20"]
    if flags["ichimoku"]:
        df = add_ichimoku(df)
        overlays += ["Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_SpanA", "Ichimoku_SpanB"]
    if flags["dmi"]:
        df = add_dmi(df)
        oscillators.append("ADX")
    if flags["psar"]:
        df = add_parabolic_sar(df)
        overlays.append("PSAR")
    if flags["env"]:
        df = add_envelope(df, window=20, pct=2.0)
        overlays += ["ENV_Upper_20", "ENV_Lower_20", "ENV_Mid_20"]
    if flags["rsi"]:
        df = add_rsi(df, window=14)
        oscillators.append("RSI_14")
    if flags["macd"]:
        df = add_macd(df)
        oscillators.append("MACD")
    if flags["stoch"]:
        df = add_stochastics(df)
        oscillators += ["Stoch_K", "Stoch_D"]
    if flags["psych"]:
        df = add_psychological_line(df, window=12)
        oscillators.append("PsychLine_12")
    if flags["rci"]:
        df = add_rci(df, window=9)
        oscillators.append("RCI_9")
    if flags["madev"]:
        df = add_ma_deviation(df, window=25)
        oscillators.append("MA_Dev_25")
    if flags["hv"]:
        df = add_historical_volatility(df, window=21)
        oscillators.append("HV_21")
    if flags["fib"]:
        df = add_fibonacci(df)
        overlays += ["Fib_236", "Fib_382", "Fib_500", "Fib_618", "Fib_786"]

    return df, overlays, oscillators


def main():
    apply_basic_css()
    header()

    if "detail_symbol_ticker" not in st.session_state:
        st.session_state.detail_symbol_ticker = None
    if "detail_symbol_label" not in st.session_state:
        st.session_state.detail_symbol_label = None

    if st.session_state.detail_symbol_ticker is None:
        # 検索画面: サイドバーは説明のみ
        with st.sidebar:
            st.caption("📌 銘柄を検索し、「詳細を見る」で分析画面へ進みます。")
            st.markdown("---")
            st.caption("⚠️ データは Yahoo Finance 経由のため約15〜20分遅延があります。")
        render_search_screen()
    else:
        # 詳細画面
        render_detail_screen(
            st.session_state.detail_symbol_ticker,
            st.session_state.detail_symbol_label,
        )


if __name__ == "__main__":
    main()

