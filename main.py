import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.data_fetcher import (
    fetch_fx_data, get_available_pairs, get_available_pairs_grouped, get_timeframes
)
from utils.market_status import get_market_status, get_market_status_detailed
from utils.indicators import (
    add_sma, add_ema, add_bollinger_bands, add_macd, add_rsi,
    add_ichimoku, add_dmi, add_parabolic_sar, add_envelope,
    add_stochastics, add_psychological_line, add_rci, add_ma_deviation,
    add_historical_volatility, add_fibonacci
)
from components.charts import create_mini_line_chart, render_google_candlestick_chart
from components.gauges import compute_signals, render_signal_dashboard
from utils.currency_info import get_pair_flags, get_pair_market_info


def validate_rate(pair_name: str, rate: float) -> bool:
    """
    通貨ペアのレートが妥当な範囲内か検証する
    
    Args:
        pair_name: 通貨ペア名（例: "USD/JPY", "JPY/NZD"）
        rate: 検証するレート
    
    Returns:
        bool: レートが妥当な範囲内の場合True、そうでない場合False
    """
    if "/" not in pair_name:
        return False
    
    base, quote = pair_name.split("/", 1)
    
    metal_codes = {"XAU", "XAG", "XPT", "XPD"}

    if rate <= 0:
        return False

    if base in metal_codes or quote in metal_codes:
        return True

    # JPYがbase通貨の場合、通常のレート範囲はかなり小さい
    if base == "JPY":
        return 0.001 <= rate <= 1.0
    
    # JPYがquote通貨の場合、通常は2桁-3桁
    elif quote == "JPY":
        if base == "USD":
            return 50 <= rate <= 300
        else:
            return 10 <= rate <= 1000
    
    # その他の通貨はエキゾチックを含むため広めに許容
    else:
        return 0.000001 <= rate <= 1000.0


st.set_page_config(
    page_title="FX 為替マーケットダッシュボード",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_theme_css(light_mode: bool = False):
    """
    ダーク/ライトモードのCSSを適用する。
    CSS変数（カスタムプロパティ）でテーマカラーを管理し、
    .light-mode クラスの有無で切り替える。
    """
    st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ─── テーマカラー変数（ダークモード） ───────────── */
:root {
    --bg-base:      #080c14;
    --bg-card:      #161b22;
    --bg-card2:     #0d1117;
    --bg-sidebar:   #0d1117;
    --border:       #21262d;
    --border2:      rgba(48,54,61,0.8);
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --text-sub:     #c9d1d9;
    --accent:       #388bfd;
    --accent2:      #58a6ff;
    --accent-dark:  #1f6feb;
    --green:        #3fb950;
    --red:          #f85149;
    --dot-color:    rgba(255,255,255,0.04);
    --hero-grad1:   rgba(31,111,235,0.22);
    --hero-grad2:   rgba(63,185,80,0.12);
    --hero-grid:    rgba(56,139,253,0.06);
    --shadow:       rgba(0,0,0,0.5);
    --shadow2:      rgba(0,0,0,0.4);
    --border-rgba: rgba(48,54,61,0.7);
    --green-rgba:  rgba(63,185,80,0.3);
    --green-rgba-light: rgba(63,185,80,0.1);
    --accent-rgba: rgba(56,139,253,0.2);
    --accent-rgba-light: rgba(56,139,253,0.06);
    --accent-dark-rgba: rgba(31,111,235,0.22);
    --accent-dark-rgba-light: rgba(31,111,235,0.45);
    --accent2-rgba: rgba(88,166,255,0.2);
    --bg-card-rgba: rgba(22,27,34,0.9);
    --bg-overlay: rgba(255,255,255,0.04);
    --hero-bg-start: #0d1823;
    --hero-bg-end: #0b0f18;
    --warning: #f0a500;
    --warning-rgba: rgba(240,165,0,0.1);
    --warning-rgba-border: rgba(240,165,0,0.35);
    --green-badge-bg: rgba(63,185,80,0.14);
    --red-badge-bg: rgba(248,81,73,0.14);
}

/* ─── ライトモード変数上書き ────────────────────── */
.light-mode {
    --bg-base:      #f0f4f8;
    --bg-card:      #ffffff;
    --bg-card2:     #f8fafc;
    --bg-sidebar:   #f8fafc;
    --border:       #dde3ea;
    --border2:      rgba(200,210,220,0.9);
    --text-primary: #1a202c;
    --text-muted:   #64748b;
    --text-sub:     #4a5568;
    --accent:       #2563eb;
    --accent2:      #3b82f6;
    --accent-dark:  #1d4ed8;
    --green:        #16a34a;
    --red:          #dc2626;
    --dot-color:    rgba(0,0,0,0.04);
    --hero-grad1:   rgba(37,99,235,0.12);
    --hero-grad2:   rgba(22,163,74,0.06);
    --hero-grid:    rgba(37,99,235,0.05);
    --shadow:       rgba(0,0,0,0.1);
    --shadow2:      rgba(0,0,0,0.06);
    --border-rgba: rgba(200,210,220,0.7);
    --green-rgba:  rgba(22,163,74,0.3);
    --green-rgba-light: rgba(22,163,74,0.1);
    --accent-rgba: rgba(37,99,235,0.2);
    --accent-rgba-light: rgba(37,99,235,0.06);
    --accent-dark-rgba: rgba(29,78,216,0.15);
    --accent-dark-rgba-light: rgba(29,78,216,0.3);
    --accent2-rgba: rgba(59,130,246,0.2);
    --bg-card-rgba: rgba(248,250,252,0.9);
    --bg-overlay: rgba(0,0,0,0.04);
    --hero-bg-start: #f0f4f8;
    --hero-bg-end: #e2e8f0;
    --warning: #f59e0b;
    --warning-rgba: rgba(245,158,11,0.1);
    --warning-rgba-border: rgba(245,158,11,0.35);
    --green-badge-bg: rgba(22,163,74,0.12);
    --red-badge-bg: rgba(220,38,38,0.12);
}

/* ─── フォント・リセット ────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Hiragino Sans', sans-serif !important;
}

/* ─── アプリ背景（ドットグリッド付き） ─────────── */
.stApp {
    background-color: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, var(--hero-grad1), transparent),
        radial-gradient(circle, var(--dot-color) 1px, transparent 1px);
    background-size: 100% 100%, 28px 28px;
    color: var(--text-primary) !important;
    transition: background-color 0.3s, color 0.3s;
}
.main .block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
    max-width: 1440px;
}

/* ─── サイドバー ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border2) !important;
    box-shadow: 4px 0 24px var(--shadow) !important;
}
section[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-dark), var(--accent2), var(--green));
    position: sticky;
    top: 0;
}
section[data-testid="stSidebar"] * {
    color: var(--text-sub) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
}

/* ─── セレクトボックス ──────────────────────────── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* ─── テキスト入力・ナンバー入力 ────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    caret-color: var(--accent) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-rgba) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
    opacity: 0.7;
}
/* ナンバー入力のステッパーボタン */
.stNumberInput > div > div > div > button {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
.stNumberInput > div > div > div > button:hover {
    background: var(--accent-rgba) !important;
    border-color: var(--accent) !important;
}

/* ─── メトリクスカード ──────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--accent) !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    box-shadow: 0 8px 32px var(--shadow), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s !important;
    position: relative;
    overflow: hidden;
}
div[data-testid="metric-container"]:hover {
    border-color: var(--accent-dark) !important;
    border-top-color: var(--accent2) !important;
    box-shadow: 0 12px 40px rgba(56,139,253,0.22) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
div[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* ─── セクション見出し ──────────────────────────── */
h1, h2, h3 { color: var(--text-primary) !important; }
h3 {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 1.8rem !important;
    margin-bottom: 0.8rem !important;
}
h3::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ─── タイルカード ──────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
    box-shadow: 0 4px 20px var(--shadow2);
    overflow: hidden;
    transition: border-color 0.25s, box-shadow 0.3s, transform 0.22s;
    position: relative;
}
div[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--accent-dark));
    border-radius: 3px 0 0 3px;
    opacity: 0;
    transition: opacity 0.3s;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 8px 36px rgba(56,139,253,0.2) !important;
    transform: translateY(-3px);
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover::before { opacity: 1; }

/* ─── ボタン ─────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-dark) 0%, var(--accent) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
    transition: all 0.2s !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.4), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    box-shadow: 0 5px 20px rgba(59,130,246,0.55) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ─── 戻るボタン ────────────────────────────────── */
div[data-testid="stHorizontalBlock"]:first-child .stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"]:first-child .stButton > button:hover {
    border-color: var(--accent2) !important;
    color: var(--text-primary) !important;
    background: rgba(56,139,253,0.1) !important;
    box-shadow: none !important;
}

/* ─── スピナー ──────────────────────────────────── */
div[data-testid="stSpinner"] > div { color: var(--accent) !important; }

/* ─── エクスパンダー ─────────────────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card) !important;
}

/* ─── データフレーム ─────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ─── Streamlit デフォルトUI を非表示 ─────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ─── 価格カラー ─────────────────────────────────── */
.price-up   { color: var(--green) !important; }
.price-down { color: var(--red)   !important; }

/* ─── データ遅延バジー (点滅アニメーション) ──────── */
@keyframes delay-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.55; }
}
.delay-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--warning);
    display: inline-block;
    animation: delay-pulse 2.5s ease-in-out infinite;
}

/* ─── チャートタイル内セクション ────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border) 20%, var(--border) 80%, transparent);
    margin: 1.5rem 0;
}

/* ─── ページネーション固定表示 ────────────────────── */
.pagination-container {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--bg-base);
    padding: 12px 0;
    margin: -12px 0 12px 0;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

    # ─── ライトモード切替スクリプト ────────────────────────────
    # 注意: 実際のテーマ適用はmain()関数内のスクリプトで行う
    # ここではCSS変数の定義のみを行う




# ──────────────────────────────────────────────────────────
# ヘッダー（詳細ページのみ）
# ──────────────────────────────────────────────────────────
def render_header(subtitle: str = ""):
    # 市場状態を取得
    market_info = get_market_status_detailed()
    status = market_info["status"]
    color = market_info["color"]
    
    # 市場状態に応じたスタイルを設定
    if color == "green":
        bg_color = "var(--green-rgba-light)"
        border_color = "var(--green-rgba)"
        text_color = "var(--green)"
    elif color == "orange":
        bg_color = "var(--warning-rgba)"
        border_color = "var(--warning-rgba-border)"
        text_color = "var(--warning)"
    else:  # red
        bg_color = "var(--red-badge-bg)"
        border_color = "var(--red-rgba)"
        text_color = "var(--red)"
    
    st.markdown(f"""
    <div style="
        display:flex;align-items:center;gap:14px;
        padding:16px 0 20px;
        border-bottom:1px solid var(--border-rgba);
        margin-bottom:22px;
    ">
        <div style="
            width:40px;height:40px;border-radius:10px;
            background:linear-gradient(135deg,var(--accent-dark),var(--accent2));
            display:flex;align-items:center;justify-content:center;
            font-size:20px;box-shadow:0 4px 16px var(--accent-dark-rgba-light);
            flex-shrink:0;
        ">📈</div>
        <div>
            <div style="font-size:1.2rem;font-weight:800;color:var(--text-primary);line-height:1.1;letter-spacing:-0.02em;">
                FX 為替マーケットダッシュボード
            </div>
            <div style="font-size:0.76rem;color:var(--text-muted);margin-top:2px;">{subtitle}</div>
        </div>
        <div style="margin-left:auto;">
            <div style="
                display:inline-flex;align-items:center;gap:7px;
                background:{bg_color};border:1px solid {border_color};
                padding:5px 13px;border-radius:20px;
                font-size:0.73rem;color:{text_color};font-weight:600;
                box-shadow:0 0 16px {bg_color};
            ">
                {status}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# ダッシュボード
# ──────────────────────────────────────────────────────────
def render_dashboard(available_pairs):
    import datetime
    now_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    time_str = now_jst.strftime("%Y/%m/%d  %H:%M JST")
    
    
    # ── HERO BANNER ────────────────────────────────────────
    st.markdown(f"""
    <div style="
        position:relative;
        border-radius:20px;
        overflow:hidden;
        margin-bottom:28px;
        border:1px solid var(--accent-rgba);
    ">
        <!-- グラデーション背景 -->
        <div style="
            position:absolute;inset:0;
            background:
                radial-gradient(ellipse 70% 80% at 0% 50%,  var(--accent-dark-rgba), transparent 60%),
                radial-gradient(ellipse 50% 60% at 100% 50%, var(--hero-grad2),  transparent 60%),
                linear-gradient(135deg, var(--hero-bg-start) 0%, var(--hero-bg-end) 100%);
        "></div>
        <!-- グリッドオーバーレイ -->
        <div style="
            position:absolute;inset:0;
            background-image: linear-gradient(var(--accent-rgba-light) 1px, transparent 1px),
                              linear-gradient(90deg, var(--accent-rgba-light) 1px, transparent 1px);
            background-size: 40px 40px;
        "></div>
        <!-- コンテンツ -->
        <div style="position:relative;padding:36px 40px;">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:20px;">
                <div>
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                        <div style="
                            width:48px;height:48px;border-radius:14px;
                            background:linear-gradient(135deg,var(--accent-dark) 0%,var(--accent2) 100%);
                            display:flex;align-items:center;justify-content:center;
                            font-size:26px;
                            box-shadow:0 6px 24px var(--accent-dark-rgba-light),0 0 0 1px var(--accent2-rgba);
                        ">📈</div>
                        <div>
                            <div style="font-size:1.6rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.03em;line-height:1;">
                                FX 為替マーケットダッシュボード
                            </div>
                        </div>
                    </div>
                </div>
                <!-- 右側の統計情報 -->
                <div style="text-align:center;">
                    <div style="font-size:2.2rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.04em;line-height:1;">{len(available_pairs)}</div>
                    <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.12em;margin-top:4px;">通貨ペア</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 検索とフィルタ機能 ──────────────────────────────────────
    st.markdown("""
    <div style="
        padding:16px;background:var(--bg-card);
        border:1px solid var(--border);border-radius:12px;
        margin-bottom:20px;
    ">
        <div style="font-size:0.7rem;font-weight:700;color:var(--accent);
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px;">
            🔍 検索・フィルタ
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_search, col_filter = st.columns([2, 1])
    
    # サジェスト選択用のセッション状態
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    
    with col_search:
        # サジェストが選択された場合はその値を使用、そうでなければ入力値を使用
        if st.session_state.selected_suggestion:
            search_query = st.text_input(
                "🔍 通貨ペアを検索", 
                value=st.session_state.selected_suggestion, 
                placeholder="例: USD, JPY, EUR...", 
                key="search_input", 
                help="通貨コードで検索できます（例: USD, JPY）",
                autocomplete="off"
            )
            st.session_state.selected_suggestion = None  # リセット
        else:
            search_query = st.text_input(
                "🔍 通貨ペアを検索", 
                placeholder="例: USD, JPY, EUR...", 
                key="search_input", 
                help="通貨コードで検索できます（例: USD, JPY）",
                autocomplete="off"
            )
        
        # リアルタイムサジェスト表示（入力中に自動表示）
        if search_query and len(search_query.strip()) >= 1:
            search_upper = search_query.upper().strip()
            suggestions = [
                pair for pair in available_pairs.keys()
                if search_upper in pair.replace('/', '').upper() or search_upper in pair.upper()
            ][:8]  # 最大8件
            
            if suggestions:
                st.markdown("""
                <div style="
                    font-size:0.7rem;color:var(--text-muted);
                    margin-top:6px;margin-bottom:6px;
                    font-weight:600;
                ">
                    💡 サジェスト:
                </div>
                """, unsafe_allow_html=True)
                
                # サジェストをボタンとして表示（4列）
                cols_suggestions = st.columns(min(len(suggestions), 4))
                for idx, pair in enumerate(suggestions[:4]):
                    with cols_suggestions[idx]:
                        if st.button(pair, key=f"suggest_{pair}_{idx}", use_container_width=True):
                            # サジェスト選択をセッション状態に保存
                            st.session_state.selected_suggestion = pair
                            st.rerun()
    
    # ドル円 (USD/JPY) のティッカーは yfinance の仕様に合わせて常に "JPY=X" に固定しておく
    # （1 USD = ? JPY を意味するレート）
    if "USD/JPY" in available_pairs:
        available_pairs["USD/JPY"] = "JPY=X"

    with col_filter:
        categories = get_available_pairs_grouped()
        category_options = ["すべて"] + list(categories.keys())
        # 初期値を「クロス円」に設定
        default_index = category_options.index("クロス円") if "クロス円" in category_options else 0
        filter_category = st.selectbox("📂 カテゴリで絞り込み", category_options, index=default_index, key="filter_category", help="通貨ペアをカテゴリで絞り込みます")
    
    # カテゴリに基づいてペアを取得（フィルタリングではなく、選択したカテゴリのみを取得）
    filtered_pairs = {}
    
    if filter_category != "すべて":
        # 選択したカテゴリのペアのみを取得
        if filter_category in categories:
            category_pairs = categories[filter_category]
            for pair_name in category_pairs:
                if pair_name in available_pairs:
                    filtered_pairs[pair_name] = available_pairs[pair_name]
    else:
        # 「すべて」が選択されている場合は全ペアを取得
        filtered_pairs = available_pairs.copy()
    
    # 検索クエリがある場合は、カテゴリ内で検索
    if search_query:
        search_upper = search_query.upper().strip()
        filtered_pairs = {
            pair_name: ticker 
            for pair_name, ticker in filtered_pairs.items()
            if search_upper in pair_name.replace('/', '').upper() or search_upper in pair_name.upper()
        }
    
    # 主要通貨の定義（pair_generator.MINOR_CURRENCIES と合わせてホワイトリストを構成）
    PRIORITY_CURRENCIES = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]
    MINOR_CURRENCIES = ["SEK", "NOK", "DKK", "HKD", "CNH", "SGD", "MXN", "TRY", "ZAR"]
    ALLOWED_CURRENCIES = set(PRIORITY_CURRENCIES + MINOR_CURRENCIES)

    # ペア名がホワイトリスト通貨のみで構成されているものだけを対象にする
    def _is_allowed_pair(pair_name: str) -> bool:
        if "/" not in pair_name:
            return False
        base, quote = pair_name.split("/", 1)
        return base in ALLOWED_CURRENCIES and quote in ALLOWED_CURRENCIES

    filtered_pairs = {
        pair_name: ticker
        for pair_name, ticker in filtered_pairs.items()
        if _is_allowed_pair(pair_name)
    }

    # 検索結果数を表示
    if search_query or filter_category != "すべて":
        result_count = len(filtered_pairs)
        st.caption(f"✅ {result_count} 件の通貨ペアが見つかりました")
    
    # カテゴリで絞り込んだ結果が空の場合は処理を終了
    if len(filtered_pairs) == 0:
        st.info(f"📂 「{filter_category}」カテゴリに該当する通貨ペアが見つかりませんでした。")
        return
    
    # データが取得できるペアのみをフィルタリング（主要通貨優先、キャッシュ付きマルチスレッド処理）
    # 重要: ここで処理されるのは filtered_pairs（カテゴリで絞り込んだペア）のみです
    # 高速化のため、検証用には短い期間のデータを使用（1週間で十分）
    validation_period = "5d"
    validation_interval = "1d"
    # 表示用には長い期間を使用
    sparkline_period = "1mo"
    sparkline_interval = "1d"
    
    # セッション状態にキャッシュを保存（フィルタが変わらない限り再確認しない）
    # フィルタ条件のハッシュを生成（検索クエリとカテゴリで判定）
    # cache_version を変更すると valid_pairs の中身を強制的に再構築できる
    cache_version = "fx-rate-v4"
    filter_hash = hash((cache_version, search_query, filter_category))
    cache_key = f"valid_pairs_cache_{filter_hash}"
    
    # メジャーペア処理完了フラグをチェック
    major_processing_key = f"{cache_key}_major_processed"
    other_processing_key = f"{cache_key}_other_processed"
    
    if cache_key not in st.session_state:
        valid_pairs = {}
        st.session_state[cache_key] = {}
        st.session_state[major_processing_key] = False
        st.session_state[other_processing_key] = False
    
    def check_pair_validity(pair_name_ticker):
        """ペアのデータ取得可能性をチェック（タイムアウト付き）
        
        注意:
            ここではあくまで「利用可能かどうか」の判定だけを行う。
            ログは最終表示時のレートと混同すると分かりにくいため出さない。
        """
        pair_name, ticker = pair_name_ticker
        try:
            # 短い期間で高速に検証（1週間で十分）
            # 検証フェーズでは pair_name を渡さず、ログは出さない
            df = fetch_fx_data(
                ticker,
                period=validation_period,
                interval=validation_interval,
                show_warnings=False,
                pair_name=None,
            )
            if not df.empty:
                df = df.dropna(subset=['Close'])
                if not df.empty:
                    # レート検証も行う
                    latest_close = float(df['Close'].iloc[-1])
                    if validate_rate(pair_name, latest_close):
                        return pair_name, ticker
        except Exception:
            pass
        return None
    
    def get_pair_priority(pair_name):
        """ペアの優先度を取得（カテゴリに応じて優先度を変更）"""
        base, quote = pair_name.split("/")[:2] if "/" in pair_name else (pair_name[:3], pair_name[3:])
        
        # クロス円カテゴリが選択されている場合、クロス円ペア（JPYを含むペア）のみを優先度1として処理
        if filter_category == "クロス円":
            if quote == "JPY" or base == "JPY":
                return 1  # クロス円ペア（すべて優先度1として処理）
            else:
                return 999  # クロス円以外は除外（処理しない）
        
        # オセアニアカテゴリが選択されている場合、オセアニアペア（AUDまたはNZDを含むペア）のみを優先度1として処理
        if filter_category == "オセアニア":
            if base in ["AUD", "NZD"] or quote in ["AUD", "NZD"]:
                return 1  # オセアニアペア（すべて優先度1として処理）
            else:
                return 999  # オセアニア以外は除外（処理しない）
        
        # ユーロカテゴリが選択されている場合、ユーロペア（EURを含むペア）のみを優先度1として処理
        if filter_category == "ユーロ":
            if base == "EUR" or quote == "EUR":
                return 1  # ユーロペア（すべて優先度1として処理）
            else:
                return 999  # ユーロ以外は除外（処理しない）
        
        # ポンドカテゴリが選択されている場合、ポンドペア（GBPを含むペア）のみを優先度1として処理
        if filter_category == "ポンド":
            if base == "GBP" or quote == "GBP":
                return 1  # ポンドペア（すべて優先度1として処理）
            else:
                return 999  # ポンド以外は除外（処理しない）
        
        # その他のカテゴリの場合
        if base in PRIORITY_CURRENCIES and quote in PRIORITY_CURRENCIES:
            return 1  # メジャーペア
        elif base in PRIORITY_CURRENCIES or quote in PRIORITY_CURRENCIES:
            return 2  # 主要通貨を含む
        else:
            return 3  # その他
    
    # カテゴリで絞り込んだペア（filtered_pairs）を優先度順にソート
    # 重要: ここで処理されるのは選択したカテゴリのペアのみです
    sorted_pairs = sorted(filtered_pairs.items(), key=lambda x: get_pair_priority(x[0]))
    
    # 優先度1のペア（メジャーペアまたはクロス円ペア）を分離
    # 重要: priority_pairs と other_pairs は filtered_pairs（カテゴリで絞り込んだペア）から作成されます
    # クロス円カテゴリの場合は、クロス円ペア（JPYを含むペア）のみを処理
    priority_pairs = [(p, t) for p, t in sorted_pairs if get_pair_priority(p) == 1]
    other_pairs = [(p, t) for p, t in sorted_pairs if get_pair_priority(p) > 1 and get_pair_priority(p) < 999]
    
    # 優先度1のペアの処理（まだ完了していない場合）
    # カテゴリに応じてラベルを変更
    # 重要: ここで処理されるのは filtered_pairs（カテゴリで絞り込んだペア）のみです
    if filter_category == "クロス円":
        priority_label = "クロス円"
    elif filter_category == "オセアニア":
        priority_label = "オセアニア"
    elif filter_category == "ユーロ":
        priority_label = "ユーロ"
    elif filter_category == "ポンド":
        priority_label = "ポンド"
    else:
        priority_label = "メジャーペア"
    
    if not st.session_state[major_processing_key] and priority_pairs:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"⭐ {priority_label}を優先処理中... (0 / {len(priority_pairs)})")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_pair = {
                executor.submit(check_pair_validity, (pair_name, ticker)): (pair_name, ticker)
                for pair_name, ticker in priority_pairs
            }
            
            completed = 0
            for future in as_completed(future_to_pair, timeout=None):
                completed += 1
                progress = completed / len(priority_pairs)
                progress_bar.progress(progress)
                status_text.text(f"⭐ {priority_label}処理中... ({completed} / {len(priority_pairs)})")
                
                try:
                    result = future.result(timeout=2.0)
                    if result is not None:
                        pair_name, ticker = result
                        st.session_state[cache_key][pair_name] = ticker
                except Exception:
                    pass
        
        # 処理完了後、プログレスバーとステータステキストを確実に消去
        progress_bar.progress(1.0)  # 100%に設定
        status_text.text(f"✅ {priority_label}処理完了！")
        
        # フラグを設定
        st.session_state[major_processing_key] = True
        
        # プログレスバーとステータステキストを消去
        progress_bar.empty()
        status_text.empty()
        progress_container.empty()
        
        # 優先度1のペア処理完了後、即座に画面を更新して表示
        st.rerun()
        return  # st.rerun()の後は実行されないが、念のため
    
    # 優先度1のペアが空の場合、フラグを設定して残りのペアの処理に進む
    if not priority_pairs:
        st.session_state[major_processing_key] = True
    
    # 「すべて」カテゴリの場合は、すべてのペア（優先度1、2、3）を処理する
    # その他のカテゴリの場合は、優先度1のペアのみを処理し、その他のペアは処理しない（高速化のため）
    if filter_category == "すべて":
        # 「すべて」カテゴリの場合、残りのペアも処理する
        if st.session_state[major_processing_key] and not st.session_state[other_processing_key] and other_pairs:
            status_container = st.container()
            with status_container:
                progress_bar = st.progress(0.5)
                status_text = st.empty()
                status_text.text(f"✅ {priority_label}完了！残りのペアを処理中... (0 / {len(other_pairs)})")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_pair = {
                    executor.submit(check_pair_validity, (pair_name, ticker)): (pair_name, ticker)
                    for pair_name, ticker in other_pairs
                }
                
                completed = 0
                for future in as_completed(future_to_pair, timeout=None):
                    completed += 1
                    progress = 0.5 + (completed / len(other_pairs)) * 0.5
                    progress_bar.progress(progress)
                    status_text.text(f"処理中... ({completed} / {len(other_pairs)})")
                    
                    try:
                        result = future.result(timeout=2.0)
                        if result is not None:
                            pair_name, ticker = result
                            st.session_state[cache_key][pair_name] = ticker
                    except Exception:
                        pass
            
            progress_bar.empty()
            status_text.empty()
            status_container.empty()
            st.session_state[other_processing_key] = True
            st.rerun()
            return
    else:
        # 「すべて」以外のカテゴリの場合、優先度1のペアのみを処理
        # 優先度1のペア処理完了後、フラグを設定して完了とする
        if st.session_state[major_processing_key]:
            st.session_state[other_processing_key] = True
    
    # セッション状態から取得
    valid_pairs = st.session_state[cache_key]

    # ドル円 (USD/JPY) は必ず候補に含める
    # ・カテゴリ「クロス円」や「すべて」で表示対象から抜けないようにするため
    # ・検証フェーズで一時的に失敗しても、ここで最終的に復活させる
    if "USD/JPY" in filtered_pairs:
        valid_pairs["USD/JPY"] = filtered_pairs["USD/JPY"]
    
    # デバッグ: データ取得状況を確認
    if len(filtered_pairs) > 0 and len(valid_pairs) == 0:
        # データ取得処理が完了していない場合、処理中のメッセージを表示
        if not st.session_state[major_processing_key] or not st.session_state[other_processing_key]:
            st.info(f"⏳ {len(filtered_pairs)}件の通貨ペアのデータ取得を処理中です...")
            return
    
    # 表示時にも優先度順にソート（主要通貨が最初に表示される）
    def get_pair_priority_for_sort(pair_name):
        """ペアの優先度を取得（表示用）"""
        base, quote = pair_name.split("/")[:2] if "/" in pair_name else (pair_name[:3], pair_name[3:])
        # クロス円カテゴリのときは、ユーザー指定の順番でソートする
        if filter_category == "クロス円" and quote == "JPY":
            # クロス円（メジャー通貨）の順番:
            # USD/JPY, GBP/JPY, CAD/JPY, CHF/JPY, AUD/JPY, EUR/JPY, NZD/JPY
            cross_major_order = {
                "USD": 0,
                "GBP": 1,
                "CAD": 2,
                "CHF": 3,
                "AUD": 4,
                "EUR": 5,
                "NZD": 6,
            }
            # クロス円（その他）の順番:
            # NOK/JPY, SEK/JPY, HKD/JPY, MXN/JPY, DKK/JPY, TRY/JPY, ZAR/JPY
            cross_other_order = {
                "NOK": 100,
                "SEK": 101,
                "HKD": 102,
                "MXN": 103,
                "DKK": 104,
                "TRY": 105,
                "ZAR": 106,
            }
            if base in cross_major_order:
                return cross_major_order[base]
            if base in cross_other_order:
                return cross_other_order[base]
            return 999

        # それ以外のカテゴリでは、これまで通りの優先度
        # 最初のタイルには必ず USD/JPY を表示する
        if pair_name == "USD/JPY":
            return 0
        if base in PRIORITY_CURRENCIES and quote in PRIORITY_CURRENCIES:
            return 1  # メジャーペア
        elif base in PRIORITY_CURRENCIES or quote in PRIORITY_CURRENCIES:
            return 2  # 主要通貨を含む
        else:
            return 3  # その他
    
    # 有効なペアを優先度順にソート
    valid_pairs_sorted = dict(sorted(valid_pairs.items(), key=lambda x: get_pair_priority_for_sort(x[0])))

    # クロス円カテゴリが選択されている場合は、表示対象もクロス円ペアのみに限定する
    if filter_category == "クロス円":
        valid_pairs_sorted = {
            pair_name: ticker
            for pair_name, ticker in valid_pairs_sorted.items()
            if ("/" in pair_name and ("JPY" in pair_name.split("/", 1)))
        }
    
    # 優先度別の統計情報を表示
    if filter_category == "クロス円":
        # クロス円カテゴリの場合
        priority_stats = {
            "クロス円（メジャー通貨）": [],
            "クロス円（その他）": [],
            "その他": []
        }
        
        for pair_name in valid_pairs_sorted.keys():
            base, quote = pair_name.split("/")[:2] if "/" in pair_name else (pair_name[:3], pair_name[3:])
            if quote == "JPY":
                if base in PRIORITY_CURRENCIES:
                    priority_stats["クロス円（メジャー通貨）"].append(pair_name)
                else:
                    priority_stats["クロス円（その他）"].append(pair_name)
            else:
                priority_stats["その他"].append(pair_name)

        # 表示順をユーザー指定の順番に固定する
        # クロス円（メジャー通貨）
        major_display_order = [
            "USD/JPY",
            "GBP/JPY",
            "CAD/JPY",
            "CHF/JPY",
            "AUD/JPY",
            "EUR/JPY",
            "NZD/JPY",
        ]
        priority_stats["クロス円（メジャー通貨）"] = [
            pair for pair in major_display_order
            if pair in priority_stats["クロス円（メジャー通貨）"]
        ]

        # クロス円（その他）
        other_display_order = [
            "NOK/JPY",
            "SEK/JPY",
            "HKD/JPY",
            "MXN/JPY",
            "DKK/JPY",
            "TRY/JPY",
            "ZAR/JPY",
        ]
        priority_stats["クロス円（その他）"] = [
            pair for pair in other_display_order
            if pair in priority_stats["クロス円（その他）"]
        ]

        # ユーザー指定の順番で並び替え
        # クロス円（メジャー通貨）: USD/JPY, GBP/JPY, CAD/JPY, CHF/JPY, AUD/JPY, EUR/JPY
        if priority_stats["クロス円（メジャー通貨）"]:
            major_order = {
                "USD": 0,
                "GBP": 1,
                "CAD": 2,
                "CHF": 3,
                "AUD": 4,
                "EUR": 5,
            }

            def _cross_major_sort_key(pair_name: str):
                if "/" in pair_name:
                    base, _ = pair_name.split("/", 1)
                else:
                    base = pair_name[:3]
                return (
                    major_order.get(base, 999),
                    base,
                    pair_name,
                )

            priority_stats["クロス円（メジャー通貨）"] = sorted(
                priority_stats["クロス円（メジャー通貨）"],
                key=_cross_major_sort_key,
            )

        # クロス円（その他）: NOK/JPY, SEK/JPY, HKD/JPY, MXN/JPY, DKK/JPY, TRY/JPY, ZAR/JPY
        if priority_stats["クロス円（その他）"]:
            other_order = {
                "NOK": 0,
                "SEK": 1,
                "HKD": 2,
                "MXN": 3,
                "DKK": 4,
                "TRY": 5,
                "ZAR": 6,
            }

            def _cross_other_sort_key(pair_name: str):
                if "/" in pair_name:
                    base, _ = pair_name.split("/", 1)
                else:
                    base = pair_name[:3]
                return (
                    other_order.get(base, 999),
                    base,
                    pair_name,
                )

            priority_stats["クロス円（その他）"] = sorted(
                priority_stats["クロス円（その他）"],
                key=_cross_other_sort_key,
            )

        # 統計情報を表示
        st.markdown("""
        <div style="
            padding:12px 16px;background:var(--bg-card);
            border:1px solid var(--border);border-radius:10px;
            margin-bottom:16px;
        ">
            <div style="font-size:0.7rem;font-weight:700;color:var(--accent);
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
                📊 クロス円優先処理の結果
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric(
                "クロス円（メジャー）",
                f"{len(priority_stats['クロス円（メジャー通貨）'])}件",
                help="USD/JPY, EUR/JPY, GBP/JPYなど"
            )
        with stat_cols[1]:
            st.metric(
                "クロス円（その他）",
                f"{len(priority_stats['クロス円（その他）'])}件",
                help="その他のクロス円ペア"
            )
        with stat_cols[2]:
            st.metric(
                "その他",
                f"{len(priority_stats['その他'])}件",
                help="その他の通貨ペア"
            )
    else:
        # その他のカテゴリの場合
        priority_stats = {
            "メジャーペア（優先度1）": [],
            "主要通貨を含む（優先度2）": [],
            "その他（優先度3）": []
        }
        
        for pair_name in valid_pairs_sorted.keys():
            base, quote = pair_name.split("/")[:2] if "/" in pair_name else (pair_name[:3], pair_name[3:])
            if base in PRIORITY_CURRENCIES and quote in PRIORITY_CURRENCIES:
                priority_stats["メジャーペア（優先度1）"].append(pair_name)
            elif base in PRIORITY_CURRENCIES or quote in PRIORITY_CURRENCIES:
                priority_stats["主要通貨を含む（優先度2）"].append(pair_name)
            else:
                priority_stats["その他（優先度3）"].append(pair_name)
        
        # 統計情報を表示
        st.markdown("""
        <div style="
            padding:12px 16px;background:var(--bg-card);
            border:1px solid var(--border);border-radius:10px;
            margin-bottom:16px;
        ">
            <div style="font-size:0.7rem;font-weight:700;color:var(--accent);
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
                📊 主要通貨優先処理の結果
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric(
                "メジャーペア",
                f"{len(priority_stats['メジャーペア（優先度1）'])}件",
                help="USD, EUR, JPY, GBP, AUD, CAD, CHF, NZDの組み合わせ"
            )
        with stat_cols[1]:
            st.metric(
                "主要通貨を含む",
                f"{len(priority_stats['主要通貨を含む（優先度2）'])}件",
                help="主要通貨のいずれかを含むペア"
            )
        with stat_cols[2]:
            st.metric(
                "その他",
                f"{len(priority_stats['その他（優先度3）'])}件",
                help="その他の通貨ペア"
            )
    
    # 詳細を展開可能なセクションで表示
    with st.expander("📋 優先度別の詳細リスト", expanded=False):
        for priority_name, pairs in priority_stats.items():
            if pairs:
                st.markdown(f"**{priority_name}** ({len(pairs)}件)")
                # 10件ずつ表示
                display_pairs = pairs[:10]
                pairs_text = ", ".join(display_pairs)
                if len(pairs) > 10:
                    pairs_text += f" ... 他{len(pairs) - 10}件"
                st.caption(pairs_text)
                st.markdown("---")
    
    # ページネーション設定
    items_per_page = 20  # 1ページあたりのアイテム数
    total_items = len(valid_pairs)
    total_pages = (total_items + items_per_page - 1) // items_per_page if total_items > 0 else 1
    
    # セッション状態でページ番号を管理
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # 検索やフィルタが変更されたらページをリセット
    if 'last_search' not in st.session_state:
        st.session_state.last_search = search_query
        st.session_state.last_filter = filter_category
    
    if (st.session_state.last_search != search_query or 
        st.session_state.last_filter != filter_category):
        st.session_state.current_page = 1
        st.session_state.last_search = search_query
        st.session_state.last_filter = filter_category
        # フィルタが変更されたらキャッシュをクリア
        # 古いキャッシュキーを削除（すべてのキャッシュキーをクリア）
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("valid_pairs_cache_")]
        for k in keys_to_remove:
            del st.session_state[k]
    
    # ページネーションコントロール（検索・フィルタの直下に配置）
    if total_pages > 1:
        st.markdown("<div style='margin:8px 0 24px 0;'></div>", unsafe_allow_html=True)
        
        col_page_info, col_page_prev, col_page_next, col_page_jump = st.columns([3, 1, 1, 1])
        
        with col_page_info:
            st.markdown(f"""
            <div style="
                display:flex;align-items:center;gap:8px;
                padding:12px 14px 16px 14px;background:var(--bg-card);
                border:1px solid var(--border);border-radius:8px;
                box-shadow:0 2px 8px var(--shadow2);
                white-space:nowrap;overflow:hidden;
                min-height:50px;
            ">
                <span style="font-size:0.75rem;color:var(--text-muted);flex-shrink:0;">📄</span>
                <span style="font-size:0.85rem;color:var(--text-primary);font-weight:600;flex-shrink:0;">
                    {st.session_state.current_page} / {total_pages}
                </span>
                <span style="font-size:0.75rem;color:var(--text-muted);flex-shrink:0;">
                    ({total_items}件)
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_page_prev:
            if st.button("◀ 前へ", use_container_width=True, disabled=(st.session_state.current_page == 1), key="prev_page_top"):
                st.session_state.current_page -= 1
                # page_jump_topも同期
                if "page_jump_top" in st.session_state:
                    st.session_state.page_jump_top = st.session_state.current_page
                st.rerun()
        
        with col_page_next:
            if st.button("次へ ▶", use_container_width=True, disabled=(st.session_state.current_page >= total_pages), key="next_page_top"):
                st.session_state.current_page += 1
                # page_jump_topも同期
                if "page_jump_top" in st.session_state:
                    st.session_state.page_jump_top = st.session_state.current_page
                st.rerun()
        
        with col_page_jump:
            # セッション状態にキーが存在しない場合の初期化
            if "page_jump_top" not in st.session_state:
                st.session_state.page_jump_top = st.session_state.current_page
            # current_pageが変更された場合、page_jump_topも同期
            elif st.session_state.page_jump_top != st.session_state.current_page:
                st.session_state.page_jump_top = st.session_state.current_page
            
            # valueパラメータを削除し、セッション状態から値を取得
            page_num = st.number_input(
                "ページ番号", 
                min_value=1, 
                max_value=total_pages, 
                key="page_jump_top", 
                label_visibility="visible",
                help="ページ番号を入力して移動"
            )
            # ページ番号が変更された場合、current_pageを更新
            if page_num != st.session_state.current_page:
                st.session_state.current_page = page_num
                st.rerun()
        
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
    
    # 現在のページのアイテムを取得（データが取得できるペアのみ、優先度順）
    pairs_list = list(valid_pairs_sorted.items())
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_page_pairs = pairs_list[start_idx:end_idx]
    
    if not current_page_pairs:
        st.markdown("""
        <div style="
            padding:40px 20px;text-align:center;
            background:var(--bg-card);border:1px solid var(--border);
            border-radius:14px;margin:20px 0;
        ">
            <div style="font-size:3rem;margin-bottom:12px;">🔍</div>
            <div style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:6px;">
                検索結果が見つかりませんでした
            </div>
            <div style="font-size:0.85rem;color:var(--text-muted);">
                別のキーワードやカテゴリで検索してみてください
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # ── TILE GRID ──────────────────────────────────────────
    # 最後の行の空の列を非表示にするCSS
    total_items_on_page = len(current_page_pairs)
    items_per_row = 4
    full_rows = total_items_on_page // items_per_row
    remaining_items = total_items_on_page % items_per_row
    
    # 最後の行に空の列がある場合、CSSで非表示にする
    if remaining_items > 0 and remaining_items < items_per_row:
        # Streamlitの列は順番に配置されるので、最後の行の空の列を非表示にする
        # 最後の行の開始インデックス
        last_row_start = full_rows * items_per_row
        # 空の列の数
        empty_cols = items_per_row - remaining_items
        
        # CSSで最後の行の空の列を非表示にする
        # Streamlitの列はstColumnクラスを持つが、直接指定は難しいため、
        # データコンテナの親要素を使って非表示にする
        st.markdown(f"""
        <style>
        /* 最後の行の空の列を非表示にする */
        div[data-testid="column"]:nth-child(n+{last_row_start + remaining_items + 1}):nth-child(-n+{last_row_start + items_per_row}) {{
            display: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    cols = st.columns(4, gap="small")

    with st.spinner("マーケットデータを表示中..."):
        for i, (pair_name, ticker) in enumerate(current_page_pairs):
            # 念のため、最新のティッカー情報で上書きしてからレートを取得する
            if pair_name in available_pairs:
                ticker = available_pairs[pair_name]

            # データは既に取得済みなので、再度取得
            try:
                df = fetch_fx_data(ticker, period=sparkline_period, interval=sparkline_interval, show_warnings=False, pair_name=pair_name)
                if df.empty:
                    continue
                df = df.dropna(subset=['Close'])
                if df.empty:
                    continue
                
                # データが1行しかない場合の処理
                has_sufficient_data = len(df) >= 2
                if not has_sufficient_data and len(df) == 1:
                    # 1行しかない場合は、その1行を2回使用してチャートを表示可能にする
                    df = pd.concat([df, df], ignore_index=False)
                
            except Exception as e:
                # USD/JPYの場合はエラー詳細を表示
                if pair_name == "USD/JPY":
                    st.error(f"❌ {pair_name} ({ticker}) のデータ取得エラー: {str(e)}")
                continue

            latest          = df.iloc[-1]
            start_of_period = df.iloc[0]

            # pandas の仕様により、MultiIndex 等から Series が返る場合があるため安全に float へ変換
            close_value = latest["Close"]
            if isinstance(close_value, pd.Series):
                current_price = float(close_value.iloc[0])
            else:
                current_price = float(close_value)
            
            # 表示用にペアを分割（例: USD / JPY）
            if "/" in pair_name:
                base, quot = pair_name.split("/", 1)
            else:
                base = pair_name[:3] if len(pair_name) >= 6 else pair_name
                quot = pair_name[3:] if len(pair_name) >= 6 else ""
            
            # レートの妥当性を検証
            if not validate_rate(pair_name, current_price):
                # レートが異常な範囲の場合は表示をスキップ
                import sys
                print(
                    f"[レート検証失敗] {pair_name}: ticker={ticker}, レート={current_price:.6f}は異常な範囲です。表示をスキップします。",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            
            # デバッグ: レート情報をコンソールに出力（High/Low/Open も安全に float へ変換）
            import sys
            high_val = latest["High"]
            low_val = latest["Low"]
            open_val = latest["Open"]
            if isinstance(high_val, pd.Series):
                high_val = high_val.iloc[0]
            if isinstance(low_val, pd.Series):
                low_val = low_val.iloc[0]
            if isinstance(open_val, pd.Series):
                open_val = open_val.iloc[0]
            print(
                f"[レート情報] {pair_name}: ticker={ticker}, レート={current_price:.6f}, "
                f"High={float(high_val):.6f}, Low={float(low_val):.6f}, Open={float(open_val):.6f}",
                file=sys.stderr,
                flush=True,
            )
            
            # データが1行しかない場合は価格変動を0にする
            if len(df) == 1:
                price_change = 0.0
                pct_change = 0.0
            else:
                start_close_value = start_of_period["Close"]
                if isinstance(start_close_value, pd.Series):
                    start_price = float(start_close_value.iloc[0])
                else:
                    start_price = float(start_close_value)
                price_change    = current_price - start_price
                pct_change      = (price_change / start_price) * 100

            is_up       = bool(price_change >= 0)
            sign        = "+" if is_up else ""
            color_class = "price-up" if is_up else "price-down"
            
            # JPYが含まれるペアのレート表示フォーマットを改善
            if base == "JPY":
                # JPYがbaseの場合、通常0.01-0.1程度なので、小数点以下4桁表示
                price_fmt   = f"{current_price:.4f}"
                change_fmt  = f"{sign}{price_change:.4f}"
            elif quot == "JPY":
                # JPYがquoteの場合、通常50-200程度なので、小数点以下2桁表示
                price_fmt   = f"{current_price:.2f}"
                change_fmt  = f"{sign}{price_change:.2f}"
            else:
                # その他のペアは小数点以下4桁表示
                price_fmt   = f"{current_price:.4f}"
                change_fmt  = f"{sign}{price_change:.4f}"

            accent_color_class = "var(--green)" if is_up else "var(--red)"
            badge_bg_class = "var(--green-badge-bg)" if is_up else "var(--red-badge-bg)"
            arrow        = "▲" if is_up else "▼"

            # デバッグ: 実際にタイルとして表示している値を明示的にログ出力
            # これにより、画面表示とコンソールログの対応関係を確認しやすくする
            import sys
            print(
                f"[タイル表示] {pair_name}: ticker={ticker}, 表示レート={price_fmt}, 変化={change_fmt}, 変化率={sign}{pct_change:.2f}%",
                file=sys.stderr,
                flush=True,
            )
            
            # 通貨情報を取得
            pair_info = get_pair_market_info(pair_name)
            base_flag = pair_info.get("base_flag", "")
            quote_flag = pair_info.get("quote_flag", "")
            base_market = pair_info.get("base_market")
            quote_market = pair_info.get("quote_market")
            
            # 市場時間情報を生成（変数として準備）
            base_market_info = ""
            quote_market_info = ""
            if base_market:
                base_market_info = f"{base_flag} {base_market['name']}: {base_market['open_str']}-{base_market['close_str']} JST"
            if quote_market:
                quote_market_info = f"{quote_flag} {quote_market['name']}: {quote_market['open_str']}-{quote_market['close_str']} JST"

            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"""
                    <div style="padding:6px 4px 0;">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                            <div style="
                                width:36px;height:36px;border-radius:10px;
                                background:linear-gradient(135deg,var(--accent-dark),var(--accent));
                                border:1px solid var(--accent-rgba);
                                display:flex;align-items:center;justify-content:center;
                                font-size:14px;font-weight:800;color:white;
                                flex-shrink:0;
                                box-shadow:0 2px 8px var(--accent-rgba);
                            ">{base}</div>
                            <div style="flex:1;">
                                <div style="font-size:0.75rem;font-weight:700;color:var(--text-muted);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:2px;">
                                    {base_flag if base_flag else ''} {base} / {quote_flag if quote_flag else ''} {quot if quot else ''}
                                </div>
                            </div>
                        </div>
                        <div style="margin-bottom:8px;">
                            <div style="font-size:1.9rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.03em;line-height:1.1;margin-bottom:4px;">{price_fmt}</div>
                            <div style="display:flex;align-items:center;gap:6px;">
                                <span style="font-size:0.85rem;font-weight:600;color:{accent_color_class};">
                                    {arrow} {change_fmt}
                                </span>
                                <span style="font-size:0.75rem;font-weight:600;color:{accent_color_class};opacity:0.8;">
                                    ({sign}{pct_change:.2f}%)
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # チャート表示領域
                    fig = create_mini_line_chart(df, height=110)
                    # チャートデータが存在するか確認
                    if fig.data and len(fig.data) > 0:
                        # yデータの存在確認
                        has_y_data = False
                        for trace in fig.data:
                            if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
                                has_y_data = True
                                break
                        
                        if has_y_data:
                            # チャートを表示（マージンを調整）
                            st.plotly_chart(
                                fig, 
                                use_container_width=True, 
                                config={
                                    'displayModeBar': False,
                                    'staticPlot': False,
                                    'responsive': True
                                }, 
                                key=f"chart_{pair_name}_{i}",
                                height=110
                            )
                        else:
                            # yデータがない場合のプレースホルダー
                            st.markdown(f"""
                            <div style="
                                height:110px;background:var(--bg-card2);
                                border:1px dashed var(--border);
                                border-radius:6px;display:flex;
                                align-items:center;justify-content:center;
                            ">
                                <span style="color:var(--text-muted);font-size:0.75rem;">チャートデータなし ({pair_name})</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # データがない場合のプレースホルダー
                        st.markdown(f"""
                        <div style="
                            height:110px;background:var(--bg-card2);
                            border:1px dashed var(--border);
                            border-radius:6px;display:flex;
                            align-items:center;justify-content:center;
                        ">
                            <span style="color:var(--text-muted);font-size:0.75rem;">チャートなし ({pair_name})</span>
                        </div>
                        """, unsafe_allow_html=True)

                    if st.button("分析する →", key=f"btn_{pair_name}", use_container_width=True):
                        st.session_state.selected_pair = pair_name
                        st.rerun()
    



# ──────────────────────────────────────────────────────────
# 詳細分析
# ──────────────────────────────────────────────────────────
def render_detailed_analysis(available_pairs):
    selected_pair = st.session_state.selected_pair
    
    # selected_pairがNoneの場合は処理を終了（戻るボタンが押された場合など）
    if selected_pair is None:
        return
    
    # selected_pairがavailable_pairsに存在しない場合も処理を終了
    if selected_pair not in available_pairs:
        st.session_state.selected_pair = None
        st.rerun()
        return
    
    ticker = available_pairs[selected_pair]

    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("← 戻る", key="back_button"):
            # 戻るボタンが押された場合、selected_pairをNoneに設定してダッシュボードに戻る
            # 詳細画面用のセッション状態をクリアし、サマリー（一覧）だけが表示されるようにする
            st.session_state.selected_pair = None
            # 詳細画面で使っているウィジェットの状態をクリア（戻ったあと一覧だけ表示するため）
            detail_keys = (
                "sma20", "sma50", "ema20", "bb", "ichimoku", "dmi", "psar", "env",
                "rsi", "macd", "stoch", "psych", "rci", "madev", "hv", "fib",
            )
            for key in detail_keys:
                st.session_state.pop(key, None)
            st.rerun()
            return  # 戻るボタンが押された場合は処理を終了

    # 戻るボタンが押された後の再実行時にもチェック
    if st.session_state.selected_pair is None:
        return

    # 通貨情報を取得
    pair_info = get_pair_market_info(selected_pair)
    base_flag = pair_info.get("base_flag", "")
    quote_flag = pair_info.get("quote_flag", "")
    base_market = pair_info.get("base_market")
    quote_market = pair_info.get("quote_market")
    
    # サブタイトルに国旗を追加
    subtitle_with_flags = f"{base_flag if base_flag else ''} {selected_pair.split('/')[0]} / {quote_flag if quote_flag else ''} {selected_pair.split('/')[1] if '/' in selected_pair else ''} 詳細テクニカル分析"
    render_header(subtitle=subtitle_with_flags)
    
    # 市場時間情報を表示
    if base_market or quote_market:
        st.markdown("### 📅 市場時間情報")
        market_cols = st.columns(2 if (base_market and quote_market) else 1)
        
        col_idx = 0
        if base_market:
            with market_cols[col_idx]:
                st.markdown(f"""
                <div style="
                    padding:12px;background:var(--bg-card);
                    border:1px solid var(--border);border-radius:10px;
                    margin-bottom:10px;
                ">
                    <div style="font-size:0.75rem;font-weight:700;color:var(--text-primary);margin-bottom:6px;">
                        {base_flag} {base_market['name']} ({pair_info.get('base', '')})
                    </div>
                    <div style="font-size:0.7rem;color:var(--text-muted);margin-bottom:4px;">
                        🕐 オープン: {base_market['open_str']} JST
                    </div>
                    <div style="font-size:0.7rem;color:var(--text-muted);">
                        🕐 クローズ: {base_market['close_str']} JST
                    </div>
                    <div style="font-size:0.65rem;color:var(--text-muted);opacity:0.7;margin-top:4px;">
                        タイムゾーン: {base_market['timezone']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            col_idx += 1
        
        if quote_market:
            with market_cols[col_idx]:
                st.markdown(f"""
                <div style="
                    padding:12px;background:var(--bg-card);
                    border:1px solid var(--border);border-radius:10px;
                    margin-bottom:10px;
                ">
                    <div style="font-size:0.75rem;font-weight:700;color:var(--text-primary);margin-bottom:6px;">
                        {quote_flag} {quote_market['name']} ({pair_info.get('quote', '')})
                    </div>
                    <div style="font-size:0.7rem;color:var(--text-muted);margin-bottom:4px;">
                        🕐 オープン: {quote_market['open_str']} JST
                    </div>
                    <div style="font-size:0.7rem;color:var(--text-muted);">
                        🕐 クローズ: {quote_market['close_str']} JST
                    </div>
                    <div style="font-size:0.65rem;color:var(--text-muted);opacity:0.7;margin-top:4px;">
                        タイムゾーン: {quote_market['timezone']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────
    st.sidebar.markdown("""
    <div style="
        padding:14px 0 10px;
        border-bottom:1px solid var(--border);
        margin-bottom:14px;
    ">
        <div style="font-size:0.65rem;font-weight:700;color:var(--accent);
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">
            データパラメータ
        </div>
    </div>
    """, unsafe_allow_html=True)

    timeframes  = get_timeframes()
    selected_tf = st.sidebar.selectbox("⏱ 時間足（期間）", list(timeframes.keys()), index=4)
    period, interval = timeframes[selected_tf]

    st.sidebar.markdown("""
    <div style="
        padding:14px 0 10px;
        border-bottom:1px solid var(--border);
        margin:14px 0;
    ">
        <div style="font-size:0.65rem;font-weight:700;color:var(--accent);
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">
            テクニカル指標
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar.expander("📈 トレンド分析", expanded=True):
        show_sma20    = st.checkbox("SMA (20)", value=True, key="sma20")
        show_sma50    = st.checkbox("SMA (50)",             key="sma50")
        show_ema20    = st.checkbox("EMA (20)",             key="ema20")
        show_bb       = st.checkbox("ボリンジャーバンド (20,2)", key="bb")
        show_ichimoku = st.checkbox("一目均衡表",             key="ichimoku")
        show_dmi      = st.checkbox("DMI / ADX (14)",       key="dmi")
        show_psar     = st.checkbox("パラボリック SAR",       key="psar")
        show_env      = st.checkbox("エンベロープ (20,±2%)", key="env")

    with st.sidebar.expander("📉 オシレーター分析"):
        show_rsi   = st.checkbox("RSI (14)",               key="rsi")
        show_macd  = st.checkbox("MACD",                   key="macd")
        show_stoch = st.checkbox("ストキャスティクス (14,3)", key="stoch")
        show_psych = st.checkbox("サイコロジカルライン (12)", key="psych")
        show_rci   = st.checkbox("RCI (9)",                key="rci")
        show_madev = st.checkbox("移動平均乖離率 (25)",      key="madev")

    with st.sidebar.expander("🔍 フォーメーション分析"):
        st.caption("チャートパターン（目視確認）")
        st.markdown("""
- **ダブルボトム/トップ** — 底値・天井の反転
- **H&S** — トレンド転換の典型
- **三角保合い** — ブレイクアウト前兆
- **ソーサー** — 緩やかな反転
        """)

    with st.sidebar.expander("🕯️ ローソク足分析"):
        st.markdown("""
- **陽線（青）** — 終値 > 始値
- **陰線（赤）** — 終値 < 始値
- **上/下ヒゲ** — 高値/安値の攻防
- **十字線** — 迷いのサイン
        """)

    with st.sidebar.expander("🔢 その他の分析"):
        show_hv  = st.checkbox("ヒストリカル VoL (21)", key="hv")
        show_fib = st.checkbox("フィボナッチ",           key="fib")

    # ── データ取得 ──────────────────────────────────
    with st.spinner(f"{selected_pair} のデータを読み込み中..."):
        df = fetch_fx_data(ticker, period=period, interval=interval, show_warnings=True, pair_name=selected_pair)

    if df.empty:
        st.warning(f"{selected_pair} ({ticker}) のデータが見つかりませんでした。この通貨ペアはYahoo Financeで利用できない可能性があります。")
        st.info("💡 ヒント: 他の通貨ペアを選択するか、メインダッシュボードに戻って利用可能なペアを確認してください。")
        return

    # ── インジケーター適用 ───────────────────────
    overlays_to_show    = []
    oscillators_to_show = []

    if show_sma20:    df = add_sma(df, window=20);            overlays_to_show.append("SMA_20")
    if show_sma50:    df = add_sma(df, window=50);            overlays_to_show.append("SMA_50")
    if show_ema20:    df = add_ema(df, window=20);            overlays_to_show.append("EMA_20")
    if show_bb:
        df = add_bollinger_bands(df, window=20)
        overlays_to_show += ["BB_High_20", "BB_Low_20", "BB_Mid_20"]
    if show_ichimoku:
        df = add_ichimoku(df)
        overlays_to_show += ["Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_SpanA", "Ichimoku_SpanB"]
    if show_dmi:      df = add_dmi(df);                       oscillators_to_show.append("ADX")
    if show_psar:     df = add_parabolic_sar(df);             overlays_to_show.append("PSAR")
    if show_env:
        df = add_envelope(df, window=20, pct=2.0)
        overlays_to_show += ["ENV_Upper_20", "ENV_Lower_20", "ENV_Mid_20"]
    if show_rsi:      df = add_rsi(df, window=14);            oscillators_to_show.append("RSI_14")
    if show_macd:     df = add_macd(df);                      oscillators_to_show.append("MACD")
    if show_stoch:    df = add_stochastics(df);               oscillators_to_show += ["Stoch_K", "Stoch_D"]
    if show_psych:    df = add_psychological_line(df, window=12); oscillators_to_show.append("PsychLine_12")
    if show_rci:      df = add_rci(df, window=9);             oscillators_to_show.append("RCI_9")
    if show_madev:    df = add_ma_deviation(df, window=25);   oscillators_to_show.append("MA_Dev_25")
    if show_hv:       df = add_historical_volatility(df, window=21); oscillators_to_show.append("HV_21")
    if show_fib:
        df = add_fibonacci(df)
        overlays_to_show += ["Fib_236", "Fib_382", "Fib_500", "Fib_618", "Fib_786"]

    # ── メトリクス ────────────────────────────────
    st.markdown("### 概要")
    if len(df) >= 2:
        latest   = df.iloc[-1]
        previous = df.iloc[-2]
        price_change = latest['Close'] - previous['Close']
        pct_change   = (price_change / previous['Close']) * 100
        high_low     = latest['High'] - latest['Low']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現在価格",             f"{latest['Close']:.4f}",  f"{price_change:+.4f} ({pct_change:+.2f}%)")
        c2.metric("24H 高値",             f"{latest['High']:.4f}")
        c3.metric("24H 安値",             f"{latest['Low']:.4f}")
        c4.metric("ボラティリティ（H-L）", f"{high_low:.4f}")
    else:
        st.info("データが足りません")

    # ── チャート ──────────────────────────────────
    st.markdown("### インタラクティブチャート")
    chart_html = render_google_candlestick_chart(df, selected_pair, overlays=overlays_to_show)
    components.html(chart_html, height=630, scrolling=False)

    # ── シグナルダッシュボード ───────────────────────
    signals = compute_signals(df, overlays_to_show, oscillators_to_show)
    total   = signals['total']
    has_any = (total['buy'] + total['neutral'] + total['sell']) > 0
    if has_any:
        st.markdown("### 🎯 テクニカル シグナル")
        gauge_html  = render_signal_dashboard(signals)
        n_osc       = len(signals['oscillator']['rows'])
        n_ma        = len(signals['ma']['rows'])
        gauge_height = 700 + (n_osc + n_ma) * 40
        components.html(gauge_html, height=gauge_height, scrolling=False)

    # ── 生データ ───────────────────────────────
    with st.expander("📋 生データ / シグナルを表示"):
        st.dataframe(df.tail(50).sort_index(ascending=False), use_container_width=True)


# ──────────────────────────────────────────────────────────
# メイン関数
# ──────────────────────────────────────────────────────────
def main():
    # ─ セッション状態の初期化 ─
    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True  # デフォルトはダークモード

    # ─ テーマCSS適用 ─
    # dark_modeがTrueならlight_modeはFalse、dark_modeがFalseならlight_modeはTrue
    apply_theme_css(light_mode=not st.session_state.dark_mode)
    
    # ─ テーマ適用スクリプト（CSS変数を直接設定する方法） ─
    light_mode_value = not st.session_state.dark_mode
    
    # ライトモードのCSS変数値を定義
    if light_mode_value:
        theme_vars = {
            '--bg-base': '#f0f4f8',
            '--bg-card': '#ffffff',
            '--bg-card2': '#f8fafc',
            '--bg-sidebar': '#f8fafc',
            '--border': '#dde3ea',
            '--border2': 'rgba(200,210,220,0.9)',
            '--text-primary': '#1a202c',
            '--text-muted': '#64748b',
            '--text-sub': '#4a5568',
            '--accent': '#2563eb',
            '--accent2': '#3b82f6',
            '--accent-dark': '#1d4ed8',
            '--green': '#16a34a',
            '--red': '#dc2626',
            '--dot-color': 'rgba(0,0,0,0.04)',
            '--hero-grad1': 'rgba(37,99,235,0.12)',
            '--hero-grad2': 'rgba(22,163,74,0.06)',
            '--hero-grid': 'rgba(37,99,235,0.05)',
            '--shadow': 'rgba(0,0,0,0.1)',
            '--shadow2': 'rgba(0,0,0,0.06)',
            '--border-rgba': 'rgba(200,210,220,0.7)',
            '--green-rgba': 'rgba(22,163,74,0.3)',
            '--green-rgba-light': 'rgba(22,163,74,0.1)',
            '--accent-rgba': 'rgba(37,99,235,0.2)',
            '--accent-rgba-light': 'rgba(37,99,235,0.06)',
            '--accent-dark-rgba': 'rgba(29,78,216,0.15)',
            '--accent-dark-rgba-light': 'rgba(29,78,216,0.3)',
            '--accent2-rgba': 'rgba(59,130,246,0.2)',
            '--bg-card-rgba': 'rgba(248,250,252,0.9)',
            '--bg-overlay': 'rgba(0,0,0,0.04)',
            '--hero-bg-start': '#f0f4f8',
            '--hero-bg-end': '#e2e8f0',
            '--warning': '#f59e0b',
            '--warning-rgba': 'rgba(245,158,11,0.1)',
            '--warning-rgba-border': 'rgba(245,158,11,0.35)',
            '--green-badge-bg': 'rgba(22,163,74,0.12)',
            '--red-badge-bg': 'rgba(220,38,38,0.12)',
        }
    else:
        theme_vars = {
            '--bg-base': '#080c14',
            '--bg-card': '#161b22',
            '--bg-card2': '#0d1117',
            '--bg-sidebar': '#0d1117',
            '--border': '#21262d',
            '--border2': 'rgba(48,54,61,0.8)',
            '--text-primary': '#e6edf3',
            '--text-muted': '#8b949e',
            '--text-sub': '#c9d1d9',
            '--accent': '#388bfd',
            '--accent2': '#58a6ff',
            '--accent-dark': '#1f6feb',
            '--green': '#3fb950',
            '--red': '#f85149',
            '--dot-color': 'rgba(255,255,255,0.04)',
            '--hero-grad1': 'rgba(31,111,235,0.22)',
            '--hero-grad2': 'rgba(63,185,80,0.12)',
            '--hero-grid': 'rgba(56,139,253,0.06)',
            '--shadow': 'rgba(0,0,0,0.5)',
            '--shadow2': 'rgba(0,0,0,0.4)',
            '--border-rgba': 'rgba(48,54,61,0.7)',
            '--green-rgba': 'rgba(63,185,80,0.3)',
            '--green-rgba-light': 'rgba(63,185,80,0.1)',
            '--accent-rgba': 'rgba(56,139,253,0.2)',
            '--accent-rgba-light': 'rgba(56,139,253,0.06)',
            '--accent-dark-rgba': 'rgba(31,111,235,0.22)',
            '--accent-dark-rgba-light': 'rgba(31,111,235,0.45)',
            '--accent2-rgba': 'rgba(88,166,255,0.2)',
            '--bg-card-rgba': 'rgba(22,27,34,0.9)',
            '--bg-overlay': 'rgba(255,255,255,0.04)',
            '--hero-bg-start': '#0d1823',
            '--hero-bg-end': '#0b0f18',
            '--warning': '#f0a500',
            '--warning-rgba': 'rgba(240,165,0,0.1)',
            '--warning-rgba-border': 'rgba(240,165,0,0.35)',
            '--green-badge-bg': 'rgba(63,185,80,0.14)',
            '--red-badge-bg': 'rgba(248,81,73,0.14)',
        }
    
    # 動的にCSS変数を上書きする<style>タグを生成（JavaScriptを使わず直接CSSを適用）
    css_vars_lines = [f'    {k}: {v};' for k, v in theme_vars.items()]
    css_vars_content = '\n'.join(css_vars_lines)
    
    # CSS文字列を生成（!importantを追加して確実に適用）
    css_content = f""":root {{
{css_vars_content}
}}
.stApp {{
{css_vars_content}
}}
* {{
    transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
}}"""
    
    # 直接<style>タグとして出力（JavaScript不要）
    st.markdown(f"""
    <style id="dynamic-theme">
{css_content}
    </style>
    """, unsafe_allow_html=True)

    # ─ サイドバーのテーマトグル ─
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 10px;">
            <div style="font-size:0.65rem;font-weight:700;color:var(--accent);
                        letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">
                🎨 テーマ設定
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ダーク/ライトモードのトグルボタン
        theme_label = "☀️ ライトモードに切替" if st.session_state.dark_mode else "🌙 ダークモードに切替"
        if st.button(theme_label, use_container_width=True, key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        # 現在のモード表示
        mode_text = "🌙 ダークモード" if st.session_state.dark_mode else "☀️ ライトモード"
        st.caption(f"現在: {mode_text}")
        st.markdown("<hr style='border-color:var(--border);margin:10px 0 14px;'>", unsafe_allow_html=True)

        # ─ データ遅延の注意書き ─
        st.markdown("""
        <div style="padding:8px 12px;border-radius:8px;
            background:var(--warning-rgba);border:1px solid var(--warning-rgba-border);
            margin-bottom:14px;">
            <div style="font-size:0.72rem;font-weight:700;color:var(--warning);margin-bottom:3px;">
                ⚠️ データについて
            </div>
            <div style="font-size:0.7rem;color:var(--text-muted);line-height:1.4;">
                yfinance経由のデータは<br>
                約<strong style="color:var(--warning);">15〜20分</strong>の遅延があります。<br>
                リアルタイムデータではありません。
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─ ページ切替 ─
    available_pairs = get_available_pairs()

    # メインコンテンツを1つのコンテナにまとめ、詳細→一覧に戻ったときに
    # 詳細のサマリーが残らないようにする
    main_container = st.container()
    with main_container:
        if st.session_state.selected_pair is None:
            render_dashboard(available_pairs)
        else:
            render_detailed_analysis(available_pairs)

if __name__ == "__main__":
    main()
