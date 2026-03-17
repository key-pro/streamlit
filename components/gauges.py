import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# 1. シグナル計算
# ──────────────────────────────────────────────

def sig_label(sig: int) -> str:
    if sig == 1:  return '買い'
    if sig == -1: return '売り'
    return '中立'


def compute_signals(df: pd.DataFrame, overlays: list, oscillators: list) -> dict:
    if df.empty or len(df) < 2:
        blank = {'buy': 0, 'neutral': 0, 'sell': 0, 'rows': []}
        return {'oscillator': blank, 'ma': blank, 'total': {'buy': 0, 'neutral': 0, 'sell': 0}}

    last = df.iloc[-1]
    close = float(last['Close'])

    osc_rows = []
    ma_rows  = []

    def row(name, val, sig):
        return {'name': name, 'value': round(float(val), 3) if val is not None and not pd.isna(val) else None, 'signal': sig_label(sig)}

    # ── Oscillators ────────────────────
    for col in [c for c in df.columns if c.startswith('RSI_')]:
        v = last[col]; sig = 1 if (not pd.isna(v) and v < 30) else (-1 if (not pd.isna(v) and v > 70) else 0)
        osc_rows.append(row(col, v, sig))

    if 'Stoch_K' in df.columns:
        v = last['Stoch_K']; sig = 1 if (not pd.isna(v) and v < 20) else (-1 if (not pd.isna(v) and v > 80) else 0)
        osc_rows.append(row('ストキャスティクス %K', v, sig))

    for col in [c for c in df.columns if c.startswith('RCI_')]:
        v = last[col]; sig = 1 if (not pd.isna(v) and v < -80) else (-1 if (not pd.isna(v) and v > 80) else 0)
        osc_rows.append(row(col, v, sig))

    for col in [c for c in df.columns if c.startswith('PsychLine_')]:
        v = last[col]; sig = 1 if (not pd.isna(v) and v < 25) else (-1 if (not pd.isna(v) and v > 75) else 0)
        osc_rows.append(row('サイコロジカルライン', v, sig))

    for col in [c for c in df.columns if c.startswith('MA_Dev_')]:
        v = last[col]; sig = 1 if (not pd.isna(v) and v < -3) else (-1 if (not pd.isna(v) and v > 3) else 0)
        osc_rows.append(row('移動平均乖離率', v, sig))

    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        m, s = last['MACD'], last['MACD_Signal']
        if not pd.isna(m) and not pd.isna(s):
            osc_rows.append(row('MACD', m, 1 if m > s else -1))

    if 'ADX' in df.columns and 'DI_Plus' in df.columns and 'DI_Minus' in df.columns:
        adx, dip, dim = last['ADX'], last['DI_Plus'], last['DI_Minus']
        if not pd.isna(dip) and not pd.isna(dim):
            osc_rows.append(row('DMI / ADX', adx, 1 if dip > dim else -1))

    for col in [c for c in df.columns if c.startswith('HV_')]:
        osc_rows.append(row('ヒストリカルVol.', last[col], 0))

    # ── Moving Averages / Trend ─────────
    for col in [c for c in df.columns if c.startswith('SMA_') or c.startswith('EMA_')]:
        v = last[col]
        if not pd.isna(v): ma_rows.append(row(col, v, 1 if close > float(v) else -1))

    for col, name in [('Ichimoku_Tenkan', '一目(転換線)'), ('Ichimoku_Kijun', '一目(基準線)')]:
        if col in df.columns:
            v = last[col]
            if not pd.isna(v): ma_rows.append(row(name, v, 1 if close > float(v) else -1))

    if 'BB_Mid_20' in df.columns:
        v = last['BB_Mid_20']
        if not pd.isna(v): ma_rows.append(row('BB(Mid)', v, 1 if close > float(v) else -1))

    for col in [c for c in df.columns if c.startswith('ENV_Mid_')]:
        v = last[col]
        if not pd.isna(v): ma_rows.append(row('エンベロープ(Mid)', v, 1 if close > float(v) else -1))

    if 'PSAR' in df.columns:
        v = last['PSAR']
        if not pd.isna(v): ma_rows.append(row('パラボリック SAR', v, 1 if close > float(v) else -1))

    def tally(rows):
        buy  = sum(1 for r in rows if r['signal'] == '買い')
        sell = sum(1 for r in rows if r['signal'] == '売り')
        return buy, len(rows) - buy - sell, sell

    ob, on, os_ = tally(osc_rows)
    mb, mn, ms  = tally(ma_rows)
    return {
        'oscillator': {'buy': ob, 'neutral': on, 'sell': os_, 'rows': osc_rows},
        'ma':         {'buy': mb, 'neutral': mn, 'sell': ms,  'rows': ma_rows},
        'total':      {'buy': ob + mb, 'neutral': on + mn, 'sell': os_ + ms},
    }


# ──────────────────────────────────────────────
# 2. ゲージHTML（SVG + CSSアニメーション）
# ──────────────────────────────────────────────

def _gauge_html(buy: int, neutral: int, sell: int, uid: str, size: int = 200) -> str:
    """アニメーション付き半円ゲージ（ダークテーマ）を返す。"""
    total = buy + neutral + sell
    score = ((buy - sell) / total * 0.5 + 0.5) if total > 0 else 0.5
    score = max(0.0, min(1.0, score))

    arc_deg   = -180 + score * 180
    css_angle = arc_deg + 90

    if score < 0.2:   verdict, vc = '強い売り', '#f85149'
    elif score < 0.4: verdict, vc = '売り',     '#ff7b72'
    elif score < 0.6: verdict, vc = '中立',     '#8b949e'
    elif score < 0.8: verdict, vc = '買い',     '#58a6ff'
    else:             verdict, vc = '強い買い',  '#388bfd'

    cx, cy   = size / 2, size / 2
    r_out    = size * 0.42
    r_in     = size * 0.28
    needle_r = r_in * 0.85

    def arc_segment(start_deg, end_deg, color):
        def pt(deg, r):
            rad = deg * np.pi / 180
            return cx + r * np.cos(rad), cy + r * np.sin(rad)
        x1o, y1o = pt(start_deg, r_out)
        x2o, y2o = pt(end_deg,   r_out)
        x1i, y1i = pt(start_deg, r_in)
        x2i, y2i = pt(end_deg,   r_in)
        large = 1 if abs(end_deg - start_deg) > 180 else 0
        return (
            f'<path d="M{x1o:.1f},{y1o:.1f} A{r_out:.1f},{r_out:.1f} 0 {large},1 {x2o:.1f},{y2o:.1f} '
            f'L{x2i:.1f},{y2i:.1f} A{r_in:.1f},{r_in:.1f} 0 {large},0 {x1i:.1f},{y1i:.1f} Z" '
            f'fill="{color}"/>'
        )

    segs = [
        (-180, -144, '#c62828'),
        (-144, -108, '#ef5350'),
        (-108,  -72, '#374151'),
        ( -72,  -36, '#3b82f6'),
        ( -36,    0, '#1d4ed8'),
    ]
    arcs  = ''.join(arc_segment(s, e, c) for s, e, c in segs)
    svg_h = int(size * 0.76)

    return f"""
<div style="text-align:center;position:relative;display:inline-block;">
  <svg width="{size}" height="{svg_h}" viewBox="0 0 {size} {svg_h}"
       xmlns="http://www.w3.org/2000/svg" overflow="visible">
    <defs>
      <style>
        .needle-{uid} {{
          transform-origin: {cx:.1f}px {cy:.1f}px;
          animation: swing-{uid} 5s cubic-bezier(.4,0,.2,1) infinite;
        }}
        @keyframes swing-{uid} {{
          /* 左端からスタート */
          0%    {{ transform: rotate(-90deg); }}
          /* 最終位置へバウンド */
          18%   {{ transform: rotate({css_angle + 14:.1f}deg); }}
          22%   {{ transform: rotate({css_angle - 6:.1f}deg); }}
          25%   {{ transform: rotate({css_angle + 3:.1f}deg); }}
          28%   {{ transform: rotate({css_angle:.1f}deg); }}
          /* 最終位置で保持 (~3秒) */
          88%   {{ transform: rotate({css_angle:.1f}deg); }}
          /* 瞬時にリセット → 次のループへ */
          88.1% {{ transform: rotate(-90deg); }}
          100%  {{ transform: rotate(-90deg); }}
        }}
      </style>
    </defs>
    {arcs}
    <g class="needle-{uid}">
      <line x1="{cx:.1f}" y1="{cy:.1f}"
            x2="{cx:.1f}" y2="{cy - needle_r:.1f}"
            stroke="#e6edf3" stroke-width="2.5" stroke-linecap="round"/>
    </g>
    <circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#e6edf3"/>
  </svg>
  <div style="font-size:15px;font-weight:700;color:{vc};margin-top:-6px;letter-spacing:.03em;">{verdict}</div>
  <div style="display:flex;justify-content:space-around;width:{size}px;margin-top:6px;">
    <div style="text-align:center;">
      <div style="font-size:10px;color:#f85149;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">売り</div>
      <div style="font-size:16px;font-weight:700;color:#f85149;">{sell}</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:10px;color:#8b949e;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">中立</div>
      <div style="font-size:16px;font-weight:700;color:#8b949e;">{neutral}</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:10px;color:#58a6ff;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">買い</div>
      <div style="font-size:16px;font-weight:700;color:#58a6ff;">{buy}</div>
    </div>
  </div>
</div>
"""


# ──────────────────────────────────────────────
# 3. 完全なダッシュボードHTML（ダークテーマ）
# ──────────────────────────────────────────────

def render_signal_dashboard(signals: dict) -> str:
    tot = signals['total']
    osc = signals['oscillator']
    ma  = signals['ma']

    g_total = _gauge_html(tot['buy'], tot['neutral'], tot['sell'], 'total', size=260)
    g_osc   = _gauge_html(osc['buy'], osc['neutral'], osc['sell'], 'osc',   size=190)
    g_ma    = _gauge_html(ma['buy'],  ma['neutral'],  ma['sell'],  'ma',    size=190)

    def badge(sig):
        if sig == '買い':
            return '<span style="background:rgba(56,139,253,.18);color:#58a6ff;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;border:1px solid rgba(56,139,253,.35);">買い</span>'
        if sig == '売り':
            return '<span style="background:rgba(248,81,73,.18);color:#f85149;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;border:1px solid rgba(248,81,73,.35);">売り</span>'
        return '<span style="background:rgba(139,148,158,.12);color:#8b949e;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid rgba(139,148,158,.25);">中立</span>'

    def table_rows(rows):
        if not rows:
            return '<tr><td colspan="3" style="color:#8b949e;text-align:center;padding:14px;font-size:12px;">指標未選択</td></tr>'
        out = ''
        for r in rows:
            sig = r['signal']
            val = str(r['value']) if r['value'] is not None else '—'
            out += f'''
<tr style="border-bottom:1px solid #21262d;transition:background .15s;" onmouseover="this.style.background='#21262d'" onmouseout="this.style.background='transparent'">
  <td style="padding:7px 12px;font-size:12px;color:#c9d1d9;">{r["name"]}</td>
  <td style="padding:7px 12px;font-size:12px;color:#8b949e;text-align:right;font-variant-numeric:tabular-nums;">{val}</td>
  <td style="padding:7px 12px;text-align:center;">{badge(sig)}</td>
</tr>'''
        return out

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing:border-box;margin:0;padding:0; }}
  body {{
    font-family:'Inter','Hiragino Sans',sans-serif;
    background:#0d1117;
    color:#e6edf3;
    padding:14px;
  }}
  .card {{
    background:#161b22;
    border:1px solid #30363d;
    border-radius:14px;
    padding:18px 16px 16px;
    box-shadow:0 4px 24px rgba(0,0,0,.4);
  }}
  .section-label {{
    font-size:11px;font-weight:700;color:#388bfd;
    text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;
  }}
  .row {{ display:flex;gap:14px; }}
  .half {{ flex:1;min-width:0; }}
  table {{ width:100%;border-collapse:collapse;margin-top:12px; }}
  thead tr {{ border-bottom:1px solid #30363d; }}
  th {{
    padding:6px 12px;font-size:10px;font-weight:600;
    color:#8b949e;text-transform:uppercase;letter-spacing:.08em;
    text-align:left;
  }}
  th:nth-child(2) {{ text-align:right; }}
  th:last-child   {{ text-align:center; }}
</style>
</head><body>

  <div class="card" style="text-align:center;margin-bottom:14px;">
    <div class="section-label">📊 テクニカル シグナル サマリー</div>
    {g_total}
  </div>

  <div class="row">
    <div class="half card" style="text-align:center;">
      <div class="title">📉 オシレーター</div>
      {g_osc}
      <table>
        <thead><tr>
          <th>名前</th><th style="text-align:right;">値</th><th style="text-align:center;">アクション</th>
        </tr></thead>
        <tbody>{table_rows(osc['rows'])}</tbody>
      </table>
    </div>
    <div class="half card" style="text-align:center;">
      <div class="title">📈 移動平均</div>
      {g_ma}
      <table>
        <thead><tr>
          <th>名前</th><th style="text-align:right;">値</th><th style="text-align:center;">アクション</th>
        </tr></thead>
        <tbody>{table_rows(ma['rows'])}</tbody>
      </table>
    </div>
  </div>

</body></html>"""
    return html
