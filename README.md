# FX 為替マーケットダッシュボード（FX版）

高機能FX（外国為替）テクニカル分析Webアプリケーション。Streamlitで構築。  
※ 米国株版は `main_us_stocks.py` で起動できます。

---

## 📋 概要

`yfinance` から為替レートデータを取得し、複数のテクニカル指標とインタラクティブなチャートで分析するダッシュボードです。

> ⚠️ **注意:** データは `yfinance` 経由のため、約 **15〜20分** の遅延があります。リアルタイムデータではありません。

---

## 🚀 起動方法

### 前提条件
- Python 3.9 以上
- pip

### インストール

```bash
cd /Users/key/Desktop/streamlit
pip install -r requirements.txt
```

### 起動

**FX版（為替）**
```bash
streamlit run main.py
```

**米国株版**
```bash
streamlit run main_us_stocks.py
```

FX版起動時、ブラウザで自動的に `http://localhost:8501` が開きます。

---

## ✨ 主な機能

### ダッシュボード（トップページ）
- **26通貨ペア** のタイル一覧表示（グループ別）
- 各タイルに価格・変動率・スパークライン表示
- 「分析する」ボタンで詳細分析へ遷移

### 詳細分析ページ
| カテゴリ | 指標 |
|---|---|
| トレンド | SMA(20/50), EMA(20), ボリンジャーバンド, 一目均衡表, DMI/ADX, パラボリックSAR, エンベロープ |
| オシレーター | RSI(14), MACD, ストキャスティクス, サイコロジカルライン, RCI, 移動平均乖離率, ヒストリカルVoL |
| フォーメーション | ダブルボトム/トップ, H&S, 三角保合い（目視参考） |
| その他 | ローソク足分析, フィボナッチリトレースメント |

### テーマ機能
- **ダーク / ライトモード** のトグル切替（サイドバー内）

---

## 🌐 対応通貨ペア（26ペア）

| カテゴリ | ペア |
|---|---|
| メジャー | USD/JPY, EUR/USD, GBP/USD, USD/CHF, USD/CAD, AUD/USD, NZD/USD |
| クロス円 | EUR/JPY, GBP/JPY, AUD/JPY, NZD/JPY, CHF/JPY, CAD/JPY, HKD/JPY |
| ユーロクロス | EUR/GBP, EUR/CHF, EUR/AUD, EUR/CAD, EUR/NZD |
| その他 | GBP/CHF, AUD/NZD, AUD/CAD, GBP/AUD, GBP/CAD, GBP/NZD |

---

## 🗂 ファイル構成

```
streamlit/
├── main.py                  # メインアプリ（UI・ルーティング）
├── requirements.txt         # 依存ライブラリ
├── README.md                # 本ファイル
├── utils/
│   ├── data_fetcher.py      # yfinanceデータ取得・ペア定義
│   └── indicators.py        # テクニカル指標計算ロジック
└── components/
    ├── charts.py            # ローソク足・スパークラインチャート
    └── gauges.py            # シグナルゲージ（TradingView風）
```

---

## 🛠 使用ライブラリ

| ライブラリ | 用途 |
|---|---|
| `streamlit` | Webフレームワーク |
| `yfinance` | 為替データ取得（約15〜20分遅延） |
| `pandas` | データ処理 |
| `plotly` | インタラクティブチャート |
| `numpy` | 数値計算 |

---

## ⚙️ 設定・カスタマイズ

### 通貨ペアの追加

`utils/data_fetcher.py` の `get_available_pairs_grouped()` 関数に追記します。

```python
"その他": {
    "MXN/JPY": "MXNJPY=X",   # 例: メキシコペソ
}
```

### テーマカラーのカスタマイズ

`main.py` の `apply_theme_css()` 関数内の CSS 変数（`:root` / `.light-mode`）を編集します。

---

## 📝 制限事項

- データ遅延: yfinanceは約15〜20分の遅延があります
- リアルタイム更新: 自動更新はありません（手動リロードが必要）
- 市場閉鎖時: 週末・祝日はデータが更新されない場合があります

---

*Built with ❤️ using Streamlit & Python*
