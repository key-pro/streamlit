"""
主要通貨＋一部マイナー通貨の通貨ペアを自動生成するモジュール（クロス円を優先）
"""
from typing import Dict, List


# 優先度の高い通貨（メジャー通貨）
PRIORITY_CURRENCIES = [
    "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD",
]

# 実運用でよく使うマイナー通貨（ホワイトリスト）
# ※ ここに含まれない通貨とのペアはダッシュボードでは扱わない
MINOR_CURRENCIES = [
    "SEK",  # スウェーデン・クローナ
    "NOK",  # ノルウェー・クローネ
    "DKK",  # デンマーク・クローネ
    "HKD",  # 香港ドル
    "CNH",  # オフショア人民元
    "SGD",  # シンガポール・ドル
    "MXN",  # メキシコペソ
    "TRY",  # トルコリラ
    "ZAR",  # 南アフリカランド
]


def generate_all_pairs() -> Dict[str, str]:
    """
    主要通貨＋一部マイナー通貨の通貨ペアを生成する（クロス円を優先的に生成）
    
    Returns:
        Dict[str, str]: 通貨ペア名とYahoo Financeティッカーの辞書
    """
    pairs: Dict[str, str] = {}
    
    # すべての通貨を統合（メジャー + 一部マイナー）
    all_currencies: List[str] = PRIORITY_CURRENCIES + MINOR_CURRENCIES
    
    # まず、クロス円ペアを優先的に生成（JPYを含むすべてのペア）
    jpy_pairs = []
    other_pairs = []
    
    for i, base in enumerate(all_currencies):
        for quote in all_currencies[i + 1 :]:
            if base == quote:
                continue

            # JPY を含むペアは常に「XXX/JPY」形式に統一する
            # 例: JPY/AUD ではなく AUD/JPY を生成
            b, q = base, quote
            if b == "JPY" and q != "JPY":
                b, q = q, b
            
            pair_name = f"{b}/{q}"
            # Yahoo Financeのティッカー形式に合わせる
            # 通常は {base}{quote}=X だが、USD/JPY のみ "JPY=X" を使用する
            if b == "USD" and q == "JPY":
                ticker = "JPY=X"
            else:
                ticker = f"{b}{q}=X"
            
            if b == "JPY" or q == "JPY":
                jpy_pairs.append((pair_name, ticker))
            else:
                other_pairs.append((pair_name, ticker))
    
    # クロス円ペアを先に追加（これによりクロス円が一番多くなる）
    for pair_name, ticker in jpy_pairs:
        pairs[pair_name] = ticker
    
    # その他のペアもすべて追加（件数制限なし）
    for pair_name, ticker in other_pairs:
        pairs[pair_name] = ticker
    
    return pairs


def get_available_pairs_grouped() -> dict:
    """
    通貨ペアをカテゴリ別にグループ化した辞書を返す（メジャー通貨のみ）。
    """
    all_pairs = generate_all_pairs()
    
    # カテゴリ別に分類（メジャー通貨のみ）
    major_pairs = []
    cross_yen = []
    euro_cross = []
    pound_cross = []
    oceania = []
    others = []
    
    for pair_name in all_pairs.keys():
        base, quote = pair_name.split("/")
        
        # メジャーペア（USD, EUR, GBP, AUD, NZD, CAD, CHFを含む）
        if base in PRIORITY_CURRENCIES and quote in PRIORITY_CURRENCIES:
            major_pairs.append(pair_name)
        
        # クロス円（JPYを含むペア、USD/JPYも含む）
        if quote == "JPY" or base == "JPY":
            cross_yen.append(pair_name)
        # ユーロクロス
        elif base == "EUR" or quote == "EUR":
            euro_cross.append(pair_name)
        # ポンドクロス
        elif base == "GBP" or quote == "GBP":
            pound_cross.append(pair_name)
        # オセアニア（AUDまたはNZDを含むペア）
        elif base in ["AUD", "NZD"] or quote in ["AUD", "NZD"]:
            oceania.append(pair_name)
        else:
            others.append(pair_name)
    
    return {
        "メジャー": sorted(major_pairs),
        "クロス円": sorted(cross_yen),
        "ユーロ": sorted(euro_cross),
        "ポンド": sorted(pound_cross),
        "オセアニア": sorted(oceania),
        "その他": sorted(others),
    }
