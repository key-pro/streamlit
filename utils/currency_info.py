"""
通貨情報と市場時間を管理するモジュール
"""
from typing import Dict, Tuple, Optional
import datetime


# 通貨コードと国旗のマッピング（拡張版）
CURRENCY_FLAGS = {
    # メジャー通貨
    "USD": "🇺🇸",  # アメリカ
    "EUR": "🇪🇺",  # ヨーロッパ
    "JPY": "🇯🇵",  # 日本
    "GBP": "🇬🇧",  # イギリス
    "AUD": "🇦🇺",  # オーストラリア
    "CAD": "🇨🇦",  # カナダ
    "CHF": "🇨🇭",  # スイス
    "NZD": "🇳🇿",  # ニュージーランド
    # 北欧
    "SEK": "🇸🇪",  # スウェーデン
    "NOK": "🇳🇴",  # ノルウェー
    "DKK": "🇩🇰",  # デンマーク
    "ISK": "🇮🇸",  # アイスランド
    "FIM": "🇫🇮",  # フィンランド（旧通貨、EURに統合）
    # アジア
    "CNH": "🇨🇳",  # 中国（オフショア）
    "CNY": "🇨🇳",  # 中国（オンスhore）
    "HKD": "🇭🇰",  # 香港
    "SGD": "🇸🇬",  # シンガポール
    "INR": "🇮🇳",  # インド
    "KRW": "🇰🇷",  # 韓国
    "TWD": "🇹🇼",  # 台湾
    "THB": "🇹🇭",  # タイ
    "MYR": "🇲🇾",  # マレーシア
    "IDR": "🇮🇩",  # インドネシア
    "PHP": "🇵🇭",  # フィリピン
    "VND": "🇻🇳",  # ベトナム
    "PKR": "🇵🇰",  # パキスタン
    "BDT": "🇧🇩",  # バングラデシュ
    "LKR": "🇱🇰",  # スリランカ
    "MMK": "🇲🇲",  # ミャンマー
    "KHR": "🇰🇭",  # カンボジア
    "LAK": "🇱🇦",  # ラオス
    "BND": "🇧🇳",  # ブルネイ
    # 中東
    "AED": "🇦🇪",  # アラブ首長国連邦
    "SAR": "🇸🇦",  # サウジアラビア
    "QAR": "🇶🇦",  # カタール
    "KWD": "🇰🇼",  # クウェート
    "BHD": "🇧🇭",  # バーレーン
    "OMR": "🇴🇲",  # オマーン
    "JOD": "🇯🇴",  # ヨルダン
    "ILS": "🇮🇱",  # イスラエル
    "EGP": "🇪🇬",  # エジプト
    "LBP": "🇱🇧",  # レバノン
    "IQD": "🇮🇶",  # イラク
    "IRR": "🇮🇷",  # イラン
    # アフリカ
    "ZAR": "🇿🇦",  # 南アフリカ
    "NGN": "🇳🇬",  # ナイジェリア
    "KES": "🇰🇪",  # ケニア
    "GHS": "🇬🇭",  # ガーナ
    "ETB": "🇪🇹",  # エチオピア
    "TZS": "🇹🇿",  # タンザニア
    "UGX": "🇺🇬",  # ウガンダ
    "MAD": "🇲🇦",  # モロッコ
    "TND": "🇹🇳",  # チュニジア
    "DZD": "🇩🇿",  # アルジェリア
    # ラテンアメリカ
    "MXN": "🇲🇽",  # メキシコ
    "BRL": "🇧🇷",  # ブラジル
    "ARS": "🇦🇷",  # アルゼンチン
    "CLP": "🇨🇱",  # チリ
    "COP": "🇨🇴",  # コロンビア
    "PEN": "🇵🇪",  # ペルー
    "UYU": "🇺🇾",  # ウルグアイ
    "PYG": "🇵🇾",  # パラグアイ
    "BOB": "🇧🇴",  # ボリビア
    "VES": "🇻🇪",  # ベネズエラ
    "GTQ": "🇬🇹",  # グアテマラ
    "HNL": "🇭🇳",  # ホンジュラス
    "NIO": "🇳🇮",  # ニカラグア
    "CRC": "🇨🇷",  # コスタリカ
    "PAB": "🇵🇦",  # パナマ
    "DOP": "🇩🇴",  # ドミニカ共和国
    "JMD": "🇯🇲",  # ジャマイカ
    "TTD": "🇹🇹",  # トリニダード・トバゴ
    "BBD": "🇧🇧",  # バルバドス
    # ヨーロッパ（その他）
    "PLN": "🇵🇱",  # ポーランド
    "HUF": "🇭🇺",  # ハンガリー
    "CZK": "🇨🇿",  # チェコ
    "RON": "🇷🇴",  # ルーマニア
    "BGN": "🇧🇬",  # ブルガリア
    "HRK": "🇭🇷",  # クロアチア（EURに統合）
    "RSD": "🇷🇸",  # セルビア
    "BAM": "🇧🇦",  # ボスニア・ヘルツェゴビナ
    "MKD": "🇲🇰",  # 北マケドニア
    "ALL": "🇦🇱",  # アルバニア
    "MDL": "🇲🇩",  # モルドバ
    "UAH": "🇺🇦",  # ウクライナ
    "BYN": "🇧🇾",  # ベラルーシ
    "RUB": "🇷🇺",  # ロシア
    "KZT": "🇰🇿",  # カザフスタン
    "UZS": "🇺🇿",  # ウズベキスタン
    "GEL": "🇬🇪",  # ジョージア
    "AMD": "🇦🇲",  # アルメニア
    "AZN": "🇦🇿",  # アゼルバイジャン
    # トルコ・中東アジア
    "TRY": "🇹🇷",  # トルコ
    # オセアニア
    "FJD": "🇫🇯",  # フィジー
    "PGK": "🇵🇬",  # パプアニューギニア
    "WST": "🇼🇸",  # サモア
    "TOP": "🇹🇴",  # トンガ
    "VUV": "🇻🇺",  # バヌアツ
    "SBD": "🇸🇧",  # ソロモン諸島
    # その他
    "XAU": "🥇",  # 金
    "XAG": "🥈",  # 銀
    "XPT": "💎",  # プラチナ
    "XPD": "💠",  # パラジウム
}


def get_currency_flag(currency_code: str) -> str:
    """
    通貨コードから国旗の絵文字を取得
    
    Args:
        currency_code: 通貨コード（例: "USD", "JPY"）
    
    Returns:
        str: 国旗の絵文字、見つからない場合は空文字列
    """
    return CURRENCY_FLAGS.get(currency_code.upper(), "")


def get_pair_flags(pair_name: str) -> Tuple[str, str]:
    """
    通貨ペア名から両方の通貨の国旗を取得
    
    Args:
        pair_name: 通貨ペア名（例: "USD/JPY"）
    
    Returns:
        Tuple[str, str]: (ベース通貨の国旗, クォート通貨の国旗)
    """
    if "/" in pair_name:
        base, quote = pair_name.split("/")
        return get_currency_flag(base), get_currency_flag(quote)
    return "", ""


# 主要市場のオープン/クローズ時間（日本時間）
MARKET_HOURS = {
    "USD": {
        "name": "ニューヨーク",
        "open_jst": (21, 0),  # 標準時は22:00、サマータイムは21:00
        "close_jst": (6, 0),  # 標準時は7:00、サマータイムは6:00
        "timezone": "EST/EDT",
        "dst": True  # サマータイムあり
    },
    "EUR": {
        "name": "ロンドン/フランクフルト",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),  # 標準時は2:00、サマータイムは1:00
        "timezone": "GMT/BST",
        "dst": True  # サマータイムあり
    },
    "JPY": {
        "name": "東京",
        "open_jst": (9, 0),
        "close_jst": (15, 0),
        "timezone": "JST",
        "dst": False  # サマータイムなし
    },
    "GBP": {
        "name": "ロンドン",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),  # 標準時は2:00、サマータイムは1:00
        "timezone": "GMT/BST",
        "dst": True
    },
    "AUD": {
        "name": "シドニー",
        "open_jst": (7, 0),  # 標準時は8:00、サマータイムは7:00
        "close_jst": (15, 0),
        "timezone": "AEST/AEDT",
        "dst": True
    },
    "CAD": {
        "name": "トロント",
        "open_jst": (22, 0),  # 標準時は23:00、サマータイムは22:00
        "close_jst": (7, 0),  # 標準時は8:00、サマータイムは7:00
        "timezone": "EST/EDT",
        "dst": True
    },
    "CHF": {
        "name": "チューリッヒ",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),  # 標準時は2:00、サマータイムは1:00
        "timezone": "CET/CEST",
        "dst": True
    },
    "NZD": {
        "name": "ウェリントン",
        "open_jst": (6, 0),  # 標準時は7:00、サマータイムは6:00
        "close_jst": (14, 0),
        "timezone": "NZST/NZDT",
        "dst": True
    },
    "SEK": {
        "name": "ストックホルム",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),
        "timezone": "CET/CEST",
        "dst": True
    },
    "NOK": {
        "name": "オスロ",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),
        "timezone": "CET/CEST",
        "dst": True
    },
    "DKK": {
        "name": "コペンハーゲン",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),
        "timezone": "CET/CEST",
        "dst": True
    },
    "ZAR": {
        "name": "ヨハネスブルグ",
        "open_jst": (15, 0),  # UTC+2固定
        "close_jst": (23, 0),
        "timezone": "SAST",
        "dst": False
    },
    "MXN": {
        "name": "メキシコシティ",
        "open_jst": (23, 0),  # 標準時は24:00、サマータイムは23:00
        "close_jst": (8, 0),
        "timezone": "CST/CDT",
        "dst": True
    },
    "TRY": {
        "name": "イスタンブール",
        "open_jst": (15, 0),  # UTC+3固定
        "close_jst": (23, 0),
        "timezone": "TRT",
        "dst": False
    },
    "BRL": {
        "name": "サンパウロ",
        "open_jst": (22, 0),  # UTC-3固定
        "close_jst": (6, 0),
        "timezone": "BRT",
        "dst": False
    },
    "CNH": {
        "name": "上海/香港",
        "open_jst": (10, 0),  # UTC+8固定
        "close_jst": (17, 0),
        "timezone": "CST/HKT",
        "dst": False
    },
    "HKD": {
        "name": "香港",
        "open_jst": (10, 0),  # UTC+8固定
        "close_jst": (17, 0),
        "timezone": "HKT",
        "dst": False
    },
    "SGD": {
        "name": "シンガポール",
        "open_jst": (10, 0),  # UTC+8固定
        "close_jst": (17, 0),
        "timezone": "SGT",
        "dst": False
    },
    "INR": {
        "name": "ムンバイ",
        "open_jst": (12, 30),  # UTC+5:30固定
        "close_jst": (19, 0),
        "timezone": "IST",
        "dst": False
    },
    "KRW": {
        "name": "ソウル",
        "open_jst": (10, 0),  # UTC+9固定
        "close_jst": (17, 0),
        "timezone": "KST",
        "dst": False
    },
    "TWD": {
        "name": "台北",
        "open_jst": (10, 0),  # UTC+8固定
        "close_jst": (17, 0),
        "timezone": "CST",
        "dst": False
    },
    "THB": {
        "name": "バンコク",
        "open_jst": (11, 0),  # UTC+7固定
        "close_jst": (18, 0),
        "timezone": "ICT",
        "dst": False
    },
    "PLN": {
        "name": "ワルシャワ",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),
        "timezone": "CET/CEST",
        "dst": True
    },
    "HUF": {
        "name": "ブダペスト",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),
        "timezone": "CET/CEST",
        "dst": True
    },
    "CZK": {
        "name": "プラハ",
        "open_jst": (16, 0),  # 標準時は17:00、サマータイムは16:00
        "close_jst": (1, 0),
        "timezone": "CET/CEST",
        "dst": True
    },
}


def is_dst_us() -> bool:
    """アメリカのサマータイム判定"""
    now = datetime.datetime.now()
    year = now.year
    march_1 = datetime.datetime(year, 3, 1)
    march_second_sunday = march_1 + datetime.timedelta(days=(6 - march_1.weekday() + 7))
    november_1 = datetime.datetime(year, 11, 1)
    november_first_sunday = november_1 + datetime.timedelta(days=(6 - november_1.weekday()))
    return march_second_sunday <= now < november_first_sunday


def is_dst_eu() -> bool:
    """ヨーロッパのサマータイム判定"""
    now = datetime.datetime.now()
    year = now.year
    march_31 = datetime.datetime(year, 3, 31)
    march_last_sunday = march_31 - datetime.timedelta(days=(march_31.weekday() + 1) % 7)
    october_31 = datetime.datetime(year, 10, 31)
    october_last_sunday = october_31 - datetime.timedelta(days=(october_31.weekday() + 1) % 7)
    return march_last_sunday <= now < october_last_sunday


def get_market_hours(currency_code: str) -> Optional[Dict]:
    """
    通貨コードから市場時間情報を取得
    
    Args:
        currency_code: 通貨コード（例: "USD", "JPY"）
    
    Returns:
        Dict: 市場時間情報、見つからない場合はNone
    """
    market_info = MARKET_HOURS.get(currency_code.upper())
    if not market_info:
        return None
    
    # サマータイムを考慮して時間を調整
    if market_info["dst"]:
        if currency_code.upper() in ["USD", "CAD"]:
            # アメリカ/カナダのサマータイム
            if is_dst_us():
                open_hour, open_min = market_info["open_jst"]
                close_hour, close_min = market_info["close_jst"]
            else:
                # 標準時は1時間遅い
                open_hour, open_min = market_info["open_jst"]
                open_hour = (open_hour + 1) % 24
                close_hour, close_min = market_info["close_jst"]
                close_hour = (close_hour + 1) % 24
        elif currency_code.upper() in ["EUR", "GBP", "CHF", "SEK", "NOK", "DKK", "PLN", "HUF", "CZK"]:
            # ヨーロッパのサマータイム
            if is_dst_eu():
                open_hour, open_min = market_info["open_jst"]
                close_hour, close_min = market_info["close_jst"]
            else:
                # 標準時は1時間遅い
                open_hour, open_min = market_info["open_jst"]
                open_hour = (open_hour + 1) % 24
                close_hour, close_min = market_info["close_jst"]
                close_hour = (close_hour + 1) % 24
        elif currency_code.upper() in ["AUD"]:
            # オーストラリアのサマータイム（10月第1日曜日から4月第1日曜日まで）
            now = datetime.datetime.now()
            year = now.year
            october_1 = datetime.datetime(year, 10, 1)
            october_first_sunday = october_1 + datetime.timedelta(days=(6 - october_1.weekday()))
            april_1_next = datetime.datetime(year + 1, 4, 1) if now.month < 4 else datetime.datetime(year, 4, 1)
            april_first_sunday = april_1_next + datetime.timedelta(days=(6 - april_1_next.weekday()))
            
            if october_first_sunday <= now < april_first_sunday:
                # サマータイム中
                open_hour, open_min = market_info["open_jst"]
                open_hour = (open_hour - 1) % 24
                close_hour, close_min = market_info["close_jst"]
            else:
                # 標準時
                open_hour, open_min = market_info["open_jst"]
                close_hour, close_min = market_info["close_jst"]
        elif currency_code.upper() in ["NZD"]:
            # ニュージーランドのサマータイム（9月最終日曜日から4月第1日曜日まで）
            now = datetime.datetime.now()
            year = now.year
            september_30 = datetime.datetime(year, 9, 30)
            september_last_sunday = september_30 - datetime.timedelta(days=(september_30.weekday() + 1) % 7)
            april_1_next = datetime.datetime(year + 1, 4, 1) if now.month < 4 else datetime.datetime(year, 4, 1)
            april_first_sunday = april_1_next + datetime.timedelta(days=(6 - april_1_next.weekday()))
            
            if september_last_sunday <= now < april_first_sunday:
                # サマータイム中
                open_hour, open_min = market_info["open_jst"]
                open_hour = (open_hour - 1) % 24
                close_hour, close_min = market_info["close_jst"]
            else:
                # 標準時
                open_hour, open_min = market_info["open_jst"]
                close_hour, close_min = market_info["close_jst"]
        else:
            open_hour, open_min = market_info["open_jst"]
            close_hour, close_min = market_info["close_jst"]
    else:
        open_hour, open_min = market_info["open_jst"]
        close_hour, close_min = market_info["close_jst"]
    
    return {
        "name": market_info["name"],
        "open_jst": (open_hour, open_min),
        "close_jst": (close_hour, close_min),
        "timezone": market_info["timezone"],
        "open_str": f"{open_hour:02d}:{open_min:02d}",
        "close_str": f"{close_hour:02d}:{close_min:02d}"
    }


def get_pair_market_info(pair_name: str) -> Dict:
    """
    通貨ペアの市場情報を取得
    
    Args:
        pair_name: 通貨ペア名（例: "USD/JPY"）
    
    Returns:
        Dict: 市場情報
    """
    if "/" not in pair_name:
        return {}
    
    base, quote = pair_name.split("/")
    base_flag = get_currency_flag(base)
    quote_flag = get_currency_flag(quote)
    
    base_market = get_market_hours(base)
    quote_market = get_market_hours(quote)
    
    return {
        "base": base,
        "quote": quote,
        "base_flag": base_flag,
        "quote_flag": quote_flag,
        "base_market": base_market,
        "quote_market": quote_market
    }
