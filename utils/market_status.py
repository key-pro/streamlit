"""
市場のオープン/クローズ状態を判定するモジュール
サマータイム（夏時間）も考慮する
"""
import datetime
from typing import Tuple


def is_dst_us() -> bool:
    """
    アメリカのサマータイム（DST）が適用されているか判定
    3月第2日曜日から11月第1日曜日まで
    """
    now = datetime.datetime.now()
    year = now.year
    
    # 3月第2日曜日を計算
    march_1 = datetime.datetime(year, 3, 1)
    march_second_sunday = march_1 + datetime.timedelta(days=(6 - march_1.weekday() + 7))
    
    # 11月第1日曜日を計算
    november_1 = datetime.datetime(year, 11, 1)
    november_first_sunday = november_1 + datetime.timedelta(days=(6 - november_1.weekday()))
    
    return march_second_sunday <= now < november_first_sunday


def is_dst_eu() -> bool:
    """
    ヨーロッパのサマータイム（DST）が適用されているか判定
    3月最終日曜日から10月最終日曜日まで
    """
    now = datetime.datetime.now()
    year = now.year
    
    # 3月最終日曜日を計算
    march_31 = datetime.datetime(year, 3, 31)
    march_last_sunday = march_31 - datetime.timedelta(days=(march_31.weekday() + 1) % 7)
    
    # 10月最終日曜日を計算
    october_31 = datetime.datetime(year, 10, 31)
    october_last_sunday = october_31 - datetime.timedelta(days=(october_31.weekday() + 1) % 7)
    
    return march_last_sunday <= now < october_last_sunday


def get_market_status() -> Tuple[str, str, bool]:
    """
    現在の市場の状態を判定する
    
    Returns:
        Tuple[str, str, bool]: (状態テキスト, カラー, オープン中かどうか)
        状態テキスト: "MARKET OPEN", "MARKET CLOSED", "休場日"
        カラー: "green", "red", "orange"
        オープン中: True/False
    """
    # 日本時間で現在時刻を取得
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now_jst = datetime.datetime.now(jst)
    
    # 曜日を取得（0=月曜日, 6=日曜日）
    weekday = now_jst.weekday()
    hour = now_jst.hour
    minute = now_jst.minute
    current_time = hour + minute / 60.0
    
    # 週末（土曜日0時〜月曜日9時）は休場
    if weekday == 6:  # 日曜日
        return "休場日", "orange", False
    if weekday == 5 and current_time >= 0:  # 土曜日0時以降
        return "休場日", "orange", False
    if weekday == 0 and current_time < 9:  # 月曜日9時前
        return "休場日", "orange", False
    
    # サマータイムのオフセットを計算
    # 日本はサマータイムなし（UTC+9固定）
    # アメリカ: UTC-5（標準時）または UTC-4（サマータイム）
    # ヨーロッパ: UTC+1（標準時）または UTC+2（サマータイム）
    
    us_offset = -4 if is_dst_us() else -5
    eu_offset = 2 if is_dst_eu() else 1
    
    # 各セッションの時間帯（日本時間）
    # 東京セッション: 9:00-15:00 JST
    # ロンドンセッション: 17:00-1:00 JST（サマータイム考慮）
    # ニューヨークセッション: 22:00-6:00 JST（サマータイム考慮）
    
    # ロンドンセッションの開始時間（サマータイム考慮）
    london_start = 16 if is_dst_eu() else 17  # サマータイム時は1時間早い
    london_end = 1 if is_dst_eu() else 2
    
    # ニューヨークセッションの開始時間（サマータイム考慮）
    ny_start = 21 if is_dst_us() else 22  # サマータイム時は1時間早い
    ny_end = 5 if is_dst_us() else 6
    
    # 市場がオープンか判定
    is_open = False
    
    # 東京セッション（9:00-15:00）
    if 9 <= current_time < 15:
        is_open = True
    
    # ロンドンセッション（17:00-1:00 または 16:00-1:00）
    if london_start <= current_time < 24 or 0 <= current_time < london_end:
        is_open = True
    
    # ニューヨークセッション（22:00-6:00 または 21:00-6:00）
    if ny_start <= current_time < 24 or 0 <= current_time < ny_end:
        is_open = True
    
    if is_open:
        return "MARKET OPEN", "green", True
    else:
        return "MARKET CLOSED", "red", False


def get_market_status_detailed() -> dict:
    """
    詳細な市場状態情報を返す
    
    Returns:
        dict: 市場状態の詳細情報
    """
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now_jst = datetime.datetime.now(jst)
    
    status, color, is_open = get_market_status()
    
    # 次のオープン/クローズ時間を計算
    weekday = now_jst.weekday()
    hour = now_jst.hour
    minute = now_jst.minute
    current_time = hour + minute / 60.0
    
    next_event = None
    next_event_time = None
    
    if status == "休場日":
        # 次のオープンは月曜日9時
        days_until_monday = (7 - weekday) % 7
        if days_until_monday == 0 and current_time < 9:
            days_until_monday = 0
        elif days_until_monday == 0:
            days_until_monday = 7
        
        next_open = now_jst.replace(hour=9, minute=0, second=0, microsecond=0)
        next_open += datetime.timedelta(days=days_until_monday)
        next_event = "次回オープン"
        next_event_time = next_open.strftime("%m/%d %H:%M JST")
    elif is_open:
        # 次のクローズ時間を計算
        if 9 <= current_time < 15:
            # 東京セッション中 → 15時にクローズ
            next_close = now_jst.replace(hour=15, minute=0, second=0, microsecond=0)
            if next_close <= now_jst:
                next_close += datetime.timedelta(days=1)
            next_event = "次回クローズ"
            next_event_time = next_close.strftime("%m/%d %H:%M JST")
        elif current_time >= 22 or current_time < 6:
            # ニューヨークセッション中 → 6時にクローズ
            next_close = now_jst.replace(hour=6, minute=0, second=0, microsecond=0)
            if next_close <= now_jst:
                next_close += datetime.timedelta(days=1)
            next_event = "次回クローズ"
            next_event_time = next_close.strftime("%m/%d %H:%M JST")
        else:
            # ロンドンセッション中 → 1時にクローズ
            next_close = now_jst.replace(hour=1, minute=0, second=0, microsecond=0)
            if next_close <= now_jst:
                next_close += datetime.timedelta(days=1)
            next_event = "次回クローズ"
            next_event_time = next_close.strftime("%m/%d %H:%M JST")
    else:
        # クローズ中 → 次のオープン時間を計算
        if current_time < 9:
            # 東京セッション前
            next_open = now_jst.replace(hour=9, minute=0, second=0, microsecond=0)
            next_event = "次回オープン"
            next_event_time = next_open.strftime("%m/%d %H:%M JST")
        elif current_time < 17:
            # ロンドンセッション前
            london_start = 16 if is_dst_eu() else 17
            next_open = now_jst.replace(hour=london_start, minute=0, second=0, microsecond=0)
            next_event = "次回オープン"
            next_event_time = next_open.strftime("%m/%d %H:%M JST")
        else:
            # ニューヨークセッション前
            ny_start = 21 if is_dst_us() else 22
            next_open = now_jst.replace(hour=ny_start, minute=0, second=0, microsecond=0)
            if next_open <= now_jst:
                next_open += datetime.timedelta(days=1)
            next_event = "次回オープン"
            next_event_time = next_open.strftime("%m/%d %H:%M JST")
    
    return {
        "status": status,
        "color": color,
        "is_open": is_open,
        "next_event": next_event,
        "next_event_time": next_event_time,
        "is_dst_us": is_dst_us(),
        "is_dst_eu": is_dst_eu()
    }
