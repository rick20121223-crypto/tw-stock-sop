"""
成交規則：決定一張委託(order)實際在哪一天、用什麼價格成交，以及遇到
漲跌停鎖住時要怎麼處理。這是刻意設計成可替換的策略——`resolve_fill()`
只是目前使用中的策略入口，之後要換別的規則只要寫一個同樣簽名的函式替換
掉呼叫端引用的名字即可，不用改呼叫端程式碼。

買賣不對稱：買單遇漲停鎖住，最多順延`max_buy_defer_days`個交易日（不含
訊號當天），超過就直接取消（status="cancelled"）——避免追高鎖死股無限
期占用資金額度。賣單遇跌停鎖住則不設上限，順延到真的賣出為止
（status="deferred"）——因為部位已經持有，沒有「取消賣出」這種選項，
只能等解鎖。

限制：只用每日收盤價（EOD OHLC）判斷是否「鎖住」，沒有逐筆/五檔資料，
無法判斷盤中實際有沒有機會成交——只要收盤價剛好等於前一天收盤價±10%
（四捨五入到小數點後2位）的價位，就假設當天完全沒有成交機會（買單遇
漲停鎖住／賣單遇跌停鎖住），這是刻意的簡化，也沒有模擬台股實際依價格
區間變動的最小跳動單位（tick size）表。
"""

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional

# price_lookup(code, date) -> {"close": float, "prev_close": float} 或 None（查無資料）
PriceLookup = Callable[[str, date], Optional[Dict[str, float]]]


@dataclass
class FillOutcome:
    filled: bool
    fill_date: Optional[date] = None
    fill_price: Optional[float] = None
    status: str = "pending"  # "filled" / "deferred" / "cancelled"


def _is_locked(side: str, close: float, prev_close: float, limit_pct: float) -> bool:
    if prev_close <= 0:
        return False
    limit_up = round(prev_close * (1 + limit_pct), 2)
    limit_down = round(prev_close * (1 - limit_pct), 2)
    if side == "買":
        return round(close, 2) == limit_up
    elif side == "賣":
        return round(close, 2) == limit_down
    raise ValueError(f"未知的買賣方向: {side!r}")


def default_defer_to_next_day(code: str, side: str, signal_date: date,
                               trading_dates: List[date], price_lookup: PriceLookup,
                               limit_pct: float = 0.10,
                               max_buy_defer_days: int = 3,
                               max_sell_defer_days: Optional[int] = None) -> FillOutcome:
    """預設策略：以signal_date當天收盤價成交；若當天「鎖住」（買單遇漲停／
    賣單遇跌停），順延到下一個可成交日，用那天的收盤價成交，依此類推。

    買單最多嘗試`max_buy_defer_days + 1`次（訊號當天 + 之後最多
    `max_buy_defer_days`個交易日），超過還是鎖住就回傳status="cancelled"。
    賣單在`max_sell_defer_days`為None時不設嘗試次數上限，把`trading_dates`
    用完還賣不掉才回傳status="deferred"（不是"cancelled"——部位還在，
    之後有新的交易日清單進來可以再試）；若指定了`max_sell_defer_days`，
    行為跟買單一樣用「嘗試次數上限」處理，但賣單用完次數後回傳的是
    "deferred"不是"cancelled"（沒有「取消賣出」這個概念）。

    trading_dates：候選交易日清單，必須依時間升冪排列，且包含signal_date
    當天或之後的交易日（由呼叫端提供，這個函式不自己算交易日曆）。

    查無價格資料（price_lookup回傳None）時視為「還不知道能不能成交」，
    回傳status="deferred"但fill_date/fill_price都是None，不會往後硬跳過
    去猜——之後有資料了再重新呼叫一次即可。"""
    if side == "買":
        max_attempts = max_buy_defer_days + 1
    else:
        max_attempts = (max_sell_defer_days + 1) if max_sell_defer_days is not None else None

    candidates = [d for d in trading_dates if d >= signal_date]
    for attempt, d in enumerate(candidates):
        if max_attempts is not None and attempt >= max_attempts:
            break
        price = price_lookup(code, d)
        if price is None:
            return FillOutcome(filled=False, status="deferred")
        if _is_locked(side, price["close"], price["prev_close"], limit_pct):
            continue
        return FillOutcome(filled=True, fill_date=d, fill_price=price["close"], status="filled")

    if side == "買" and max_attempts is not None and len(candidates) >= max_attempts:
        return FillOutcome(filled=False, status="cancelled")
    return FillOutcome(filled=False, status="deferred")


def resolve_fill(code: str, side: str, signal_date: date, trading_dates: List[date],
                  price_lookup: PriceLookup, limit_pct: float = 0.10,
                  max_buy_defer_days: int = 3,
                  max_sell_defer_days: Optional[int] = None) -> FillOutcome:
    """目前使用中的成交規則入口，預設行為=default_defer_to_next_day。要換
    別的策略時，呼叫端改引用別的函式即可，不用改這個函式的呼叫方式。"""
    return default_defer_to_next_day(code, side, signal_date, trading_dates, price_lookup,
                                      limit_pct, max_buy_defer_days, max_sell_defer_days)
