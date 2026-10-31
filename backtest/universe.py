"""
回測標的清單：自選清單（stock_core.STOCK_NAME_MAP，即 PRODUCT.md 講的「核心
持股」自選股）＋使用者額外指定的 2308/6830＋流動性前100名（對照組）。
三組清單各自保留標籤，輸出CSV時用「清單分類」欄位區分，供報告分開統計
（使用者自選清單 vs 對照組），避免流動性對照組稀釋/混淆自選股的統計。
"""

from typing import List, Tuple

from stock_core import STOCK_NAME_MAP, unique_watchlist

from backtest.data_fetch import build_liquidity_universe

EXTRA_CODES = [("2308", "台達電"), ("6830", "汎銓")]


def build_universe(liquidity_top_n: int = 100) -> List[Tuple[str, str, str, str]]:
    """回傳 [(name, code, market, 清單分類), ...]，清單分類 in
    {"自選清單", "指定對照", "流動性前N對照組"}。代碼重複時只保留第一次
    出現的分類（自選清單優先），避免同一檔在多個分類重複計算。"""
    seen = set()
    rows = []

    for name, code, market in unique_watchlist(STOCK_NAME_MAP):
        if code in seen:
            continue
        seen.add(code)
        rows.append((name, code, market, "自選清單"))

    for code, name in EXTRA_CODES:
        if code in seen:
            continue
        seen.add(code)
        rows.append((name, code, "TW", "指定對照"))

    for code, market, name in build_liquidity_universe(top_n=liquidity_top_n):
        if code in seen:
            continue
        seen.add(code)
        rows.append((name, code, market, "流動性前100對照組"))

    return rows
