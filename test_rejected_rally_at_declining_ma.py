"""
測試2026-10-31新增的「反彈打到下彎均線測不過」扣分項
（indicators.rejected_rally_at_declining_ma，見sop_decision.py
_detect_rejected_rally_at_declining_ma()/_step2_rejected_rally_at_
declining_ma()），以及bear_regime_fake_rally_guard空單提示文字在這個
型態確認時改用更明確的措辭。

背景：用test_rejected_rally_at_declining_ma_backtest.py對24個月/13檔
回測驗證過：均線本身仍下彎時，若反彈貼近均線卻測不過（收盤從未真正
站上），20日後平均報酬3.64%/負報酬比例47.4%，明顯比無條件基準
(7.68%/38.2%)更弱更負——單獨使用就有效，加上KD365死叉同步確認後樣本
大幅縮小、優勢沒有明顯更好，所以不要求KD365同步確認。只適用日線
（生死線MA35）。

用合成資料，不需要API，執行：
    python3 test_rejected_rally_at_declining_ma.py
"""
import pandas as pd

import sop_decision as sd
from sop_decision import RULES
from stock_core import run_all_indicators


def _build_df(n: int = 300) -> pd.DataFrame:
    """建立一個「背景」全部中性的日線資料：close=95全程持平、MA35=100
    全程持平、high剛好低於MA35門檻（95.3，遠低於100*0.98=98），確保
    除非測試明確覆寫某個offset，背景資料本身不會意外觸發任何條件
    （declining/approached/never_closed_above都預設為否）。"""
    closes = [95.0] * n
    highs = [c + 0.3 for c in closes]
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": closes, "max": highs, "min": [c - 0.3 for c in closes],
        "close": closes, "volume": [1000] * n,
    })
    df = run_all_indicators(df, "日")
    df["MA35"] = 100.0
    return df


def _set_offset(df: pd.DataFrame, offset: int, ma35=None, close=None, high=None) -> None:
    """offset=0是今天(最後一筆)，offset=N是N天前。只覆寫傳入的欄位，
    其他欄位維持_build_df()的中性背景值。"""
    idx = df.index[-1 - offset]
    if ma35 is not None:
        df.loc[idx, "MA35"] = ma35
    if close is not None:
        df.loc[idx, "close"] = close
    if high is not None:
        df.loc[idx, "max"] = high


SLOPE_ANCHOR_OFFSET = 3   # 比照cfg.slope_lookback預設值
APPROACH_OFFSET = 1       # 隨便選reject_lookback視窗(offset 1~5)裡的一天來測試貼近均線


def test_no_penalty_when_ma_not_declining():
    df = _build_df()
    # 不覆寫任何東西：MA35全程持平100，declining條件本身就不成立
    reasons = []
    penalty = sd._step2_rejected_rally_at_declining_ma(df, "日", df.iloc[-1], reasons)
    assert penalty == 0.0, f"均線走平(不是下彎)時不該扣分，實際={penalty}"
    print("✅ 均線走平（不是下彎）時不扣分")


def test_no_penalty_when_no_rejection_attempt():
    df = _build_df()
    _set_offset(df, 0, ma35=100.0, close=96.0)           # 今天：均線100，收盤96(仍在下方)
    _set_offset(df, SLOPE_ANCHOR_OFFSET, ma35=110.0)      # 均線下彎(100<110)
    # 其餘offset維持背景值(MA35=100, high=95.3)，high從未貼近過100*0.98=98
    reasons = []
    penalty = sd._step2_rejected_rally_at_declining_ma(df, "日", df.iloc[-1], reasons)
    assert penalty == 0.0, f"沒有反彈貼近均線的動作時不該扣分，實際={penalty}"
    print("✅ 沒有反彈貼近均線的動作時不扣分")


def test_no_penalty_when_already_closed_above():
    """今天收盤已經站上均線了，不算「測不過」，是真的站上去了。"""
    df = _build_df()
    _set_offset(df, 0, ma35=100.0, close=101.0)           # 今天收盤101，站上均線100
    _set_offset(df, SLOPE_ANCHOR_OFFSET, ma35=110.0)      # 均線下彎
    _set_offset(df, APPROACH_OFFSET, high=99.5)           # 貼近均線(100)但沒超過，且never_closed_above仍成立
    reasons = []
    penalty = sd._step2_rejected_rally_at_declining_ma(df, "日", df.iloc[-1], reasons)
    assert penalty == 0.0, f"今天收盤已經站上均線，不算測不過，不該扣分，實際={penalty}"
    print("✅ 今天收盤已經站上均線時不扣分（真的站上去了，不是測不過）")


def test_penalty_when_rejected_rally_confirmed():
    df = _build_df()
    _set_offset(df, 0, ma35=100.0, close=96.0)            # 今天收盤96，仍在均線之下
    _set_offset(df, SLOPE_ANCHOR_OFFSET, ma35=110.0)      # 均線下彎
    _set_offset(df, APPROACH_OFFSET, high=99.5)           # 反彈貼近均線(100)但沒超過
    reasons = []
    penalty = sd._step2_rejected_rally_at_declining_ma(df, "日", df.iloc[-1], reasons)
    cfg = RULES["indicators"]["rejected_rally_at_declining_ma"]
    assert penalty == cfg["penalty"], f"反彈測不過均線應該扣{cfg['penalty']}分，實際={penalty}"
    assert any("測不過" in r or "反彈" in r for r in reasons)
    print(f"✅ 反彈測不過均線時扣分 {penalty}，reasons有說明")


def test_not_applicable_outside_daily_timeframe():
    df = _build_df()
    _set_offset(df, 0, ma35=100.0, close=96.0)
    _set_offset(df, SLOPE_ANCHOR_OFFSET, ma35=110.0)
    _set_offset(df, APPROACH_OFFSET, high=99.5)
    reasons = []
    penalty = sd._step2_rejected_rally_at_declining_ma(df, "60分", df.iloc[-1], reasons)
    assert penalty == 0.0, f"非日線不該扣分，實際={penalty}"
    print("✅ 非日線（60分）不套用這條扣分")


def test_disabled_flag_suppresses_penalty():
    df = _build_df()
    _set_offset(df, 0, ma35=100.0, close=96.0)
    _set_offset(df, SLOPE_ANCHOR_OFFSET, ma35=110.0)
    _set_offset(df, APPROACH_OFFSET, high=99.5)
    RULES["indicators"]["rejected_rally_at_declining_ma"]["enabled"] = False
    try:
        reasons = []
        penalty = sd._step2_rejected_rally_at_declining_ma(df, "日", df.iloc[-1], reasons)
    finally:
        RULES["indicators"]["rejected_rally_at_declining_ma"]["enabled"] = True
    assert penalty == 0.0, f"關閉開關後不該扣分，實際={penalty}"
    print("✅ 開關關閉時不扣分")


if __name__ == "__main__":
    test_no_penalty_when_ma_not_declining()
    test_no_penalty_when_no_rejection_attempt()
    test_no_penalty_when_already_closed_above()
    test_penalty_when_rejected_rally_confirmed()
    test_not_applicable_outside_daily_timeframe()
    test_disabled_flag_suppresses_penalty()
    print()
    print("全部通過。")
