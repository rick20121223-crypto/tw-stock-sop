"""
驗證「MACD零軸下死叉 + KD365死叉 + CCI示警（容許3個交易日內）三者同步」
是不是比現有已採用的「MACD死叉+KD365死叉」(Double Check，見
test_kd365_double_check.py)更有保護力——這是sop_rules.yaml 2026-10-30
changelog提到、還沒backtest驗證的新想法（見該檔開頭「CCI最快示警→MTM
同步確認→KD365雙重確認→均線裁決」那段註解）。

使用者確認過兩個定義問題：
    A. 訊號時序只是排序說明，不影響判讀（已經做成signal_priority_note，
       見test_signal_priority_note.py，跟這支腳本無關）。
    B. 「三者同步死叉」的同步容許3個交易日內的誤差（不要求同一根K棒）。

這支腳本只是一次性驗證用，不是正式功能，也不會改動sop_decision.py的
判讀邏輯或sop_rules.yaml的預設行為——驗證有效才考慮比照kd365_veto_enabled
的模式（新增設定開關+對應test_*.py）正式納入。

比較三組「訊號後N日報酬／下跌比例」，B2/C都是B的子集合（遞進篩選，
不是互斥三分——要回答的是「在B的基礎上再疊加條件會不會更準」）：
    B)  MACD死叉 + KD365死叉同時成立（現有已採用的Double Check，直接
        沿用test_kd365_double_check.py的_collect_events()邏輯）
    B2) B再加上「MACD死叉當下DIF在零軸之下」（排除零軸上死叉，零軸上
        死叉通常只是多頭格局裡的正常拉回，但丁老師SOP裡零軸下/上的死叉
        意涵本來就不同）
    C)  B2再加上「CCI exit_alert在事件日前後3個交易日內（含事件當天）
        也曾經出現」（三者同步死叉，CCI是比照MACD/KD365同一套expanding
        window、逐日重算_step5b_cci()，不會用到未來資料往前預知事件本身
        要不要觸發，但C組的「事件後幾天CCI才示警」這個檢查方式，本質上
        跟B2判定事件成立的時間點不同——這是為了回答「三者同步」這個假設
        本身，不是要產生一個「當下就能即時判斷」的即時訊號，先說明清楚）

CCI應用在這裡刻意不受cci.applicable_timeframes限制（正式規則目前只套
60分/5分），用RULES暫時加入"日"的方式沿用_step5b_cci()既有邏輯本身
（跟test_mtm_consolidation_suppress.py暫改RULES旗標做法一致），跑完
記得還原，避免影響其他測試。

沿用test_kd365_double_check.py的「訊號已反轉就排除對應天期統計」修正。

用法（在專案資料夾內執行，需要FinMind token）：
    python3 test_triple_dead_cross_veto_compare.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import RULES, _step3_macd, _step5b_cci, _step_kd365
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24
MAX_REVERSAL_LOOKAHEAD = max(FORWARD_WINDOWS)
CCI_SYNC_TOLERANCE_DAYS = 3  # 使用者確認：三者同步容許3個交易日內的誤差


def _find_reversal_day(df: pd.DataFrame, event_idx: int, fast_col: str, slow_col: str,
                        max_days: int = MAX_REVERSAL_LOOKAHEAD):
    """同test_kd365_double_check.py：從event_idx後一天開始找fast_col重新
    golden cross穿越slow_col之上的第一個交易日偏移量，找不到回傳None。"""
    for offset in range(1, max_days + 1):
        idx = event_idx + offset
        if idx >= len(df):
            return None
        fast, slow = df.iloc[idx].get(fast_col), df.iloc[idx].get(slow_col)
        if pd.isna(fast) or pd.isna(slow):
            continue
        if fast > slow:
            return offset
    return None


def _cci_alert_at(df: pd.DataFrame, idx: int) -> bool:
    """用到idx這天為止的window（不看idx之後的資料），重算_step5b_cci()，
    回傳當天是否有cci_exit_alert。呼叫端負責在RULES裡暫時把"日"加進
    cci.applicable_timeframes，這裡不處理那個開關。"""
    if idx < 0 or idx >= len(df):
        return False
    window = df.iloc[:idx + 1]
    latest = window.iloc[-1]
    if pd.isna(latest.get("CCI")):
        return False
    cci_exit_alert, _ = _step5b_cci(window, "日", latest, [], [])
    return cci_exit_alert


def _cci_synced_within_tolerance(df: pd.DataFrame, event_idx: int,
                                  tolerance_days: int = CCI_SYNC_TOLERANCE_DAYS) -> bool:
    """事件日前後tolerance_days個交易日內（含事件當天），CCI是否曾經
    exit_alert過一次即可算「同步」（不要求同一天，見本檔docstring）。"""
    lo = max(0, event_idx - tolerance_days)
    hi = min(len(df) - 1, event_idx + tolerance_days)
    return any(_cci_alert_at(df, j) for j in range(lo, hi + 1))


def _collect_events(df: pd.DataFrame) -> dict:
    """逐日重建MACD死叉(_step3_macd)、KD365死叉(_step_kd365)，標出
    B/B2/C三組事件（B2⊆B、C⊆B2，遞進篩選，見本檔docstring）。

    迴圈起點跳過KD365暖身期（n+m1+m2根K棒）。
    """
    kd365_cfg = RULES["indicators"]["kd365"]
    warmup_bars = kd365_cfg["n"] + kd365_cfg["m1"] + kd365_cfg["m2"]

    double_check, below_zero, triple_synced = [], [], []

    for i in range(warmup_bars, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]

        reasons, caveats = [], []
        _, macd_dead_cross, _ = _step3_macd(window, latest, reasons, caveats)
        if not macd_dead_cross:
            continue

        reasons2, caveats2 = [], []
        _, kd365_cross_down, _ = _step_kd365(window, "日", latest, reasons2, caveats2)
        if not kd365_cross_down:
            continue

        signal_date = df.iloc[i]["date"]
        reversal = _find_reversal_day(df, i, "K365", "D365")
        double_check.append({"date": signal_date, "reversal_day": reversal})

        if latest["DIF"] >= 0:
            continue  # B2要求零軸下死叉，零軸上的死叉不進B2（也不會進C）
        below_zero.append({"date": signal_date, "reversal_day": reversal})

        if _cci_synced_within_tolerance(df, i):
            triple_synced.append({"date": signal_date, "reversal_day": reversal})

    return {"double_check": double_check, "below_zero": below_zero, "triple_synced": triple_synced}


def _events_to_forward_returns(df: pd.DataFrame, events: list) -> list:
    results = []
    for e in events:
        raw = _forward_returns(df, e["date"])
        reversal = e["reversal_day"]
        if reversal is None:
            results.append(raw)
        else:
            results.append({w: (v if reversal > w else None) for w, v in raw.items()})
    return results


def run(api_token: str, months: int) -> list:
    start = (date.today() - timedelta(days=months * 31 + WARMUP_CALENDAR_DAYS)).isoformat()
    results = []

    def _one(name, code, market):
        try:
            df = get_stock_data(code, market, start, str(date.today()), api_token)
            if df.empty:
                return {"name": name, "code": code, "error": "無資料"}
            df = run_all_indicators(df, "日")
            events = _collect_events(df)
            return {
                "name": name, "code": code,
                "double_check_events": _events_to_forward_returns(df, events["double_check"]),
                "below_zero_events": _events_to_forward_returns(df, events["below_zero"]),
                "triple_synced_events": _events_to_forward_returns(df, events["triple_synced"]),
            }
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "code": code, "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_one, name, code, market)
                   for name, code, market in unique_watchlist(STOCK_NAME_MAP)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, events: list) -> None:
    if not events:
        print(f"{label}：無事件")
        return
    n = len(events)
    print(f"{label}（{n}筆事件）：")
    for window in FORWARD_WINDOWS:
        rets = [e[window] for e in events if e.get(window) is not None]
        excluded = n - len(rets)
        if not rets:
            continue
        avg = sum(rets) / len(rets)
        down_ratio = sum(1 for r in rets if r < 0) / len(rets)
        excl_note = f"，排除{excluded}筆訊號已反轉" if excluded else ""
        print(f"  {window}日後：平均報酬{avg:+.2%}，下跌比例{down_ratio:.1%}"
              f"（{len(rets)}筆{excl_note}）")


def main() -> None:
    api_token = sys.argv[1] if len(sys.argv) > 1 else ""
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    cci_cfg = RULES["indicators"]["cci"]
    original_applicable = cci_cfg.get("applicable_timeframes", [])
    cci_cfg["applicable_timeframes"] = list(set(original_applicable) | {"日"})
    try:
        results = run(api_token, months_arg)
    finally:
        cci_cfg["applicable_timeframes"] = original_applicable  # 還原，避免影響其他測試

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    all_double_check, all_below_zero, all_triple_synced = [], [], []
    for r in results:
        if r.get("error"):
            continue
        all_double_check.extend(r["double_check_events"])
        all_below_zero.extend(r["below_zero_events"])
        all_triple_synced.extend(r["triple_synced_events"])

    print(f"\n=== 回測{months_arg}個月，{len(unique_watchlist(STOCK_NAME_MAP))}檔加總 ===")
    print("（B2、C都是B的子集合，不是三分互斥——要回答的是「在B的基礎上\n"
          "疊加條件會不會更準」，不是三組各自獨立的勝率比較）\n")
    _print_stats("B) MACD死叉+KD365死叉（現有已採用的Double Check）", all_double_check)
    print()
    _print_stats("B2) B再加上MACD死叉當下DIF在零軸之下", all_below_zero)
    print()
    _print_stats(f"C) B2再加上CCI在±{CCI_SYNC_TOLERANCE_DAYS}個交易日內也曾示警（三者同步死叉）",
                 all_triple_synced)


if __name__ == "__main__":
    main()
