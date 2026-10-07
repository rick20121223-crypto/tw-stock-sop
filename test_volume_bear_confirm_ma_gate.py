"""
測試「MACD頂背離+OBV量能背離（volume_bear_confirm）要不要也補上均線同步偏空
的閘門」這個修改規則。

背景：sop_decision.py 的 hard_sell 四個條件裡，dead_cross_confirmed 這條
("MACD死亡交叉") 有 `ma_bias < 0` 的均線同步閘門才算確認；但
volume_bear_confirm 這條 ("MACD頂背離+OBV背離同步") 完全沒有均線閘門，
就算生死線(MA35)還在上揚且站上（結構偏多），只要頂背離+OBV同步出現，一樣
直接判「賣出減碼」——這跟但丁老師「均線斜率具最終裁決權，一代指標未破壞
不能單憑二三代指標背離盲目砍單」的原則不一致，也跟_step3_macd自己的文件
字串「MACD背離僅供警戒，不單獨判賣」矛盾。

2026-10-05 金像電(2368) 案例：10/2收盤觸發這條規則判賣出減碼，隔一個交易日
(10/5) 卻在PCB族群消息面帶動下直接鎖漲停——促成這次測試。

修改規則：volume_bear_confirm 額外要求 ma_bias < 0（跟dead_cross_confirmed
用同一個閘門）才算數，否則降級走 caution_only（觀望+戒備防守的但書提示，
不直接判賣）。

做法：不用文字關鍵字比對的簡化模擬（那只能抓「單獨觸發」，抓不到「均線
到底偏多偏空」這個數值條件），直接重新呼叫 sop_decision.py 裡的私有步驟
函式（_step1~_step6等），自己重組一份「修改版」evaluate_timeframe，只有
hard_sell / caution_only 那兩行判斷邏輯改掉，其餘完全沿用正式函式，確保
跟真正改程式碼的行為一致。

用法（在專案資料夾內執行）：
    python3 test_volume_bear_confirm_ma_gate.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

import sop_decision as sd
from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS, _build_weekly
from sop_decision import Verdict, classify_final, combine_timeframes, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, FAST_MA, KEY_MA, get_stock_data, ma_slope, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24


def evaluate_timeframe_modified(df: pd.DataFrame, timeframe_label: str) -> Verdict:
    """跟 sop_decision.evaluate_timeframe 完全相同，唯一差異：
    volume_bear_confirm 額外要求 ma_bias < 0 才能觸發 hard_sell / 提升confidence，
    否則落到 caution_only（觀望+戒備），不直接判賣出減碼。"""
    RULES = sd.RULES
    if df.empty:
        return Verdict(conclusion="觀望", confidence="低", score=0.0,
                        reasons=["資料為空，無法判讀"], caveats=[])

    scoring_cfg = RULES["indicators"]["scoring"]
    buy_cfg = RULES["buy_rules"]
    add_cfg = RULES["add_rules"]
    watch_cfg = RULES["watch_rules"]
    sell_cfg = RULES["sell_rules"]
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]

    tf = sd.normalize_timeframe(timeframe_label)
    latest = df.iloc[-1]
    reasons = []
    caveats = []

    four_key_bias = sd._step1_four_key_prices(tf, latest, reasons, caveats)
    ma_bias, key_ma_down_veto = sd._step2_ma(df, tf, latest, reasons, caveats)
    ma_bias += sd._step2_weekly_pullback_zone(df, tf, latest, reasons)
    ma_bias += sd._step2_weekly_kd_golden_cross(df, tf, latest, reasons)
    ma_bias += sd._step2_60min_turtle_confirm(df, tf, latest, reasons)
    macd_bias, dead_cross, macd_bearish_divergence = sd._step3_macd(df, latest, reasons, caveats)
    volume_bias, volume_bear_confirm = sd._step4_volume(
        df, tf, latest, macd_bearish_divergence, reasons, caveats)
    mtm_exit_alert = sd._step5_mtm(df, tf, latest, reasons)
    cci_exit_alert = sd._step5b_cci(df, tf, latest, reasons, caveats)

    # ★ 唯一的修改：volume_bear_confirm 補上均線同步閘門（比照dead_cross_confirmed）
    volume_bear_confirm_gated = volume_bear_confirm and ma_bias < 0
    if volume_bear_confirm and not volume_bear_confirm_gated:
        caveats.append("[測試規則] 頂背離+OBV同步走壞，但生死線仍偏多，先降級為戒備觀望、不直接砍單")

    fast_col = FAST_MA.get(tf)
    fast_slope = ma_slope(df, fast_col, lookback=slope_lookback) if fast_col else "資料不足"
    fast_slope_up = fast_slope == "上揚"
    fast_slope_down = fast_slope == "下彎"

    w_bottom_neckline = sd._detect_w_bottom(df, fast_slope_up, macd_bias, reasons)
    m_top_neckline = sd._detect_m_top(df, fast_slope_down, macd_bias, reasons)
    triangle = sd._detect_triangle_consolidation(df, tf, reasons, caveats)
    triangle_force_watch = triangle["active"] and not triangle["volume_confirmed"]

    platform_bias = sd._step6_left_side_platform(
        df, tf, latest, fast_slope_up, reasons, caveats,
        w_bottom_neckline=w_bottom_neckline, m_top_neckline=m_top_neckline)
    ma120_second_leg = sd._detect_ma120_second_leg(df, tf)
    extreme_deviation = sd._check_extreme_deviation(tf, latest, reasons)
    weekly_blowoff = sd._check_weekly_blowoff(tf, df, latest, reasons, caveats)

    score = (four_key_bias * scoring_cfg["four_key_weight"] + ma_bias * scoring_cfg["ma_weight"]
             + macd_bias + volume_bias + platform_bias)

    four_key_broken = tf == "日" and four_key_bias == -1
    dead_cross_confirmed = dead_cross and ma_bias < 0
    hard_sell = (
        (key_ma_down_veto and sell_cfg.get("key_ma_down_veto", True))
        or (four_key_broken and sell_cfg.get("four_key_broken", True))
        or (dead_cross_confirmed and sell_cfg.get("dead_cross_confirmed", True))
        or (volume_bear_confirm_gated and sell_cfg.get("volume_bear_confirm", True))  # ★ 改這行
    )

    caution_only = (
        watch_cfg.get("caution_only_on_macd_divergence", True)
        and macd_bearish_divergence and not volume_bear_confirm_gated and not key_ma_down_veto  # ★ 改這行
    )

    strong_bullish = (
        ma_bias >= add_cfg["ma_bias_min"] and macd_bias >= add_cfg["macd_bias_min"]
        and four_key_bias >= add_cfg["four_key_bias_min"] and volume_bias >= add_cfg["volume_bias_min"]
    )

    mtm_alert_active = tf == "60分" and mtm_exit_alert and watch_cfg.get("mtm_60m_exit_downgrade", True)
    cci_alert_active = cci_exit_alert and watch_cfg.get("cci_exit_alert_downgrade", True)
    leading_indicator_exit_alert = (mtm_alert_active or cci_alert_active) and not key_ma_down_veto

    if hard_sell:
        conclusion = "賣出減碼"
    elif leading_indicator_exit_alert:
        conclusion = "觀望"
        if mtm_alert_active:
            caveats.append("60分 MTM 已示警，建議短線先出場觀察，暫不否定較長天期結構")
        if cci_alert_active:
            caveats.append(f"{tf} CCI 已示警（由+100折返），建議短線先出場觀察，暫不否定較長天期結構")
    elif caution_only:
        conclusion = "觀望"
        caveats.append("MACD出現警戒訊號，但均線與量能尚未同步走壞，先觀望、不追空")
    elif strong_bullish:
        conclusion = "加碼" if (ma_bias >= add_cfg["addon_ma_bias"] and volume_bias > 0) else "買進"
    elif score >= buy_cfg["score_threshold"]:
        conclusion = "買進"
    elif score <= sell_cfg["score_threshold"]:
        conclusion = "賣出減碼"
    else:
        conclusion = "觀望"

    if extreme_deviation and watch_cfg.get("extreme_deviation_downgrade", True) and conclusion in ("買進", "加碼"):
        caveats.append(f"雖符合買進/加碼條件，但乖離過大，先降級為觀望，等乖離收斂再說（原結論：{conclusion}）")
        conclusion = "觀望"

    conclusion = sd._apply_buy_add_sync_gate(
        tf, conclusion, fast_slope_up, macd_bias, volume_bias,
        key_ma_down_veto, four_key_broken, caveats)

    if weekly_blowoff and conclusion in ("買進", "加碼"):
        conclusion = "觀望"

    if triangle_force_watch and watch_cfg.get("triangle_consolidation_downgrade", True) \
            and conclusion in ("買進", "加碼"):
        conclusion = "觀望"

    if ma120_second_leg and RULES["special_notes"].get("ma120_second_leg_hint", True) \
            and conclusion not in ("賣出減碼", "加碼"):
        caveats.append(
            "偵測到大跌測到半年線(MA120)打出第二隻腳未破的特殊情境：可考慮「帶著鋼盔慢買」——"
            "先進 5-10% 基本部位，待5分/60分指標確認低點墊高、均線轉平緩後，每日加碼約5%"
            "（僅供分批佈局參考，非立即買進訊號）"
        )

    abs_score = abs(score)
    if key_ma_down_veto or volume_bear_confirm_gated or (ma_bias >= add_cfg["addon_ma_bias"] and volume_bias > 0):  # ★ 改這行
        confidence = "高"
    elif abs_score >= scoring_cfg["confidence_mid_abs_score"]:
        confidence = "中"
    else:
        confidence = "低" if abs_score < scoring_cfg["confidence_low_abs_score"] else "中"

    if not reasons:
        reasons.append("各項指標訊號不明顯，暫無明確方向")

    return Verdict(conclusion=conclusion, confidence=confidence, score=score,
                    reasons=reasons, caveats=caveats)


def _evaluate_both(df_day_full: pd.DataFrame, as_of_date: str):
    day_slice = df_day_full[df_day_full["date"] <= as_of_date]
    if day_slice.empty:
        return None
    week_slice = _build_weekly(day_slice)
    if week_slice.empty:
        return None
    week_slice = run_all_indicators(week_slice, "週")

    verdict_day_orig = evaluate_timeframe(day_slice, "日")
    verdict_week_orig = evaluate_timeframe(week_slice, "週")
    original_signal = classify_final(
        combine_timeframes({"日": verdict_day_orig, "週": verdict_week_orig})["最終建議"])

    verdict_day_mod = evaluate_timeframe_modified(day_slice, "日")
    verdict_week_mod = evaluate_timeframe_modified(week_slice, "週")
    modified_signal = classify_final(
        combine_timeframes({"日": verdict_day_mod, "週": verdict_week_mod})["最終建議"])

    # 這次事件原本是不是靠 volume_bear_confirm 觸發（用來分桶統計)
    gated_out = (
        original_signal == "賣出減碼" and modified_signal != "賣出減碼"
    )

    return original_signal, modified_signal, gated_out


def analyze_one(name: str, code: str, market: str, api_token: str, backtest_months: int) -> dict:
    today = date.today()
    backtest_start = today - timedelta(days=backtest_months * 30)
    fetch_start = (backtest_start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    try:
        df_day = get_stock_data(code, market, fetch_start, str(today), api_token)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": str(exc)}
    if df_day.empty:
        return {"name": name, "code": code, "error": "日線資料為空"}
    df_day = run_all_indicators(df_day, "日")
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    as_of_dates = [d for d in df_day["date"] if d >= backtest_start_str]

    original_events, modified_events, gated_out_events, newly_added_events = [], [], [], []
    prev_original, prev_modified = None, None
    for d in as_of_dates:
        result = _evaluate_both(df_day, d)
        if result is None:
            continue
        original_signal, modified_signal, gated_out = result
        close = float(df_day.loc[df_day["date"] == d, "close"].iloc[-1])
        fwd = _forward_returns(df_day, d)
        row_fwd = {f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS}

        if original_signal == "賣出減碼" and prev_original not in (None, "賣出減碼"):
            original_events.append({"name": name, "code": code, "date": d, "close": close, **row_fwd})
        if modified_signal == "賣出減碼" and prev_modified not in (None, "賣出減碼"):
            modified_events.append({"name": name, "code": code, "date": d, "close": close, **row_fwd})
        # 第一次轉成賣出減碼，而且是被均線閘門擋下來的那一筆，才算「被拔掉的事件」
        if (original_signal == "賣出減碼" and prev_original not in (None, "賣出減碼") and gated_out):
            gated_out_events.append({"name": name, "code": code, "date": d, "close": close, **row_fwd})
        if (modified_signal == "賣出減碼" and prev_modified not in (None, "賣出減碼")
                and original_signal != "賣出減碼"):
            newly_added_events.append({"name": name, "code": code, "date": d, "close": close, **row_fwd})

        prev_original, prev_modified = original_signal, modified_signal

    return {
        "name": name, "code": code, "error": None,
        "original_events": original_events, "modified_events": modified_events,
        "gated_out_events": gated_out_events, "newly_added_events": newly_added_events,
    }


def run(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(analyze_one, name, code, market, api_token, backtest_months)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _stats_table(events: list) -> dict:
    df = pd.DataFrame(events)
    row = {"事件數": len(df)}
    for n in FORWARD_WINDOWS:
        valid = df[f"fwd_ret_{n}d"].dropna() if not df.empty and f"fwd_ret_{n}d" in df.columns else pd.Series(dtype=float)
        row[f"{n}日_平均報酬%"] = round(valid.mean() * 100, 2) if len(valid) else None
        row[f"{n}日_續跌比例%"] = round((valid < 0).mean() * 100, 1) if len(valid) else None
        row[f"{n}日_上漲比例%"] = round((valid > 0).mean() * 100, 1) if len(valid) else None
    return row


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)

    all_original, all_modified, all_gated_out, all_newly_added = [], [], [], []
    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")
            continue
        all_original.extend(r["original_events"])
        all_modified.extend(r["modified_events"])
        all_gated_out.extend(r["gated_out_events"])
        all_newly_added.extend(r["newly_added_events"])

    print(f"\n=== 回測{months_arg}個月、核心持股{len(unique_watchlist(STOCK_NAME_MAP))}檔：長期(日/週)賣出減碼事件統計 ===\n")

    print("【現行規則】volume_bear_confirm 無均線閘門，單獨可觸發硬賣：")
    print(_stats_table(all_original))
    print()
    print("【修改規則】volume_bear_confirm 要求 ma_bias<0 同步，否則降級戒備觀望：")
    print(_stats_table(all_modified))

    print(f"\n=== 被均線閘門拔掉的事件（原本靠頂背離+OBV單獨觸發賣出，均線其實還偏多）===")
    print(_stats_table(all_gated_out))
    if all_gated_out:
        print("\n明細：")
        for e in sorted(all_gated_out, key=lambda x: x["date"]):
            fwd_str = "  ".join(f"{n}日:{e.get(f'fwd_ret_{n}d')}" for n in FORWARD_WINDOWS
                                 if e.get(f"fwd_ret_{n}d") is not None)
            print(f"  {e['name']}({e['code']}) {e['date']} 收盤{e['close']}  {fwd_str}")

    print(f"\n=== 修改規則新增的事件（理論上應該是空的，若非空代表邏輯有誤）===")
    print(_stats_table(all_newly_added))

    print("\n讀法：")
    print("- 「被拔掉的事件」如果事後平均報酬是正的、上漲比例高 → 代表這些原本是假背離/誘空，")
    print("  均線閘門擋對了，修改規則比較好。")
    print("- 如果「被拔掉的事件」事後平均報酬仍是負的、續跌比例高 → 代表這些其實是真的要賣，")
    print("  均線閘門擋錯了，不該改，應該維持現行規則或另尋更精確的分辨方式。")
    print("- 「修改規則新增的事件」理論上應為0（閘門只會讓賣出變少，不會變多），若不是0代表模擬寫錯。")
