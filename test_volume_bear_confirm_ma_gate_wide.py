"""
test_volume_bear_confirm_ma_gate.py 的擴大樣本版：原本只測13檔核心持股、
24個月，樣本數太小（只有30筆被均線閘門拔掉的事件，其中2筆嚴重反例就佔了
不小比例），不足以下結論。

擴大方式：
1. 股票池從13檔核心持股，擴大加入使用者「產業龍頭股對照表」記憶裡17個
   產業別的龍頭股（矽晶圓/晶圓代工/封測/驅動IC/IP矽智財/手機IC/PCB/
   IC載板/電子書/被動元件/原物料/貨櫃航運/散裝航運/重電/散熱/光通訊/風電），
   去重後約50檔，橫跨多個產業，不再只侷限於記憶體/PCB這種本來就集中在
   核心持股清單裡的族群，測試結果比較不會只反映單一產業的特性。
2. 回測期間從24個月拉長到36個月，涵蓋更多次不同的多空循環。

其餘邏輯（修改規則定義、事件比對、forward return計算）完全沿用
test_volume_bear_confirm_ma_gate.py，直接 import 其函式，不重複寫一份。

用法：
    python3 test_volume_bear_confirm_ma_gate_wide.py [FinMind_Token] [回測月數=36]
"""
import os
import sys

import pandas as pd

from test_volume_bear_confirm_ma_gate import (
    BACKTEST_MONTHS_DEFAULT,
    _stats_table,
    analyze_one,
)
import concurrent.futures

# 核心持股（既有）
from stock_core import STOCK_NAME_MAP

# 產業龍頭股對照表（使用者提供，見 memory/reference_industry_leader_stocks.md）
INDUSTRY_LEADER_MAP = {
    "環球晶": ("6488", "TW"), "中美晶": ("5483", "TW"), "台勝科": ("3532", "TW"),
    "台積電": ("2330", "TW"), "聯電": ("2303", "TW"), "世界": ("5347", "TW"),
    "日月光投控": ("3711", "TW"),
    "聯詠": ("3034", "TW"), "敦泰": ("3545", "TW"), "天鈺": ("4961", "TW"),
    "創意": ("3443", "TW"), "智原": ("3035", "TW"),
    "聯發科": ("2454", "TW"),
    "華通": ("2313", "TW"), "臻鼎-KY": ("4958", "TW"),
    "欣興": ("3037", "TW"), "南電": ("8046", "TW"), "景碩": ("3189", "TW"),
    "元太": ("8069", "TW"), "振曜": ("6143", "TW"),
    "國巨": ("2327", "TW"), "華新科": ("2492", "TW"),
    "中鋼": ("2002", "TW"), "台塑化": ("6505", "TW"), "南亞": ("1303", "TW"),
    "台化": ("1301", "TW"), "台塑": ("1326", "TW"),
    "長榮": ("2603", "TW"), "萬海": ("2615", "TW"), "陽明": ("2609", "TW"),
    "慧洋-KY": ("2637", "TW"), "裕民": ("2606", "TW"), "新興": ("2605", "TW"),
    "華城": ("1519", "TW"), "士電": ("1503", "TW"), "中興電": ("1513", "TW"),
    "亞力": ("1514", "TW"), "台達電": ("2308", "TW"), "大同": ("2371", "TW"),
    "奇鋐": ("3017", "TW"), "雙鴻": ("3324", "TW"), "健策": ("3653", "TW"),
    "高力": ("8996", "TW"),
    "波若威": ("3163", "TW"), "上詩": ("3363", "TW"), "眾達-KY": ("4977", "TW"),
    "世紀鋼": ("9958", "TW"), "森崴能源": ("6806", "TW"), "上緣投控": ("3708", "TW"),
}


def wide_watchlist() -> list:
    combined = dict(STOCK_NAME_MAP)
    combined.update(INDUSTRY_LEADER_MAP)
    seen, result = set(), []
    for name, (code, market) in combined.items():
        key = (code, market)
        if key in seen:
            continue
        seen.add(key)
        result.append((name, code, market))
    return result


def run_wide(api_token: str, backtest_months: int) -> list:
    watchlist = wide_watchlist()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(analyze_one, name, code, market, api_token, backtest_months)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 36

    wl = wide_watchlist()
    print(f"股票池共 {len(wl)} 檔（13核心持股 + {len(INDUSTRY_LEADER_MAP)}檔產業龍頭，去重後）")

    results = run_wide(api_token, months_arg)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)

    all_original, all_modified, all_gated_out, all_newly_added = [], [], [], []
    error_count = 0
    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")
            error_count += 1
            continue
        all_original.extend(r["original_events"])
        all_modified.extend(r["modified_events"])
        all_gated_out.extend(r["gated_out_events"])
        all_newly_added.extend(r["newly_added_events"])

    ok_count = len(results) - error_count
    print(f"\n=== 回測{months_arg}個月、{ok_count}/{len(wl)}檔成功取得資料：長期(日/週)賣出減碼事件統計 ===\n")

    print("【現行規則】volume_bear_confirm 無均線閘門，單獨可觸發硬賣：")
    print(_stats_table(all_original))
    print()
    print("【修改規則】volume_bear_confirm 要求 ma_bias<0 同步，否則降級戒備觀望：")
    print(_stats_table(all_modified))

    print(f"\n=== 被均線閘門拔掉的事件（原本靠頂背離+OBV單獨觸發賣出，均線其實還偏多）===")
    print(_stats_table(all_gated_out))

    from backtest_signal_log import FORWARD_WINDOWS
    if all_gated_out:
        print("\n明細（按最差的20日報酬排序，優先看尾部風險）：")
        sortable = [e for e in all_gated_out if e.get("fwd_ret_20d") is not None]
        sortable.sort(key=lambda x: x["fwd_ret_20d"])
        for e in sortable:
            fwd_str = "  ".join(f"{n}日:{round(e.get(f'fwd_ret_{n}d')*100,1)}%" for n in FORWARD_WINDOWS
                                 if e.get(f"fwd_ret_{n}d") is not None)
            print(f"  {e['name']}({e['code']}) {e['date']} 收盤{e['close']}  {fwd_str}")
        no_data = [e for e in all_gated_out if e.get("fwd_ret_20d") is None]
        if no_data:
            print(f"  （另有{len(no_data)}筆太接近現在，20日報酬尚未走完，未列入排序）")

    print(f"\n=== 修改規則新增的事件（理論上應該是空的）===")
    print(_stats_table(all_newly_added))
