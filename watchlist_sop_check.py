"""
把「法人買賣超排行」＋「元大投顧（等）重點股」這兩份候選觀察名單，自動跑一輪
multi_timeframe_check的SOP判讀，依「買進/加碼優先」排序輸出，省去一檔一檔
手動查的功夫。

這份名單刻意只收「新面孔」候選（institutional_ranking.py／
yuanta_stock_mentions.py 的輪替名單），不含 STOCK_NAME_MAP 固定清單——那些
核心持股已經每天由 notify_email.py 判讀、有獨立的買賣訊號通知，不需要在這裡
重複判讀。只是「篩選觀察名單」的輔助工具，不是SOP本身：結論仍是
multi_timeframe_check/sop_decision算出來的結構判讀，不會因為某檔股票同時被
法人跟投顧看好就放寬買進門檻。

用法：
    python3 watchlist_sop_check.py [FinMind Token] [Fugle Key]
    # 沒帶參數時，改讀環境變數 FINMIND_TOKEN / FUGLE_API_KEY（跟
    # notify_email.py 同一套慣例）
"""

import concurrent.futures
import os
import sys
from typing import Dict, List, Tuple

import institutional_ranking as ir
import yuanta_stock_mentions as ysm
from multi_timeframe_check import full_check
from sop_decision import classify_final


def candidates() -> List[Tuple[str, str, str, str]]:
    """
    合併法人買賣超排行＋元大投顧重點股兩份輪替名單，回傳
    [(name, code, market, source), ...]。同一檔被兩個來源同時選到時，來源
    標籤合併並加🔥前綴（跟 notify_email.full_watchlist() 同一套邏輯——多個
    獨立來源同時關注，本身就比單一來源更值得注意）。
    """
    merged: Dict[str, dict] = {}

    def _merge(name, code, market, label):
        entry = merged.setdefault(code, {"name": name, "market": market, "sources": []})
        entry["sources"].append(label)

    for name, code, market, side in ir.rotating_watchlist():
        _merge(name, code, market, f"法人排行({side})")
    for name, code, market, source in ysm.rotating_watchlist():
        _merge(name, code, market, source)

    result = []
    for code, info in merged.items():
        label = "＋".join(info["sources"])
        if len(info["sources"]) >= 2:
            label = f"🔥{label}"
        result.append((info["name"], code, info["market"], label))
    return result


def run(api_token: str, fugle_api_key: str) -> List[dict]:
    """
    對 candidates() 逐檔跑 multi_timeframe_check.full_check，個別失敗
    （例如FinMind查不到該代碼、該股剛上市資料不足）不影響其他檔，回傳依
    「買進/加碼優先」排序的結果列表：
        [{name, code, source, 結論, error}, ...]
    error 非None代表該檔判讀失敗，結論欄位會是None。
    """
    rows = []

    def _check_one(name, code, market, source):
        try:
            result = full_check(code, market, api_token, fugle_api_key)
            return {"name": name, "code": code, "source": source,
                    "結論": result["最終建議"], "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "code": code, "source": source,
                    "結論": None, "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_check_one, *c) for c in candidates()]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())

    priority = {"買進": 0, "加碼": 0, "觀望": 1, "賣出減碼": 2}

    def sort_key(row):
        if row["error"] is not None:
            return (3, row["code"])
        return (priority.get(classify_final(row["結論"]), 1), row["code"])

    rows.sort(key=sort_key)
    return rows


def print_rows(rows: List[dict]) -> None:
    if not rows:
        print("目前沒有候選名單（法人排行／元大投顧重點股皆是空的，可能還沒跑過對應的 rotate/add 指令）。")
        return
    for row in rows:
        if row["error"] is not None:
            print(f"⚠️ {row['name']}（{row['code']}）[{row['source']}]：查詢失敗 - {row['error']}")
            continue
        print(f"{row['name']}（{row['code']}）[{row['source']}] → {row['結論']}")


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINMIND_TOKEN", "")
    fugle_key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FUGLE_API_KEY", "")
    print_rows(run(token, fugle_key))
