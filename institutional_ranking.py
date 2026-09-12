"""
三大法人「本週買賣超前十大」自動輪替名單。

每週一把「上週（週一~週五）三大法人買賣超淨額」加總後排出：
    買超前10 + 賣超前10 = 20檔（跳過已經在固定清單 STOCK_NAME_MAP 裡的
    代碼，確保選出的都是清單外的新面孔）
寫進 data/weekly_institutional_picks.json，之後下週一再被下一次執行覆蓋
掉，達成「自動加入→隔週一自動更換」。這20檔跟固定清單分開存放、分開
顯示（見 rotating_watchlist()），不會混進 STOCK_NAME_MAP。

資料來源：
- 上市(TWSE)：www.twse.com.tw/rwd/zh/fund/T86，免金鑰，可查任何過去日期，
  一次拿全上市市場當天所有股票的三大法人買賣超股數，用來回頭補算「上週」
  完全沒問題。
- 上櫃(TPEx)：www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading，
  免金鑰，但**只回傳「最新一個交易日」，沒有查歷史日期的參數**。所以上櫃
  這邊改成「每天執行時都存一份快照」（archive_tpex_snapshot()，由每天
  的 notify_email.py 排程順便呼叫），存到
  data/tpex_3insti_daily/YYYY-MM-DD.json，算整週排行時直接把本週已經
  存到的快照加總，而不是回頭呼叫API查歷史。剛上線的第一週因為還沒有
  存檔，上櫃部分會是空的，只會排出上市股，屬預期行為。

用法：
    python3 institutional_ranking.py archive   # 每天執行：存今天的上櫃法人快照
    python3 institutional_ranking.py rotate    # 每週一執行：算上週排行、覆寫輪替名單
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

TWSE_T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_3INSTI_URL = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"

DATA_DIR = Path(__file__).parent / "data"
TPEX_ARCHIVE_DIR = DATA_DIR / "tpex_3insti_daily"
WEEKLY_PICKS_FILE = DATA_DIR / "weekly_institutional_picks.json"

TOP_N = 10  # 買超前N + 賣超前N


def _to_int(value) -> int:
    if value is None:
        return 0
    s = str(value).strip().replace(",", "")
    if not s or s in ("-", "--"):
        return 0
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


# ------------------------------------------------------------------
# 上市(TWSE)：可查任何過去日期，一次拿全上市市場當天三大法人買賣超股數
# ------------------------------------------------------------------
def fetch_twse_daily(trade_date: date) -> Dict[str, dict]:
    """回傳 {股票代號: {"name":, "net": 三大法人買賣超股數}}；非交易日/查詢
    失敗都回傳空 dict（不讓整週的加總掛掉，該天就當作沒有資料略過）。"""
    date_str = trade_date.strftime("%Y%m%d")
    try:
        resp = requests.get(
            TWSE_T86_URL,
            params={"response": "json", "date": date_str, "selectType": "ALL"},
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return {}

    if payload.get("stat") != "OK":
        return {}

    result = {}
    for row in payload.get("data", []):
        if len(row) < 19:
            continue
        code = str(row[0]).strip()
        name = str(row[1]).strip()
        net = _to_int(row[18])  # 「三大法人買賣超股數」固定是第19欄
        result[code] = {"name": name, "net": net}
    return result


# ------------------------------------------------------------------
# 上櫃(TPEx)：開放資料只有「最新一天」，靠每天執行時自己存快照累積整週
# ------------------------------------------------------------------
def fetch_tpex_latest_snapshot() -> Dict[str, dict]:
    try:
        resp = requests.get(TPEX_3INSTI_URL, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}

    result = {}
    for row in data:
        code = str(row.get("SecuritiesCompanyCode", "")).strip()
        if not code:
            continue
        name = str(row.get("CompanyName", "")).strip()
        # 欄位名稱可能因API版本略有差異，依序嘗試幾種可能的合計淨額欄位
        net_raw = (
            row.get("TotalDifference")
            or row.get("TotalNet")
            or row.get("三大法人買賣超股數")
        )
        result[code] = {"name": name, "net": _to_int(net_raw)}
    return result


_ARCHIVE_RETENTION_DAYS = 14  # 只有算「上週」排行會用到，留2週緩衝就夠，避免檔案無限累積


def _prune_old_archives(keep_after: date) -> None:
    if not TPEX_ARCHIVE_DIR.exists():
        return
    for path in TPEX_ARCHIVE_DIR.glob("*.json"):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if file_date < keep_after:
            path.unlink(missing_ok=True)


def archive_tpex_snapshot(trade_date: date) -> Optional[Path]:
    """存一份「現在」查到的上櫃法人快照，檔名用 trade_date 標記。呼叫端
    通常是每天早上執行的通知腳本，此時查到的就是前一交易日收盤後的資料，
    用 trade_date=該次執行對應的交易日 即可。抓取失敗或無資料回傳 None，
    不會覆寫掉舊檔（避免一次失敗就把已經存好的資料弄丟空檔案）。"""
    snapshot = fetch_tpex_latest_snapshot()
    if not snapshot:
        return None
    TPEX_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = TPEX_ARCHIVE_DIR / f"{trade_date.isoformat()}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    _prune_old_archives(trade_date - timedelta(days=_ARCHIVE_RETENTION_DAYS))
    return path


def _load_tpex_archive(trade_date: date) -> Dict[str, dict]:
    path = TPEX_ARCHIVE_DIR / f"{trade_date.isoformat()}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ------------------------------------------------------------------
# 整週加總 + 排行
# ------------------------------------------------------------------
def _week_weekdays(reference: date) -> List[date]:
    """回傳 reference 那一週的週一~週五日期。"""
    monday = reference - timedelta(days=reference.weekday())
    return [monday + timedelta(days=i) for i in range(5)]


def aggregate_week(reference_date: date) -> Dict[tuple, dict]:
    """
    加總 reference_date 那一週（週一~週五）的三大法人買賣超淨額。
    上市：直接查歷史日期加總，5天都查得到。
    上櫃：只能用已經存檔的快照加總，缺檔的交易日就跳過（不會讓整週掛掉，
    只是樣本天數會不足5天，剛上線的第一週甚至會是0天/完全空）。
    回傳 {(代碼, 來源市場): {"name":, "net":, "days": 實際加總的天數}}
    """
    weekdays = _week_weekdays(reference_date)
    totals: Dict[tuple, dict] = {}

    for d in weekdays:
        for code, info in fetch_twse_daily(d).items():
            key = (code, "TWSE")
            entry = totals.setdefault(key, {"name": info["name"], "net": 0, "days": 0})
            entry["net"] += info["net"]
            entry["days"] += 1

        for code, info in _load_tpex_archive(d).items():
            key = (code, "TPEX")
            entry = totals.setdefault(key, {"name": info["name"], "net": 0, "days": 0})
            entry["net"] += info["net"]
            entry["days"] += 1

    return totals


def _is_etf(code: str) -> bool:
    """台股ETF/受益憑證代碼一律「00」開頭（0050、00878、00631L...），
    申購/贖回股數規模遠大於一般個股的法人買賣，混進來會直接洗掉真正的
    個股訊號，所以法人買賣超排行預設排除，只留一般個股。"""
    return code.startswith("00")


def compute_weekly_top(reference_date: date, exclude_codes: set, top_n: int = TOP_N) -> dict:
    """
    排出「買超前top_n + 賣超前top_n」，跳過已經在 exclude_codes（現有固定
    清單的代碼）裡的股票、以及ETF（見 _is_etf），確保排行選出的都是清單
    外、有代表性的個股新面孔。
    """
    totals = aggregate_week(reference_date)
    rows = [
        {"code": code, "market": "TW", "name": v["name"], "net": v["net"], "days": v["days"]}
        for (code, _src), v in totals.items()
        if code not in exclude_codes and not _is_etf(code)
    ]

    buy = sorted([r for r in rows if r["net"] > 0], key=lambda r: r["net"], reverse=True)[:top_n]
    sell = sorted([r for r in rows if r["net"] < 0], key=lambda r: r["net"])[:top_n]

    weekdays = _week_weekdays(reference_date)
    tpex_days_available = sum(1 for d in weekdays if _load_tpex_archive(d))

    return {
        "week_start": weekdays[0].isoformat(),
        "week_end": weekdays[-1].isoformat(),
        "generated_at": date.today().isoformat(),
        "buy": buy,
        "sell": sell,
        "tpex_days_available": tpex_days_available,
    }


def save_weekly_picks(result: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEEKLY_PICKS_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_weekly_picks() -> dict:
    if not WEEKLY_PICKS_FILE.exists():
        return {}
    try:
        return json.loads(WEEKLY_PICKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def rotating_watchlist() -> list:
    """
    給 streamlit 頁面/notify_email.py 用的清單格式：[(name, code, market, side), ...]，
    跟 stock_core.unique_watchlist() 的 (name, code, market) 慣例一致，多帶一個
    side（"買超"/"賣超"）方便UI標註。market 統一回傳 "TW"（不論上市/上櫃，
    FinMind 的 get_stock_data 都是用同一組 TaiwanStockPrice + market="TW" 查）。
    """
    picks = load_weekly_picks()
    result = []
    for r in picks.get("buy", []):
        result.append((r["name"], r["code"], "TW", "買超"))
    for r in picks.get("sell", []):
        result.append((r["name"], r["code"], "TW", "賣超"))
    return result


if __name__ == "__main__":
    import sys

    from stock_core import STOCK_NAME_MAP

    cmd = sys.argv[1] if len(sys.argv) > 1 else "archive"
    existing_codes = {code for _, (code, _market) in STOCK_NAME_MAP.items()}

    if cmd == "archive":
        # 每天早上執行：把「現在」查到的上櫃法人最新快照存檔，檔名用今天
        # 日期標記（此時查到的是前一交易日收盤後的資料）。
        saved_path = archive_tpex_snapshot(date.today())
        print(f"已存上櫃法人快照：{saved_path}" if saved_path else "上櫃法人快照抓取失敗或無資料，略過本次存檔")

    elif cmd == "rotate":
        # 每週一執行：算「上週」的排行，覆寫輪替名單。
        today = date.today()
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(days=7)
        weekly_result = compute_weekly_top(last_monday, existing_codes)
        save_weekly_picks(weekly_result)
        print(json.dumps(weekly_result, ensure_ascii=False, indent=2))

    else:
        print(f"未知指令：{cmd}（可用 archive / rotate）")
        sys.exit(1)
