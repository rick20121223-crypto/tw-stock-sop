"""
投顧報告（元大投顧、宏遠投顧...等）「重點股/買進/持有」提及，近兩週輪替觀察
清單。

跟 institutional_ranking.py／youtube_stock_mentions.py／jason_stock_mentions.py
同一套「近期提到就留著、太舊自動換掉」邏輯，差別在於**沒有可以自動抓取的公開
來源**——這些投顧報告是使用者自己下載/收到的PDF，沒有試算表或RSS可以打，
所以這裡不做fetch，改成由使用者（或Claude讀完PDF之後）呼叫 record_report()
把當篇報告解讀出來的「重點股/買進/持有」清單記錄進來，寫入
data/brokerage_mentions_log.json（原始逐篇記錄，用(日期,券商,標籤)當key，
重複記錄同一篇報告會覆蓋舊的那一筆、不會重複累加），再重新彙整近
LOOKBACK_DAYS天內還沒過期的提及（依報告日期，不是記錄時間），覆寫
data/weekly_brokerage_picks.json（彙整結果，給 rotating_watchlist() 讀）。
跟固定清單（STOCK_NAME_MAP）已經有的代碼會在記錄時被排除。

這份清單本來只給「元大投顧」用（舊檔名yuanta_stock_mentions.py），後來發現
使用者也會丟其他投顧（例如宏遠投顧）的個股深度報告，才通用化成現在這版——
每一篇記錄都要帶firm（券商名稱），同一檔股票被不同券商提及時，
rotating_watchlist()的來源標籤會把券商名稱合併顯示（例如"元大投顧＋宏遠投顧"），
讓多個獨立投顧同時關注的訊號更明顯。

用法：
    python3 brokerage_stock_mentions.py add 2026-10-01 元大投顧 "10/01專欄重點股" 6196:帆宣 6257:矽格
    python3 brokerage_stock_mentions.py add 2026-09-22 宏遠投顧 "個股報告" 2368:金像電:買進:1370
    # 重點股清單參數格式：代碼:名稱[:評等[:目標價]]，評等/目標價可省略
    python3 brokerage_stock_mentions.py show   # 印出目前彙整結果
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"
LOG_FILE = DATA_DIR / "brokerage_mentions_log.json"
WEEKLY_PICKS_FILE = DATA_DIR / "weekly_brokerage_picks.json"

LOOKBACK_DAYS = 14  # 近兩週，跟其他輪替名單同一個節奏
_DEFAULT_FIRM = "元大投顧"  # 舊檔(yuanta_stock_mentions.py)留下的紀錄沒有firm欄位，一律視為元大投顧的提及


def load_log() -> List[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        entries = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    for entry in entries:
        entry.setdefault("firm", _DEFAULT_FIRM)  # 相容舊格式（通用化前只有元大投顧一個來源）
    return entries


def save_log(entries: List[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def record_report(report_date: str, firm: str, label: str, picks: List[dict],
                   exclude_codes: Optional[set] = None) -> dict:
    """
    記錄一篇報告的重點股清單。picks是[{code, name, rating?, target_price?}, ...]，
    rating/target_price是選填（元大投顧的早會重點股通常沒有，宏遠投顧的個股報告
    通常有）。同一個 (report_date, firm, label) 重複呼叫會覆蓋舊的那一筆，不會
    重複累加。exclude_codes（通常是STOCK_NAME_MAP既有代碼）會在記錄前先濾掉，
    固定清單裡的股票不需要被標成「新面孔」候選。回傳重新彙整後的 weekly picks
    dict（同時已寫入 WEEKLY_PICKS_FILE）。
    """
    exclude_codes = exclude_codes or set()
    filtered = [p for p in picks if p["code"] not in exclude_codes]

    entries = load_log()
    entries = [e for e in entries
               if not (e["date"] == report_date and e["firm"] == firm and e["label"] == label)]
    entries.append({"date": report_date, "firm": firm, "label": label, "picks": filtered})
    save_log(entries)

    weekly = recompute_weekly(entries)
    save_weekly_mentions(weekly)
    return weekly


def recompute_weekly(entries: List[dict], today: Optional[date] = None,
                      lookback_days: int = LOOKBACK_DAYS) -> dict:
    """
    彙整entries裡「還沒過期」的提及（依report_date，不是記錄時間）。同一檔股票
    可能被不只一篇報告（甚至不只一家券商）提到，mentions裡的reports會列出全部
    來源報告；rating/target_price取「最新一篇有填這個欄位的報告」的值。
    """
    today = today or date.today()
    cutoff = today - timedelta(days=lookback_days)

    mentions: Dict[str, dict] = {}
    for entry in sorted(entries, key=lambda e: e.get("date", "")):
        try:
            report_date = date.fromisoformat(entry["date"])
        except (KeyError, ValueError):
            continue
        if report_date < cutoff:
            continue
        for pick in entry.get("picks", []):
            code, name = pick["code"], pick["name"]
            info = mentions.setdefault(code, {"name": name, "reports": []})
            report_ref = {"date": entry["date"], "firm": entry["firm"], "label": entry["label"]}
            if pick.get("rating"):
                report_ref["rating"] = pick["rating"]
                info["rating"] = pick["rating"]
            if pick.get("target_price"):
                report_ref["target_price"] = pick["target_price"]
                info["target_price"] = pick["target_price"]
            info["reports"].append(report_ref)

    return {
        "generated_at": date.today().isoformat(),
        "lookback_days": lookback_days,
        "mentions": [
            {"code": code, **info}
            for code, info in sorted(mentions.items())
        ],
    }


def save_weekly_mentions(result: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEEKLY_PICKS_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def load_weekly_mentions() -> dict:
    if not WEEKLY_PICKS_FILE.exists():
        return {}
    try:
        return json.loads(WEEKLY_PICKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def rotating_watchlist() -> list:
    """
    給 notify_email.py／watchlist_sop_check.py 用的清單格式：
    [(name, code, market, source_label), ...]，跟其他輪替名單同一個慣例
    （market固定回傳"TW"——目前這些投顧報告只涵蓋台股個股）。source_label是
    該檔股票被哪些券商提及的聯集，用"＋"合併（例如同時被元大投顧和宏遠投顧
    提及會是"元大投顧＋宏遠投顧"），比單一券商名稱更能反映多方關注的訊號。
    """
    picks = load_weekly_mentions()
    result = []
    for m in picks.get("mentions", []):
        firms = sorted({r["firm"] for r in m["reports"]})
        label = "＋".join(firms)
        result.append((m["name"], m["code"], "TW", label))
    return result


def _parse_pick_arg(arg: str) -> dict:
    parts = arg.split(":")
    if len(parts) < 2:
        raise ValueError(f"格式錯誤：'{arg}'，重點股參數要用 代碼:名稱[:評等[:目標價]]（例如 6257:矽格 或 2368:金像電:買進:1370）")
    pick = {"code": parts[0].strip(), "name": parts[1].strip()}
    if len(parts) >= 3 and parts[2].strip():
        pick["rating"] = parts[2].strip()
    if len(parts) >= 4 and parts[3].strip():
        pick["target_price"] = parts[3].strip()
    return pick


if __name__ == "__main__":
    from stock_core import STOCK_NAME_MAP

    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    existing_codes = {code for _, (code, _market) in STOCK_NAME_MAP.items()}

    if cmd == "add":
        if len(sys.argv) < 6:
            print("用法: python3 brokerage_stock_mentions.py add <日期YYYY-MM-DD> <券商名稱> <標籤> <代碼:名稱[:評等[:目標價]]> [...]")
            sys.exit(1)
        report_date, firm, label = sys.argv[2], sys.argv[3], sys.argv[4]
        picks = [_parse_pick_arg(a) for a in sys.argv[5:]]
        weekly = record_report(report_date, firm, label, picks, existing_codes)
        print(json.dumps(weekly, ensure_ascii=False, indent=2))

    elif cmd == "show":
        print(json.dumps(load_weekly_mentions(), ensure_ascii=False, indent=2))

    else:
        print(f"未知指令：{cmd}（可用 add / show）")
        sys.exit(1)
