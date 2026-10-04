"""
元大投顧（或其他投顧）早會報告「重點股」提及，近兩週輪替觀察清單。

跟 institutional_ranking.py／youtube_stock_mentions.py／jason_stock_mentions.py
同一套「近期提到就留著、太舊自動換掉」邏輯，差別在於**沒有可以自動抓取的公開
來源**——元大投顧的早會摘要/專欄是使用者自己下載/收到的PDF，沒有試算表或
RSS可以打，所以這裡不做fetch，改成由使用者（或Claude讀完PDF之後）呼叫
record_report() 把當篇報告解讀出來的「重點股」清單記錄進來，寫入
data/yuanta_mentions_log.json（原始逐篇記錄，用(日期,標籤)當key，重複記錄
同一篇報告會覆蓋舊的那一筆、不會重複累加），再重新彙整近LOOKBACK_DAYS天內
還沒過期的提及（依報告日期，不是記錄時間），覆寫 data/weekly_yuanta_picks.json
（彙整結果，給 rotating_watchlist() 讀）。跟固定清單（STOCK_NAME_MAP）已經
有的代碼會在記錄時被排除。

用法：
    python3 yuanta_stock_mentions.py add 2026-10-01 "元大投顧10/01專欄重點股" 6196:帆宣 6257:矽格 8039:台虹 3211:順達
    python3 yuanta_stock_mentions.py show   # 印出目前彙整結果
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"
LOG_FILE = DATA_DIR / "yuanta_mentions_log.json"
WEEKLY_PICKS_FILE = DATA_DIR / "weekly_yuanta_picks.json"

SOURCE_LABEL = "元大投顧重點股"
LOOKBACK_DAYS = 14  # 近兩週，跟其他三份輪替名單同一個節奏


def load_log() -> List[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_log(entries: List[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def record_report(report_date: str, label: str, picks: List[Tuple[str, str]],
                   exclude_codes: Optional[set] = None) -> dict:
    """
    記錄一篇報告的重點股清單（picks是[(code, name), ...]），同一個
    (report_date, label) 重複呼叫會覆蓋舊的那一筆，不會重複累加。
    exclude_codes（通常是STOCK_NAME_MAP既有代碼）會在記錄前先濾掉，固定
    清單裡的股票不需要被標成「新面孔」候選。回傳重新彙整後的 weekly
    picks dict（同時已寫入 WEEKLY_PICKS_FILE）。
    """
    exclude_codes = exclude_codes or set()
    filtered = [{"code": code, "name": name} for code, name in picks if code not in exclude_codes]

    entries = load_log()
    entries = [e for e in entries if not (e["date"] == report_date and e["label"] == label)]
    entries.append({"date": report_date, "label": label, "picks": filtered})
    save_log(entries)

    weekly = recompute_weekly(entries)
    save_weekly_mentions(weekly)
    return weekly


def recompute_weekly(entries: List[dict], today: Optional[date] = None,
                      lookback_days: int = LOOKBACK_DAYS) -> dict:
    """
    彙整entries裡「還沒過期」的提及（依report_date，不是記錄時間），同一檔
    股票可能被不只一篇報告提到，mentions裡的reports會列出全部來源報告。
    """
    today = today or date.today()
    cutoff = today - timedelta(days=lookback_days)

    mentions: Dict[str, dict] = {}
    for entry in entries:
        try:
            report_date = date.fromisoformat(entry["date"])
        except (KeyError, ValueError):
            continue
        if report_date < cutoff:
            continue
        for pick in entry.get("picks", []):
            code, name = pick["code"], pick["name"]
            info = mentions.setdefault(code, {"name": name, "reports": []})
            info["reports"].append({"date": entry["date"], "label": entry["label"]})

    return {
        "generated_at": date.today().isoformat(),
        "lookback_days": lookback_days,
        "mentions": [
            {"code": code, "name": info["name"], "reports": info["reports"]}
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
    [(name, code, market, source_label), ...]，跟其他三份輪替名單同一個慣例
    （market固定回傳"TW"——元大投顧報告目前只涵蓋台股個股）。
    """
    picks = load_weekly_mentions()
    return [(m["name"], m["code"], "TW", SOURCE_LABEL) for m in picks.get("mentions", [])]


def _parse_pick_arg(arg: str) -> Tuple[str, str]:
    code, _, name = arg.partition(":")
    if not name:
        raise ValueError(f"格式錯誤：'{arg}'，重點股參數要用 代碼:名稱（例如 6257:矽格）")
    return code.strip(), name.strip()


if __name__ == "__main__":
    from stock_core import STOCK_NAME_MAP

    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    existing_codes = {code for _, (code, _market) in STOCK_NAME_MAP.items()}

    if cmd == "add":
        if len(sys.argv) < 5:
            print("用法: python3 yuanta_stock_mentions.py add <日期YYYY-MM-DD> <標籤> <代碼:名稱> [<代碼:名稱> ...]")
            sys.exit(1)
        report_date, label = sys.argv[2], sys.argv[3]
        picks = [_parse_pick_arg(a) for a in sys.argv[4:]]
        weekly = record_report(report_date, label, picks, existing_codes)
        print(json.dumps(weekly, ensure_ascii=False, indent=2))

    elif cmd == "show":
        print(json.dumps(load_weekly_mentions(), ensure_ascii=False, indent=2))

    else:
        print(f"未知指令：{cmd}（可用 add / show）")
        sys.exit(1)
