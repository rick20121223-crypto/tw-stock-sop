"""
Jason在群組聊天裡提到的股票，每週輪替觀察清單。

跟institutional_ranking.py／youtube_stock_mentions.py同一套「每週一
覆寫」邏輯：從一份Google試算表（Jason的聊天記錄，由某種自動化把訊息
匯入試算表，這裡只負責讀）抓LOOKBACK_DAYS天內的訊息，比對每則訊息
「有沒有提到」某檔股票，覆寫進data/weekly_jason_picks.json，隔週一
再被下一次執行覆蓋掉。跟固定清單（STOCK_NAME_MAP）已經有的代碼會被
排除，確保排出來的都是清單外的新面孔。

資料來源：Google試算表的CSV匯出網址（https://docs.google.com/
spreadsheets/d/<ID>/export?format=csv），前提是試算表要維持「知道連結
的人都能檢視」的分享設定──不需要金鑰，但如果之後分享設定被改回
「僅限特定人」，這裡會抓不到資料（fetch_messages()回傳空list，不會
讓整週執行掛掉，只是這份輪替名單會是空的）。試算表欄位固定是
「時間,寄件人,訊息內容」。

「有沒有提到」用兩種規則做代號比對，因為Jason的訊息型態差異很大：
1. 短訊息（長度<=SHORT_MSG_THRESHOLD，去掉網址之後）：例如「買些6274」
   「6173要減資」，這種Jason自己打的短句，只要出現有效股票代號就直接
   採信，不用旁邊有公司名稱陪襯──他打字不會意外打出一個剛好是股票
   代號的數字。
2. 長訊息（例如轉貼的新聞全文、公司重大訊息公告）：代號附近
   (PROXIMITY_WINDOW字元內)要出現對應公司名稱前2個字才算確認，避免
   長文裡的年份、金額等數字剛好對到某檔冷門股代號（跟
   youtube_stock_mentions.py用同一個理由/同一套邏輯）。
兩種規則都會先把「數字-數字」型態的價格區間（例如"1435-1460都可"）
從文字裡拿掉，不然會被誤判成兩檔股票代號。

用法：
    python3 jason_stock_mentions.py rotate   # 每週一執行：算最近兩週提到的股票、覆寫輪替名單
"""

import csv
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import requests

DATA_DIR = Path(__file__).parent / "data"
WEEKLY_PICKS_FILE = DATA_DIR / "weekly_jason_picks.json"

SHEET_ID = "1VW0m_sH4UE3pEKVp5wQ2hSz3FRMXADNnA11NfozYAzA"
SENDER_FILTER = "Jason"  # 試算表目前只有Jason的訊息，篩選是防呆，避免以後變成多人群組時混進別人的話
SOURCE_LABEL = "Jason提及"

LOOKBACK_DAYS = 14  # 近兩週，跟institutional_ranking/youtube_stock_mentions同一個節奏
SHORT_MSG_THRESHOLD = 80  # 字元數，去URL/價格區間之後；以內＝Jason自己的短句，以上＝可能是轉貼長文
PROXIMITY_WINDOW = 20  # 長訊息裡，股票代號跟公司名稱要在文字上相距多近才算「確認提到」
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

_URL_RE = re.compile(r"https?://\S+")
_PRICE_RANGE_RE = re.compile(r"\d{3,6}\s*[-~至到]\s*\d{3,6}")  # 例如"1435-1460"是價格區間，不是兩檔代號
_CODE_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")


def fetch_messages(sheet_id: str = SHEET_ID, lookback_days: int = LOOKBACK_DAYS) -> List[dict]:
    """
    用Google試算表的CSV匯出網址抓訊息（試算表要設定「知道連結的人都能
    檢視」），只回傳lookback_days天內、寄件人符合SENDER_FILTER的訊息。
    回傳 [{time, text}]，抓取失敗（網路問題、分享設定被收回等）回傳
    空list，不算錯誤。
    """
    try:
        resp = requests.get(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/export",
            params={"format": "csv"}, timeout=20,
        )
        resp.raise_for_status()
    except Exception:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    messages = []
    try:
        reader = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))
        for row in reader:
            if row.get("寄件人") != SENDER_FILTER:
                continue
            text = (row.get("訊息內容") or "").strip()
            if not text:
                continue
            try:
                sent_at = datetime.fromisoformat(row["時間"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if sent_at < cutoff:
                continue
            messages.append({"time": sent_at.isoformat(), "text": text})
    except Exception:
        return []
    return messages


def fetch_stock_universe(token: str = "") -> Dict[str, str]:
    """回傳 {股票代號: 公司名稱}，涵蓋全部上市櫃（FinMind TaiwanStockInfo）。"""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo"},
                             headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return {}
    return {
        str(row["stock_id"]).strip(): str(row["stock_name"]).strip()
        for row in payload.get("data", [])
        if row.get("stock_id")
    }


def _clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _PRICE_RANGE_RE.sub(" ", text)
    return text


def extract_codes(text: str, stock_universe: Dict[str, str],
                   short_threshold: int = SHORT_MSG_THRESHOLD,
                   window: int = PROXIMITY_WINDOW) -> set:
    """
    從一則訊息裡找出「確認提到」的股票代號，見模組docstring的規則1/2。
    """
    cleaned = _clean_text(text)
    found = set()

    if len(cleaned) <= short_threshold:
        for m in _CODE_RE.finditer(cleaned):
            if m.group() in stock_universe:
                found.add(m.group())
        return found

    for m in _CODE_RE.finditer(cleaned):
        code = m.group()
        name = stock_universe.get(code)
        if not name:
            continue
        start, end = max(0, m.start() - window), min(len(cleaned), m.end() + window)
        context = cleaned[start:end]
        name_key = name[:min(2, len(name))]
        if name_key and name_key in context:
            found.add(code)
    return found


def compute_weekly_mentions(exclude_codes: set, token: str = "",
                             lookback_days: int = LOOKBACK_DAYS) -> dict:
    """
    掃過近lookback_days天Jason的訊息，回傳提到哪些（清單外的）股票、
    在哪些訊息提到過。結構：
        {generated_at, lookback_days,
         mentions: [{code, name, messages: [{time, text}]}]}
    """
    messages = fetch_messages(SHEET_ID, lookback_days)
    stock_universe = fetch_stock_universe(token)

    mentions: Dict[str, dict] = {}
    for msg in messages:
        for code in extract_codes(msg["text"], stock_universe):
            if code in exclude_codes:
                continue
            entry = mentions.setdefault(code, {"name": stock_universe[code], "messages": []})
            entry["messages"].append({"time": msg["time"], "text": msg["text"][:60]})

    return {
        "generated_at": date.today().isoformat(),
        "lookback_days": lookback_days,
        "mentions": [
            {"code": code, "name": info["name"], "messages": info["messages"]}
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
    給 notify_email.py 用的清單格式：[(name, code, market, source_label), ...]，
    跟institutional_ranking.rotating_watchlist()/youtube_stock_mentions.
    rotating_watchlist()同一個慣例（market固定回傳"TW"）。
    """
    picks = load_weekly_mentions()
    return [(m["name"], m["code"], "TW", SOURCE_LABEL) for m in picks.get("mentions", [])]


if __name__ == "__main__":
    import os
    import sys

    from stock_core import STOCK_NAME_MAP

    cmd = sys.argv[1] if len(sys.argv) > 1 else "rotate"
    existing_codes = {code for _, (code, _market) in STOCK_NAME_MAP.items()}

    if cmd == "rotate":
        token = os.environ.get("FINMIND_TOKEN", "")
        result = compute_weekly_mentions(existing_codes, token)
        save_weekly_mentions(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"未知指令：{cmd}（可用 rotate）")
        sys.exit(1)
