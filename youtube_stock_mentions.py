"""
雷老闆YT頻道《股市電電電》提到的股票，每週輪替觀察清單。

跟法人買賣超排行（institutional_ranking.py）同一套「每週一覆寫」邏輯：
往回抓LOOKBACK_DAYS天內頻道發布的影片，比對每部影片「有沒有提到」某檔
股票，覆寫進data/weekly_youtube_picks.json，隔週一再被下一次執行覆蓋
掉，達成「近1~2週有出現就留著，太舊自動被換掉」。跟固定清單
（STOCK_NAME_MAP）已經有的代碼會被排除，確保排出來的都是清單外的新
面孔（跟institutional_ranking.py同一個過濾邏輯）。

「有沒有提到」用兩種方式偵測，取聯集：
1. 影片描述裡的hashtag（例如"#2409友達"）——這是雷老闆自己標的，最準。
2. 逐字稿全文比對——用youtube_transcript_api抓自動字幕，找4位數股票
   代號，且代號附近(PROXIMITY_WINDOW字元內)要出現對應公司名稱前2個字
   才算確認。這個proximity要求是刻意的：口語提到年份、價格等，剛好是
   4位數字又剛好對到某檔冷門股代號的情況並不少見（例如"2027年"剛好
   對到大成鋼2027），只看4位數字太容易誤判；但反過來，AI字幕常常把
   公司名稱轉錯（官方影片說明也寫「字幕有誤屬正常現象」），要求代號+
   名稱同時出現，會讓一些真的有講到、但名稱被轉錯的股票被漏掉——這是
   刻意選擇「寧可少抓，不要抓錯」，因為這份清單只是「讓使用者維持
   印象」的輔助清單，不是交易訊號，抓一堆不相關的股票進來反而讓清單
   失去意義。

抓字幕需要 youtube-transcript-api（見requirements.txt），這個套件是直接
打YouTube的公開字幕端點，不需要金鑰，但**YouTube有時會封鎖雲端/機房的
IP**（GitHub Actions runner正是這種IP），字幕抓取失敗是預期中可能發生
的情況，那部影片就只靠hashtag偵測，不會讓整週執行掛掉。

用法：
    python3 youtube_stock_mentions.py rotate   # 每週一執行：算最近兩週提到的股票、覆寫輪替名單
"""

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

from stock_core import fetch_finmind

DATA_DIR = Path(__file__).parent / "data"
WEEKLY_PICKS_FILE = DATA_DIR / "weekly_youtube_picks.json"

# 頻道字典：名稱 -> YouTube Channel ID，之後要追蹤其他頻道只要加一筆。
# 雷老闆《股市電電電》：https://www.youtube.com/@remus_boss
CHANNELS = {"雷老闆": "UCFsyPpT525Fass_s7fA2qhg"}

LOOKBACK_DAYS = 14  # 近兩週的影片才算，每週一整批覆寫，太舊的會自然被換掉
PROXIMITY_WINDOW = 20  # 股票代號跟公司名稱要在文字上相距多近才算「確認提到」

_RSS_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}
_HASHTAG_CODE_RE = re.compile(r"#(\d{4})(?!\d)")
_CODE_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")


def fetch_recent_videos(channel_id: str, lookback_days: int = LOOKBACK_DAYS) -> Optional[List[dict]]:
    """
    用YouTube公開RSS feed（不需要API金鑰/額度）抓頻道最近發布的影片，只
    回傳lookback_days天內的。RSS通常只保留最新十幾部影片，對幾乎每天
    開播的頻道，兩週內的份量抓得到；如果頻道更新沒那麼頻繁，本來就不會
    漏，只是清單可能比較空。
    回傳 [{video_id, title, description, published}]；「這段期間真的沒有
    符合的影片」回傳空list，但「根本抓不到RSS」（網路問題、頻道ID錯等）
    回傳 None——呼叫端要能分辨這兩種情況，不然會把「抓取失敗」誤判成
    「這週真的沒有任何提及」，把上週好不容易累積的輪替名單洗成空的。
    """
    try:
        resp = requests.get(
            "https://www.youtube.com/feeds/videos.xml",
            params={"channel_id": channel_id}, timeout=20,
        )
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
    except Exception:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    videos = []
    for entry in root.findall("atom:entry", _RSS_NS):
        video_id_el = entry.find("yt:videoId", _RSS_NS)
        published_el = entry.find("atom:published", _RSS_NS)
        if video_id_el is None or published_el is None:
            continue
        try:
            published = datetime.fromisoformat(published_el.text)
        except ValueError:
            continue
        if published < cutoff:
            continue
        title_el = entry.find("atom:title", _RSS_NS)
        desc_el = entry.find("media:group/media:description", _RSS_NS)
        videos.append({
            "video_id": video_id_el.text,
            "title": title_el.text if title_el is not None else "",
            "description": desc_el.text if desc_el is not None else "",
            "published": published.date().isoformat(),
        })
    return videos


def fetch_transcript_text(video_id: str) -> str:
    """
    抓自動字幕全文（依序嘗試中文繁體/簡體/英文）。抓不到（該影片沒有
    字幕、或YouTube擋掉這次請求的IP等）都回傳空字串，呼叫端據此只靠
    hashtag偵測，不算錯誤、不影響其他影片的判讀。
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        result = YouTubeTranscriptApi().fetch(
            video_id, languages=["zh-TW", "zh-Hant", "zh", "zh-CN", "en"])
        return " ".join(snippet.text for snippet in result.snippets)
    except Exception:
        return ""


def fetch_stock_universe(token: str = "") -> Optional[Dict[str, str]]:
    """
    回傳 {股票代號: 公司名稱}，涵蓋全部上市櫃（FinMind TaiwanStockInfo），
    用來判斷逐字稿裡的4位數字是不是真的股票代號、以及取得該代號的正式
    公司名稱（不用逐字稿裡AI可能轉錄錯誤的名稱）。抓取失敗回傳None（不是
    空dict）——呼叫端要能分辨「FinMind真的失敗」跟「回傳但剛好沒資料」，
    不然token過期這種情況會讓整份輪替名單被誤判成「這週沒有任何提及」
    而洗成空的。TaiwanStockInfo是全市場靜態清單，不需要data_id/日期區間，
    沿用stock_core.fetch_finmind()帶空字串即可，跟其他模組共用同一套
    FinMind呼叫/錯誤處理邏輯，不用自己重寫一份requests.get。
    """
    try:
        df = fetch_finmind("TaiwanStockInfo", "", "", "", token)
    except Exception:
        return None
    if df.empty:
        return {}
    return dict(zip(df["stock_id"].astype(str).str.strip(), df["stock_name"].astype(str).str.strip()))


def _hashtag_codes(description: str) -> set:
    return set(_HASHTAG_CODE_RE.findall(description or ""))


def _transcript_codes(text: str, stock_universe: Dict[str, str],
                       window: int = PROXIMITY_WINDOW) -> set:
    confirmed = set()
    for m in _CODE_RE.finditer(text):
        code = m.group()
        name = stock_universe.get(code)
        if not name:
            continue
        start, end = max(0, m.start() - window), min(len(text), m.end() + window)
        context = text[start:end]
        name_key = name[:min(2, len(name))]
        if name_key and name_key in context:
            confirmed.add(code)
    return confirmed


def compute_weekly_mentions(channel_name: str, channel_id: str, exclude_codes: set,
                             token: str = "", lookback_days: int = LOOKBACK_DAYS) -> Optional[dict]:
    """
    掃過頻道近lookback_days天的影片，回傳提到哪些（清單外的）股票、
    在哪些影片提到過。結構：
        {channel, generated_at, lookback_days,
         mentions: [{code, name, videos: [{title, date, video_id}]}]}
    只要RSS或FinMind任一個資料源「真的抓取失敗」（不是抓到但沒資料），
    就回傳None，呼叫端要據此保留上一份輪替名單、不要拿這次的失敗結果去
    覆寫（否則會把上週好不容易累積的清單洗成空的）。
    """
    videos = fetch_recent_videos(channel_id, lookback_days)
    stock_universe = fetch_stock_universe(token)
    if videos is None or stock_universe is None:
        return None

    mentions: Dict[str, dict] = {}
    for v in videos:
        codes = _hashtag_codes(v["description"])
        transcript = fetch_transcript_text(v["video_id"])
        if transcript:
            codes |= _transcript_codes(transcript, stock_universe)

        for code in codes:
            if code in exclude_codes or code not in stock_universe:
                continue  # 已經在固定清單、或不是有效股票代號（年份/價格等巧合數字）
            entry = mentions.setdefault(code, {"name": stock_universe[code], "videos": []})
            entry["videos"].append({
                "title": v["title"], "date": v["published"], "video_id": v["video_id"],
            })

    return {
        "channel": channel_name,
        "generated_at": date.today().isoformat(),
        "lookback_days": lookback_days,
        "mentions": [
            {"code": code, "name": info["name"], "videos": info["videos"]}
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
    跟 institutional_ranking.rotating_watchlist() 同一個慣例（market固定
    回傳"TW"）。
    """
    picks = load_weekly_mentions()
    channel = picks.get("channel", "YT")
    return [(m["name"], m["code"], "TW", f"{channel}YT提及") for m in picks.get("mentions", [])]


if __name__ == "__main__":
    import os
    import sys

    from stock_core import STOCK_NAME_MAP

    cmd = sys.argv[1] if len(sys.argv) > 1 else "rotate"
    channel_key = sys.argv[2] if len(sys.argv) > 2 else "雷老闆"
    existing_codes = {code for _, (code, _market) in STOCK_NAME_MAP.items()}

    if cmd == "rotate":
        channel_id = CHANNELS[channel_key]
        token = os.environ.get("FINMIND_TOKEN", "")
        result = compute_weekly_mentions(channel_key, channel_id, existing_codes, token)
        if result is None:
            print("本次抓取失敗（RSS或FinMind連不上），保留上次的輪替名單不覆寫。")
        else:
            save_weekly_mentions(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"未知指令：{cmd}（可用 rotate）")
        sys.exit(1)
