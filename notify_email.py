"""
每天排程執行一次，分別判讀「長期（週+日）」和「短期（60分+5分）」兩種
SOP結論，只要任一種「跟上次不一樣」就寄一封 Email 通知，內容分成
📅長線留倉異動／⚡短線進場異動 兩個區塊。

被 .github/workflows/notify.yml 排程呼叫，不是給 Streamlit 用的。

需要的環境變數（GitHub Actions Secrets）：
    FINMIND_TOKEN       FinMind API Token（沒有也能跑，用免費額度）
    FUGLE_API_KEY        Fugle 行情 API Key（沒有的話，短期部分會跳過，
                         只判讀長期）
    GMAIL_ADDRESS        寄件/收件用的 Gmail 帳號
    GMAIL_APP_PASSWORD   Gmail「應用程式密碼」（16碼），不是登入密碼

狀態檔：data/last_signals.json，記錄上一次每檔股票的「長期」「短期」
分類，本次執行後會覆寫最新結果並由 workflow 自動 commit 回 repo。

2026-09-15：GitHub 的 schedule cron 沒有 SLA，偶爾會延遲或整次漏跳
（曾發生排定時間過了一小時以上都沒觸發），所以 notify.yml 現在在原本
時間點之外多排了兩個備援時間點。為了避免同一天被觸發兩三次就重複判讀、
重複寄信，main() 一開始會先檢查 signal_log.csv 今天是不是已經跑過，
跑過就直接跳過（見 _already_ran_today()）。

本檔只負責「開盤前一次的每日批次」。開盤時段（09:00~13:30）的盤中
即時複查是另一支獨立腳本 notify_intraday.py（被 .github/workflows/
intraday_check.yml 每15分鐘呼叫一次），只複查本檔判讀出來、目前「短期」
為買進/加碼的股票，共用這裡的 STATE_FILE / SIGNAL_LOG_FILE 與寄信邏輯。
"""

import concurrent.futures
import csv
import json
import os
import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import institutional_ranking as ir
from multi_timeframe_check import full_check
from sop_decision import classify_final, combine_timeframes, evaluate_timeframe
from stock_core import get_intraday_data, run_all_indicators, unique_watchlist

STATE_FILE = Path(__file__).parent / "data" / "last_signals.json"
SIGNAL_LOG_FILE = Path(__file__).parent / "data" / "signal_log.csv"
SIGNAL_CHANGES_FILE = Path(__file__).parent / "data" / "signal_changes.csv"

# 跟網站同一套配色：台股慣例紅漲綠跌（買進/加碼=紅、賣出減碼=綠）
BUCKET_ORDER = {"加碼": 0, "買進": 1, "觀望": 2, "賣出減碼": 3}
BUCKET_STYLE = {
    "加碼":     {"color": "#e53935", "bg": "#fdecea", "icon": "🔺🔺"},
    "買進":     {"color": "#e53935", "bg": "#fdecea", "icon": "🔺"},
    "觀望":     {"color": "#757575", "bg": "#f5f5f5", "icon": "⚪"},
    "賣出減碼": {"color": "#43a047", "bg": "#eaf6ec", "icon": "🔻"},
}

HORIZON_LABEL = {"長期": "📅 長線留倉", "短期": "⚡ 短線進場"}


def full_watchlist():
    """
    固定清單 + 本週法人排行輪替名單，一起餵給每天的判讀。回傳
    [(name, code, market, source), ...]，source 是 "固定" 或
    "法人排行(買超)"/"法人排行(賣超)"，用來在通知信裡標註來源，
    不會混進 STOCK_NAME_MAP 本身。
    """
    result = [(name, code, market, "固定") for name, code, market in unique_watchlist()]
    seen_codes = {code for _, code, _market, _source in result}
    for name, code, market, side in ir.rotating_watchlist():
        if code in seen_codes:  # 理論上排行時已經排除固定清單，這裡是防呆
            continue
        result.append((name, code, market, f"法人排行({side})"))
        seen_codes.add(code)
    return result


def _already_ran_today(today: str) -> bool:
    """
    檢查 signal_log.csv 最後一行的日期是不是就是今天。同一天的多次執行
    一定是「原排程 + 備援排程」重複觸發（同一天不會有兩個不同日期交錯），
    所以只看最後一行就夠，不用整份掃描。
    """
    if not SIGNAL_LOG_FILE.exists():
        return False
    with open(SIGNAL_LOG_FILE, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)  # 檔案不到2個byte（理論上不會發生，防呆）
        last_line = f.readline().decode("utf-8", errors="ignore").strip()
    if not last_line or last_line.startswith("date,"):
        return False
    return last_line.split(",", 1)[0] == today


def load_last_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_signal_log(today: str, rows: list) -> None:
    """
    每天執行都會呼叫（不管有沒有寄信），把每檔股票當天的收盤價、SOP
    結論、判讀屬於「長期」或「短期」都記一筆，方便之後回頭做正式回測。

    row 可以帶自己的 "date"（長期用的是日K實際收盤日期，見
    multi_timeframe_check.full_check() 回傳的「資料日期」），沒帶就用
    today（短期沿用原本行為，一律用呼叫當下的系統日期）。寫入前會先
    讀現有檔案，同一個 (code, horizon, date, signal, close) 如果已經
    一模一樣記錄過就跳過不重複寫——這是為了避免盤前提早跑、假日手動
    重跑（workflow_dispatch）等情況抓到同一根日K、判讀結果完全相同，
    卻被記成兩個不同的「日子」，讓之後拿這份log做回測時被假的重複快照
    誤導。注意這裡連 signal/close 也一起比對，不是只比對日期：短期一天
    內本來就可能因為盤中複查記錄到好幾筆「同一天、不同訊號」的真實變化
    （見 notify_intraday.py），這種不能被當成重複而濾掉，只有內容也完全
    相同時才視為真正的重複快照。
    """
    SIGNAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SIGNAL_LOG_FILE.exists()

    already_logged = set()
    if file_exists:
        with open(SIGNAL_LOG_FILE, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_logged.add((row["code"], row["horizon"], row["date"],
                                     row["signal"], row["close"]))

    new_rows = []
    for r in rows:
        row_date = r.get("date") or today
        key = (r["code"], r["horizon"], row_date, r["signal"], str(r["close"]))
        if key in already_logged:
            continue
        already_logged.add(key)
        new_rows.append((row_date, r))

    if not new_rows:
        return

    with open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "name", "code", "market", "horizon", "close", "signal"])
        for row_date, r in new_rows:
            writer.writerow([row_date, r["name"], r["code"], r["market"],
                              r["horizon"], r["close"], r["signal"]])


def append_signal_changes(changes: list, horizon: str) -> None:
    """
    事件式紀錄：只在訊號「真的改變」時才寫一筆，跟 signal_log.csv 那種
    「不管有沒有變化、每次執行都記一筆快照」不同。目的是之後要做「照
    訊號進出場」的模擬回測時，可以直接照這份事件序列組出每一段「進場
    時間+價格 → 出場時間+價格」的持有區間，不用再從snapshot log自己
    猜訊號到底是哪個時間點變的（尤其短期一天可能複查很多次）。
    changes 的每個元素預期帶 name/code/source/prev/new/close。
    """
    if not changes:
        return
    SIGNAL_CHANGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SIGNAL_CHANGES_FILE.exists()
    now = datetime.now().isoformat(timespec="seconds")
    with open(SIGNAL_CHANGES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "name", "code", "market", "horizon",
                              "prev_signal", "new_signal", "close", "source"])
        for c in changes:
            writer.writerow([now, c["name"], c["code"], c.get("market", "TW"), horizon,
                              c["prev"], c["new"], c.get("close"), c.get("source", "固定")])


def send_email(subject: str, plain_body: str, html_body: str,
                gmail_address: str, app_password: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = gmail_address
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, app_password)
        server.sendmail(gmail_address, [gmail_address], msg.as_string())


def _html_wrap(title: str, subtitle: str, body_html: str) -> str:
    return f"""\
<html><head><meta charset="utf-8"></head><body style="margin:0; padding:0; background:#f0f2f5;">
<div style="max-width:520px; margin:0 auto; padding:24px 16px;
            font-family:-apple-system,'PingFang TC','Microsoft JhengHei',Arial,sans-serif;">
  <h2 style="margin:0 0 4px; color:#222; font-size:20px;">📊 {title}</h2>
  <p style="margin:0 0 20px; color:#888; font-size:13px;">{subtitle}</p>
  {body_html}
  <p style="margin-top:24px; color:#aaa; font-size:11px; line-height:1.6;">
    僅供輔助判讀，不構成投資建議。📅長線留倉依「週+日」整合結論，
    ⚡短線進場依「60分+5分」整合結論（僅台股個股/ETF，需要Fugle Key）。
  </p>
</div>
</body></html>"""


def _card_html(icon: str, color: str, bg: str, title_line: str, detail_line: str = "") -> str:
    detail = f'<div style="margin-top:2px; color:#555; font-size:13px;">{detail_line}</div>' if detail_line else ""
    return f"""\
  <div style="border-left:4px solid {color}; background:{bg}; border-radius:4px;
              padding:12px 14px; margin-bottom:10px;">
    <div style="font-size:15px; color:#222;">
      <span style="font-size:17px;">{icon}</span>
      <b>{title_line}</b>
    </div>
    {detail}
  </div>
"""


def _section_html(title: str, cards_html: str) -> str:
    if not cards_html:
        return ""
    return f'<h3 style="margin:20px 0 8px; color:#444; font-size:15px;">{title}</h3>{cards_html}'


def build_first_run_email(today: str, new_state: dict) -> tuple:
    rows = sorted(
        new_state.values(),
        key=lambda v: (BUCKET_ORDER.get(v.get("長期"), 9), BUCKET_ORDER.get(v.get("短期"), 9)),
    )

    plain_lines = [f"台股 SOP 通知已啟用（{today}）", "", "今天的起始分類（長期／短期）：", ""]
    for v in rows:
        short = v.get("短期") or "—"
        badge = _source_badge(v.get("來源"))
        plain_lines.append(f"・{badge}{v['名稱']}（{v['代碼']}）：長期={v['長期']}／短期={short}")
    plain_lines += ["", "之後只有分類「變化」時才會再寄信通知你。", "🔄 標記代表這是本週法人買賣超排行自動加入的股票。"]
    plain = "\n".join(plain_lines)

    cards = ""
    for v in rows:
        style = BUCKET_STYLE[v["長期"]]
        short = v.get("短期")
        short_text = f'　短期：<b style="color:{BUCKET_STYLE[short]["color"]}">{short}</b>' if short else ""
        badge = _source_badge(v.get("來源"))
        cards += _card_html(
            style["icon"], style["color"], style["bg"],
            f"{badge}{v['名稱']}（{v['代碼']}）",
            f'長期：<span style="color:{style["color"]}; font-weight:600;">{v["長期"]}</span>{short_text}',
        )
    html_body = cards + (
        '<p style="color:#888; font-size:13px; margin-top:16px;">'
        "之後只有分類「變化」時才會再寄信通知你。</p>"
    )
    html = _html_wrap("台股 SOP 通知已啟用", today, html_body)
    return plain, html


def _source_badge(source: str) -> str:
    return "🔄 " if source and source != "固定" else ""


def _changes_to_plain(changes: list) -> list:
    lines = []
    for c in changes:
        style = BUCKET_STYLE[c["new"]]
        badge = _source_badge(c.get("source"))
        suffix = f"　[{c['source']}]" if badge else ""
        lines.append(f"{style['icon']} {badge}{c['name']}（{c['code']}）：{c['prev']} → {c['new']}{suffix}")
    return lines


def _changes_to_html(changes: list) -> str:
    cards = ""
    for c in changes:
        style = BUCKET_STYLE[c["new"]]
        badge = _source_badge(c.get("source"))
        source_html = (
            f'<div style="color:#999; font-size:11px; margin-top:2px;">🔄 {c["source"]}</div>'
            if badge else ""
        )
        detail = (
            f'<span style="color:#999;">{c["prev"]}</span>'
            f' <span style="color:#bbb;">→</span> '
            f'<span style="color:{style["color"]}; font-weight:700; font-size:15px;">{c["new"]}</span>'
            f'{source_html}'
        )
        cards += _card_html(style["icon"], style["color"], style["bg"],
                             f"{c['name']}（{c['code']}）", detail)
    return cards


def _summarize_errors(errors: list) -> list:
    """
    把「Fugle 429 Rate limit」這種同一原因、常常一次影響一大串股票的錯誤，
    收斂成一行摘要，避免通知信被幾十行幾乎一樣的錯誤訊息洗版；其他比較
    少見、真的需要留意的錯誤（代碼錯誤、FinMind 額度用盡等）維持逐行列出。
    """
    rate_limited, other = [], []
    for e in errors:
        (rate_limited if "429 Rate limit exceeded" in e else other).append(e)
    summary = list(other)
    if rate_limited:
        names = [e.split("：", 1)[0].removesuffix("短期").removesuffix("長期") for e in rate_limited]
        summary.append(
            f"{len(rate_limited)} 檔短期資料因 Fugle 429 流量限制暫時略過（{'、'.join(names)}），"
            "之後排程會自動重試，不代表訊號有問題"
        )
    return summary


def build_change_email(today: str, long_changes: list, short_changes: list, errors: list) -> tuple:
    errors = _summarize_errors(errors)
    plain_lines = [f"台股 SOP 訊號異動通知（{today}）"]
    if long_changes:
        plain_lines += ["", HORIZON_LABEL["長期"] + "異動：", ""] + _changes_to_plain(long_changes)
    if short_changes:
        plain_lines += ["", HORIZON_LABEL["短期"] + "異動：", ""] + _changes_to_plain(short_changes)
    if errors:
        plain_lines += ["", "⚠️ 以下股票資料取得失敗："] + errors
    plain = "\n".join(plain_lines)

    html_body = ""
    html_body += _section_html(HORIZON_LABEL["長期"] + "異動", _changes_to_html(long_changes))
    html_body += _section_html(HORIZON_LABEL["短期"] + "異動", _changes_to_html(short_changes))
    if errors:
        err_html = "".join(f"<div>⚠️ {e}</div>" for e in errors)
        html_body += (
            '<div style="margin-top:16px; padding:10px 14px; background:#fff8e1; '
            'border-radius:4px; color:#8a6d00; font-size:12px;">' + err_html + "</div>"
        )
    html = _html_wrap("台股 SOP 訊號異動通知", today, html_body)
    return plain, html


def _check_long(code: str, market: str, api_token: str):
    """
    長期：週+日整合，跟首頁「長線留倉」同一套。
    回傳 (分類, 收盤, 資料日期, 錯誤訊息)——資料日期是日K最後一根的實際
    交易日，用來讓 signal_log.csv 記錄真正的交易日而不是呼叫當下的系統
    日期（見 append_signal_log()）。
    """
    try:
        result = full_check(code, market, api_token, "", "2024-01-01")
        return classify_final(result["最終建議"]), result.get("收盤"), result.get("資料日期"), None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, str(exc)


def _check_short(code: str, market: str, fugle_api_key: str):
    """
    短期：60分+5分整合，跟「短線進場」頁同一套。只支援台股個股/ETF，
    且需要Fugle Key；不適用時回傳 (None, None, None)，不算錯誤。
    """
    if market != "TW" or not fugle_api_key:
        return None, None, None
    try:
        verdicts = {}
        close = None
        df60 = get_intraday_data(code, "60", fugle_api_key)
        if not df60.empty:
            df60 = run_all_indicators(df60, "60分")
            verdicts["60分"] = evaluate_timeframe(df60, "60分")
            close = float(df60.iloc[-1]["close"])
        df5 = get_intraday_data(code, "5", fugle_api_key)
        if not df5.empty:
            df5 = run_all_indicators(df5, "5分")
            verdicts["5分"] = evaluate_timeframe(df5, "5分")
            close = float(df5.iloc[-1]["close"])
        if not verdicts:
            return None, None, "60分/5分皆無資料"
        result = combine_timeframes(verdicts)
        return classify_final(result["最終建議"]), close, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def main() -> None:
    api_token = os.environ.get("FINMIND_TOKEN", "")
    fugle_api_key = os.environ.get("FUGLE_API_KEY", "")
    gmail_address = os.environ.get("GMAIL_ADDRESS", "")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not gmail_address or not gmail_app_password:
        raise SystemExit("缺少環境變數 GMAIL_ADDRESS 或 GMAIL_APP_PASSWORD，無法寄信。")
    if not fugle_api_key:
        print("⚠️ 未設定 FUGLE_API_KEY，本次只會判讀長期（週+日），短期（60分/5分）略過。")

    today = str(date.today())
    if _already_ran_today(today):
        print(f"{today}：今天已經跑過一次了（可能是備援排程時間點重複觸發），跳過本次執行，不重複判讀、不重複寄信。")
        return

    # 每天執行時順便存一份上櫃法人快照（TPEx開放資料只有「最新一天」，
    # 靠每天存檔累積，才能在每週一算出「本週法人買賣超排行」，見
    # institutional_ranking.py）。存檔失敗不影響本次通知主流程。
    try:
        archived = ir.archive_tpex_snapshot(date.today())
        print(f"上櫃法人快照已存檔：{archived}" if archived else "上櫃法人快照抓取失敗或無資料，略過本次存檔")
    except Exception as exc:  # noqa: BLE001
        print(f"上櫃法人快照存檔發生例外（不影響本次通知）：{exc}")

    last_state = load_last_state()
    is_first_run = not last_state

    new_state = {}
    long_changes, short_changes = [], []
    errors = []
    log_rows = []

    def _check_one(name, code, market, source):
        long_bucket, long_close, long_date, long_err = _check_long(code, market, api_token)
        short_bucket, short_close, short_err = _check_short(code, market, fugle_api_key)
        return (name, code, market, source, long_bucket, long_close, long_date, long_err,
                short_bucket, short_close, short_err)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_check_one, name, code, market, source)
            for name, code, market, source in full_watchlist()
        ]
        for future in concurrent.futures.as_completed(futures):
            (name, code, market, source, long_bucket, long_close, long_date, long_err,
             short_bucket, short_close, short_err) = future.result()

            if long_err is not None:
                errors.append(f"{name}（{code}）長期：{long_err}")
            if short_err is not None:
                errors.append(f"{name}（{code}）短期：{short_err}")
            if long_bucket is None:
                continue  # 長期是核心判讀，抓不到就整檔略過

            key = f"{code}_{market}"
            new_state[key] = {"名稱": name, "代碼": code, "長期": long_bucket,
                               "短期": short_bucket, "來源": source}

            log_rows.append({"name": name, "code": code, "market": market,
                              "horizon": "長期", "close": long_close, "signal": long_bucket,
                              "date": long_date})
            if short_bucket is not None:
                log_rows.append({"name": name, "code": code, "market": market,
                                  "horizon": "短期", "close": short_close, "signal": short_bucket})

            prev = last_state.get(key)
            if prev:
                if prev.get("長期") and prev["長期"] != long_bucket:
                    long_changes.append({"name": name, "code": code, "market": market, "source": source,
                                          "prev": prev["長期"], "new": long_bucket, "close": long_close})
                if short_bucket is not None and prev.get("短期") and prev["短期"] != short_bucket:
                    short_changes.append({"name": name, "code": code, "market": market, "source": source,
                                           "prev": prev["短期"], "new": short_bucket, "close": short_close})

    save_state(new_state)
    append_signal_log(today, log_rows)
    append_signal_changes(long_changes, "長期")
    append_signal_changes(short_changes, "短期")

    if is_first_run:
        plain, html = build_first_run_email(today, new_state)
        print(plain)
        send_email(f"[台股SOP] 通知已啟用（{today}）", plain, html, gmail_address, gmail_app_password)
        return

    if not long_changes and not short_changes:
        print(f"{today}：無訊號變化，不寄信。")
        if errors:
            print("以下股票取得資料失敗：\n" + "\n".join(errors))
        return

    long_changes.sort(key=lambda c: BUCKET_ORDER.get(c["new"], 9))
    short_changes.sort(key=lambda c: BUCKET_ORDER.get(c["new"], 9))
    plain, html = build_change_email(today, long_changes, short_changes, errors)
    print(plain)
    send_email(f"[台股SOP] 訊號異動通知（{today}）", plain, html, gmail_address, gmail_app_password)


if __name__ == "__main__":
    main()
