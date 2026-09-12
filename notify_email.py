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
"""

import concurrent.futures
import csv
import json
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import institutional_ranking as ir
from multi_timeframe_check import full_check
from sop_decision import classify_final, combine_timeframes, evaluate_timeframe
from stock_core import get_intraday_data, run_all_indicators, unique_watchlist

STATE_FILE = Path(__file__).parent / "data" / "last_signals.json"
SIGNAL_LOG_FILE = Path(__file__).parent / "data" / "signal_log.csv"

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
    """
    SIGNAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SIGNAL_LOG_FILE.exists()
    with open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "name", "code", "market", "horizon", "close", "signal"])
        for r in rows:
            writer.writerow([today, r["name"], r["code"], r["market"],
                              r["horizon"], r["close"], r["signal"]])


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


def build_change_email(today: str, long_changes: list, short_changes: list, errors: list) -> tuple:
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
    """長期：週+日整合，跟首頁「長線留倉」同一套。回傳 (分類, 收盤, 錯誤訊息)"""
    try:
        result = full_check(code, market, api_token, "", "2024-01-01")
        return classify_final(result["最終建議"]), result.get("收盤"), None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


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
        long_bucket, long_close, long_err = _check_long(code, market, api_token)
        short_bucket, short_close, short_err = _check_short(code, market, fugle_api_key)
        return (name, code, market, source, long_bucket, long_close, long_err,
                short_bucket, short_close, short_err)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_check_one, name, code, market, source)
            for name, code, market, source in full_watchlist()
        ]
        for future in concurrent.futures.as_completed(futures):
            (name, code, market, source, long_bucket, long_close, long_err,
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
                              "horizon": "長期", "close": long_close, "signal": long_bucket})
            if short_bucket is not None:
                log_rows.append({"name": name, "code": code, "market": market,
                                  "horizon": "短期", "close": short_close, "signal": short_bucket})

            prev = last_state.get(key)
            if prev:
                if prev.get("長期") and prev["長期"] != long_bucket:
                    long_changes.append({"name": name, "code": code, "source": source,
                                          "prev": prev["長期"], "new": long_bucket})
                if short_bucket is not None and prev.get("短期") and prev["短期"] != short_bucket:
                    short_changes.append({"name": name, "code": code, "source": source,
                                           "prev": prev["短期"], "new": short_bucket})

    save_state(new_state)

    today = str(date.today())
    append_signal_log(today, log_rows)

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
