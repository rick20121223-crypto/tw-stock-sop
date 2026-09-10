"""
每天排程執行一次：用「週+日」整合SOP判讀清單內所有股票，
只在「結論跟上次不一樣」時，寄一封 Email 通知自己。

被 .github/workflows/notify.yml 排程呼叫，不是給 Streamlit 用的。

需要的環境變數（GitHub Actions Secrets）：
    FINMIND_TOKEN       FinMind API Token（沒有也能跑，用免費額度）
    GMAIL_ADDRESS        寄件/收件用的 Gmail 帳號，例如 rick20121223@gmail.com
    GMAIL_APP_PASSWORD   Gmail「應用程式密碼」（16碼），不是你的登入密碼

狀態檔：data/last_signals.json，記錄上一次每檔股票的分類，
本次執行後會覆寫最新結果並由 workflow 自動 commit 回 repo。
"""

import concurrent.futures
import json
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from multi_timeframe_check import full_check
from sop_decision import classify_final
from stock_core import STOCK_NAME_MAP

STATE_FILE = Path(__file__).parent / "data" / "last_signals.json"

# 跟網站首頁同一套配色：台股慣例紅漲綠跌（買進/加碼=紅、賣出減碼=綠）
BUCKET_ORDER = {"加碼": 0, "買進": 1, "觀望": 2, "賣出減碼": 3}
BUCKET_STYLE = {
    "加碼":     {"color": "#e53935", "bg": "#fdecea", "icon": "🔺🔺"},
    "買進":     {"color": "#e53935", "bg": "#fdecea", "icon": "🔺"},
    "觀望":     {"color": "#757575", "bg": "#f5f5f5", "icon": "⚪"},
    "賣出減碼": {"color": "#43a047", "bg": "#eaf6ec", "icon": "🔻"},
}


def unique_watchlist():
    seen, result = set(), []
    for name, (code, market) in STOCK_NAME_MAP.items():
        key = (code, market)
        if key in seen:
            continue
        seen.add(key)
        result.append((name, code, market))
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


def send_email(subject: str, plain_body: str, html_body: str,
                gmail_address: str, app_password: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = gmail_address
    # 先附純文字版（沒開HTML的信箱/通知預覽用），HTML版放後面，
    # email用戶端會優先顯示multipart/alternative裡最後一個它看得懂的版本。
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
    僅供輔助判讀，不構成投資建議。依但丁老師 SOP（四關價→均線→MACD→OBV）
    的「週+日」整合結論自動產生。
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


def build_first_run_email(today: str, new_state: dict) -> tuple:
    rows = sorted(new_state.values(), key=lambda v: BUCKET_ORDER.get(v["分類"], 9))

    plain_lines = [f"台股 SOP 通知已啟用（{today}）", "", "今天的起始分類：", ""]
    for v in rows:
        plain_lines.append(f"・{v['名稱']}（{v['代碼']}）：{v['分類']}")
    plain_lines += ["", "之後只有分類「變化」時才會再寄信通知你。"]
    plain = "\n".join(plain_lines)

    cards = ""
    for v in rows:
        style = BUCKET_STYLE[v["分類"]]
        cards += _card_html(
            style["icon"], style["color"], style["bg"],
            f"{v['名稱']}（{v['代碼']}）",
            f'<span style="color:{style["color"]}; font-weight:600;">{v["分類"]}</span>',
        )
    html_body = cards + (
        '<p style="color:#888; font-size:13px; margin-top:16px;">'
        "之後只有分類「變化」時才會再寄信通知你。</p>"
    )
    html = _html_wrap("台股 SOP 通知已啟用", today, html_body)
    return plain, html


def build_change_email(today: str, changes: list, errors: list) -> tuple:
    plain_lines = [f"台股 SOP 訊號異動通知（{today}）", ""]
    for c in changes:
        style = BUCKET_STYLE[c["new"]]
        plain_lines.append(f"{style['icon']} {c['name']}（{c['code']}）：{c['prev']} → {c['new']}")
    if errors:
        plain_lines += ["", "⚠️ 以下股票資料取得失敗："] + errors
    plain = "\n".join(plain_lines)

    cards = ""
    for c in changes:
        style = BUCKET_STYLE[c["new"]]
        detail = (
            f'<span style="color:#999;">{c["prev"]}</span>'
            f' <span style="color:#bbb;">→</span> '
            f'<span style="color:{style["color"]}; font-weight:700; font-size:15px;">{c["new"]}</span>'
        )
        cards += _card_html(style["icon"], style["color"], style["bg"],
                             f"{c['name']}（{c['code']}）", detail)
    if errors:
        err_html = "".join(f"<div>⚠️ {e}</div>" for e in errors)
        cards += (
            '<div style="margin-top:16px; padding:10px 14px; background:#fff8e1; '
            'border-radius:4px; color:#8a6d00; font-size:12px;">' + err_html + "</div>"
        )
    html = _html_wrap("台股 SOP 訊號異動通知", today, cards)
    return plain, html


def main() -> None:
    api_token = os.environ.get("FINMIND_TOKEN", "")
    gmail_address = os.environ.get("GMAIL_ADDRESS", "")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not gmail_address or not gmail_app_password:
        raise SystemExit("缺少環境變數 GMAIL_ADDRESS 或 GMAIL_APP_PASSWORD，無法寄信。")

    last_state = load_last_state()
    is_first_run = not last_state

    new_state = {}
    changes = []
    errors = []

    def _check_one(name, code, market):
        try:
            result = full_check(code, market, api_token, "", "2024-01-01")
            return name, code, market, classify_final(result["最終建議"]), None
        except Exception as exc:  # noqa: BLE001
            return name, code, market, None, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_check_one, name, code, market)
            for name, code, market in unique_watchlist()
        ]
        for future in concurrent.futures.as_completed(futures):
            name, code, market, bucket, error = future.result()
            if error is not None:
                errors.append(f"{name}（{code}）：{error}")
                continue

            key = f"{code}_{market}"
            new_state[key] = {"名稱": name, "代碼": code, "分類": bucket}

            prev = last_state.get(key)
            if prev and prev.get("分類") != bucket:
                changes.append({
                    "name": name, "code": code,
                    "prev": prev.get("分類"), "new": bucket,
                })

    save_state(new_state)

    today = str(date.today())

    if is_first_run:
        plain, html = build_first_run_email(today, new_state)
        print(plain)
        send_email(f"[台股SOP] 通知已啟用（{today}）", plain, html, gmail_address, gmail_app_password)
        return

    if not changes:
        print(f"{today}：無訊號變化，不寄信。")
        if errors:
            print("以下股票取得資料失敗：\n" + "\n".join(errors))
        return

    # 依異動後的分類排序，買進/加碼在最上面，賣出減碼在最下面
    changes.sort(key=lambda c: BUCKET_ORDER.get(c["new"], 9))
    plain, html = build_change_email(today, changes, errors)
    print(plain)
    send_email(f"[台股SOP] 訊號異動通知（{today}）", plain, html, gmail_address, gmail_app_password)


if __name__ == "__main__":
    main()
