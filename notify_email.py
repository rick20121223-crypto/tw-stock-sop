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
from email.mime.text import MIMEText
from pathlib import Path

from multi_timeframe_check import full_check
from sop_decision import classify_final
from stock_core import STOCK_NAME_MAP

STATE_FILE = Path(__file__).parent / "data" / "last_signals.json"


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


def send_email(subject: str, body: str, gmail_address: str, app_password: str) -> None:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = gmail_address
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, app_password)
        server.sendmail(gmail_address, [gmail_address], msg.as_string())


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
                arrow_icon = "🔺" if bucket in ("買進", "加碼") else "🔻" if bucket == "賣出減碼" else "⚪"
                changes.append(f"{arrow_icon} {name}（{code}）：{prev.get('分類')} → {bucket}")

    save_state(new_state)

    today = str(date.today())

    if is_first_run:
        lines = [f"台股 SOP 通知已啟用（{today}）", "", "今天的起始分類：", ""]
        for v in new_state.values():
            lines.append(f"・{v['名稱']}（{v['代碼']}）：{v['分類']}")
        lines.append("")
        lines.append("之後只有分類「變化」時才會再寄信通知你。")
        body = "\n".join(lines)
        print(body)
        send_email(f"[台股SOP] 通知已啟用（{today}）", body, gmail_address, gmail_app_password)
        return

    if not changes:
        print(f"{today}：無訊號變化，不寄信。")
        if errors:
            print("以下股票取得資料失敗：\n" + "\n".join(errors))
        return

    lines = [f"台股 SOP 訊號異動通知（{today}）", ""] + changes
    if errors:
        lines.append("")
        lines.append("⚠️ 以下股票資料取得失敗：")
        lines.extend(errors)
    body = "\n".join(lines)
    print(body)
    send_email(f"[台股SOP] 訊號異動通知（{today}）", body, gmail_address, gmail_app_password)


if __name__ == "__main__":
    main()
