"""
每天排程執行一次：用「週+日」整合SOP判讀清單內所有股票，
只在「結論跟上次不一樣」時，用 LINE Messaging API 推播通知。

被 .github/workflows/notify.yml 排程呼叫，不是給 Streamlit 用的。

需要的環境變數（GitHub Actions Secrets）：
    FINMIND_TOKEN            FinMind API Token（沒有也能跑，用免費額度）
    LINE_CHANNEL_ACCESS_TOKEN LINE Messaging API 的 Channel access token

狀態檔：data/last_signals.json，記錄上一次每檔股票的分類，
本次執行後會覆寫最新結果並由 workflow 自動 commit 回 repo。
"""

import json
import os
from datetime import date
from pathlib import Path

import requests

from multi_timeframe_check import full_check
from sop_decision import classify_final
from stock_core import STOCK_NAME_MAP

STATE_FILE = Path(__file__).parent / "data" / "last_signals.json"
LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"


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


def send_line_broadcast(text: str, token: str) -> None:
    resp = requests.post(
        LINE_BROADCAST_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"messages": [{"type": "text", "text": text}]},
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LINE 推播失敗（HTTP {resp.status_code}）：{resp.text}")


def main() -> None:
    api_token = os.environ.get("FINMIND_TOKEN", "")
    line_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not line_token:
        raise SystemExit("缺少環境變數 LINE_CHANNEL_ACCESS_TOKEN，無法推播。")

    last_state = load_last_state()
    is_first_run = not last_state

    new_state = {}
    changes = []
    errors = []

    for name, code, market in unique_watchlist():
        key = f"{code}_{market}"
        try:
            result = full_check(code, market, api_token, "", "2024-01-01")
            bucket = classify_final(result["最終建議"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}（{code}）：{exc}")
            continue

        new_state[key] = {"名稱": name, "代碼": code, "分類": bucket}

        prev = last_state.get(key)
        if prev and prev.get("分類") != bucket:
            changes.append(f"{'🔺' if bucket in ('買進', '加碼') else '🔻' if bucket == '賣出減碼' else '⚪'} "
                            f"{name}（{code}）：{prev.get('分類')} → {bucket}")

    save_state(new_state)

    today = str(date.today())

    if is_first_run:
        lines = [f"📊 台股 SOP 通知已啟用（{today}）", "今天的起始分類：", ""]
        for v in new_state.values():
            lines.append(f"・{v['名稱']}（{v['代碼']}）：{v['分類']}")
        lines.append("")
        lines.append("之後只有分類「變化」時才會再通知你。")
        message = "\n".join(lines)
        print(message)
        send_line_broadcast(message, line_token)
        return

    if not changes:
        print(f"{today}：無訊號變化，不發送通知。")
        if errors:
            print("以下股票取得資料失敗：\n" + "\n".join(errors))
        return

    lines = [f"📊 台股 SOP 訊號異動通知（{today}）", ""] + changes
    if errors:
        lines.append("")
        lines.append("⚠️ 以下股票資料取得失敗：")
        lines.extend(errors)
    message = "\n".join(lines)
    print(message)
    send_line_broadcast(message, line_token)


if __name__ == "__main__":
    main()
