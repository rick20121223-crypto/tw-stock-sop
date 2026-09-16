"""
盤中即時複查：台股開盤 09:00~13:30，每15分鐘跑一次，只重新檢查「短期」
目前已經是買進/加碼的股票（手上持股/正在盯的，即時性需求最高），60分
+5分訊號一旦變化就馬上寄信，不用等隔天早上的每日批次通知。

跟 notify_email.py 的每日批次共用同一份狀態檔 data/last_signals.json、
同一份紀錄 data/signal_log.csv，也共用寄信/格式化邏輯，差別只在「檢查
範圍」跟「觸發頻率」：
- 範圍：只檢查 last_signals.json 裡「短期」欄位目前是 買進/加碼 的股票，
  不是每天批次那樣的全部45檔自選股——這條件本身就等於只留台股個股/ETF
  （非TW市場的「短期」欄位一定是 None，永遠不會符合），大幅降低盤中
  高頻執行對 Fugle 的用量（一天跑18次 x 全部45檔會撞爆429；只挑目前
  買進/加碼的，通常只有幾檔到十幾檔）。
- 頻率：盤中09:00~13:30每15分鐘一次（見 .github/workflows/
  intraday_check.yml），本來就設計成一天要跑很多次，不需要像每日批次
  那樣做「今天跑過沒」的防重複判斷。
- 只找「訊號有變化」的，沒變化就完全不寫檔、不寄信，安靜結束。

被 .github/workflows/intraday_check.yml 排程呼叫，不是給 Streamlit 用的。
需要環境變數 FUGLE_API_KEY / GMAIL_ADDRESS / GMAIL_APP_PASSWORD（跟
notify_email.py共用同一組 GitHub Actions Secrets）。
"""

from datetime import date
import os

from notify_email import (
    BUCKET_ORDER,
    HORIZON_LABEL,
    _changes_to_html,
    _changes_to_plain,
    _check_short,
    _html_wrap,
    append_signal_changes,
    append_signal_log,
    load_last_state,
    save_state,
    send_email,
)


def build_intraday_email(today: str, changes: list) -> tuple:
    plain_lines = [f"台股 SOP 盤中訊號異動（{today}）", "", HORIZON_LABEL["短期"] + "異動：", ""]
    plain_lines += _changes_to_plain(changes)
    plain = "\n".join(plain_lines)

    html_body = (
        f'<h3 style="margin:0 0 8px; color:#444; font-size:15px;">{HORIZON_LABEL["短期"]}異動</h3>'
        + _changes_to_html(changes)
    )
    html = _html_wrap("台股 SOP 盤中訊號異動", today, html_body)
    return plain, html


def main() -> None:
    fugle_api_key = os.environ.get("FUGLE_API_KEY", "")
    gmail_address = os.environ.get("GMAIL_ADDRESS", "")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not fugle_api_key:
        print("⚠️ 未設定 FUGLE_API_KEY，盤中即時檢查沒有意義，直接跳過。")
        return
    if not gmail_address or not gmail_app_password:
        raise SystemExit("缺少環境變數 GMAIL_ADDRESS 或 GMAIL_APP_PASSWORD，無法寄信。")

    state = load_last_state()
    if not state:
        print("data/last_signals.json 還沒有資料（每日批次至少要成功跑過一次），本次跳過。")
        return

    # 「短期」是買進/加碼才進盤中複查名單：非TW市場的「短期」一定是
    # None，自然被排除，不用另外判斷市場別。
    watch_targets = [(key, v) for key, v in state.items() if v.get("短期") in ("買進", "加碼")]
    if not watch_targets:
        print("目前沒有「短期」為買進/加碼的股票需要盤中盯，跳過。")
        return

    today = str(date.today())
    changes, log_rows = [], []
    for key, v in watch_targets:
        code = v["代碼"]
        new_bucket, close, err = _check_short(code, "TW", fugle_api_key)
        if err is not None or new_bucket is None:
            # 盤中高頻執行，抓取失敗（含429 rate limit）本來就比每日批次
            # 常見，反正下一輪15分鐘後會再試，安靜略過就好，不用像每日
            # 批次那樣收斂錯誤訊息塞進信裡。
            continue
        if new_bucket != v["短期"]:
            changes.append({"name": v["名稱"], "code": code, "market": "TW", "source": v.get("來源", "固定"),
                             "prev": v["短期"], "new": new_bucket, "close": close})
            log_rows.append({"name": v["名稱"], "code": code, "market": "TW",
                              "horizon": "短期", "close": close, "signal": new_bucket})
            state[key]["短期"] = new_bucket

    if not changes:
        print(f"{today}：盤中複查 {len(watch_targets)} 檔，訊號沒有變化，不寄信。")
        return

    save_state(state)
    append_signal_log(today, log_rows)
    append_signal_changes(changes, "短期")

    changes.sort(key=lambda c: BUCKET_ORDER.get(c["new"], 9))
    plain, html = build_intraday_email(today, changes)
    print(plain)
    send_email(f"[台股SOP] 盤中訊號異動（{today}）", plain, html, gmail_address, gmail_app_password)


if __name__ == "__main__":
    main()
