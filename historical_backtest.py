"""
歷史回放回測：對「現在的核心持股」，往回倒推BACKTEST_MONTHS個月，在每
一個歷史交易日「假裝那天是today」重新跑一次長期(週+日)SOP判讀，重建出
一份「如果notify_email.py從幾個月前就開始跑」的合成訊號歷史，再接上跟
backtest_signal_log.py同一套「訊號→未來N個交易日報酬」量測邏輯。

跟backtest_signal_log.py的差異：backtest_signal_log.py只能回測
data/signal_log.csv「已經真的跑過、記錄下來」的訊號，歷史長度受限於
系統上線時間（目前才3週，樣本太小）；這支腳本改用「技術指標本身只看
過去、不會用到未來資料」的特性，直接用現有長歷史價格資料往回重建，
不受上線時間限制，可以一次看到更長的歷史表現。

只對「現在的核心持股」回測（stock_core.STOCK_NAME_MAP）：法人排行／
雷老闆YT／Jason提及這幾個動態輪替名單是最近才存在的概念，沒辦法往回
推；回測清單用現在的核心持股最貼近使用者實際關心的股票。

只回測「長期(週+日)」：短期(60分/5分)需要分K歷史資料，免費Fugle方案
通常拿不到夠久的歷史，技術上拿不到資料，不在這支腳本範圍內。

效能設計：每檔股票的日線指標只計算「一次」（在完整歷史上），之後每個
歷史日期只是把算好的序列切到那個日期為止，再丟進evaluate_timeframe()，
不是每個歷史日期都重新抓資料、重新算一次日線指標——均線/MACD/OBV這些
指標都是「只看過去」的，在完整歷史上算一次，截到某個日期的最後一筆，
等同於「真的只用那天為止的資料去算」，兩者數學上完全一致，但效能差了
上百倍（不用對每個歷史日各打一次API/各算一次指標）。週線則是每個as-of
日期都重新resample＋算指標，因為「這一週收完了沒」這件事本身就跟as-of
日期有關（見multi_timeframe_check.py同款的不完整週K保護邏輯）。

唯一例外：W底/M頭/三角收斂型態偵測用了rolling(center=True)，技術上涉及
「往後看」，歷史回放時有一點點資料洩漏，但sop_decision.py裡明文規定這
三個偵測函式「永遠不直接產生買賣結論」，只影響輔助文字說明，不影響這支
腳本實際量測的「加碼/買進/觀望/賣出減碼」分類結果，所以不影響回測的
量化結論。

用法（在專案資料夾內執行）：
    python3 historical_backtest.py [FinMind_Token] [回測月數=6]

有設定 GMAIL_ADDRESS/GMAIL_APP_PASSWORD 環境變數時，結果會寄一封摘要信
（沿用notify_email.py同一套寄信邏輯/帳密，不需要另外的Secrets）；沒設
就只印出結果，方便本機手動執行時直接看。被 .github/workflows/
monthly_backtest.yml 每月排程呼叫。
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns, summarize
from sop_decision import classify_final, combine_timeframes, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

OUTPUT_FILE = Path(__file__).parent / "data" / "historical_backtest.csv"

BACKTEST_MONTHS_DEFAULT = 6
# 指標(尤其MA240/年線，需240個交易日)要有足夠長的暖身歷史才會有效，
# 往回多抓一段緩衝，確保回測窗口「一開始」那幾天也有夠長的指標歷史可用，
# 不會整段都是「資料不足」。420天(約14個月)涵蓋MA240(約11個月)還有餘裕。
WARMUP_CALENDAR_DAYS = 420


def _build_weekly(df_day: pd.DataFrame) -> pd.DataFrame:
    """跟multi_timeframe_check.full_check()同一套週線resample+不完整週K
    保護邏輯，這裡重新實作一份是因為full_check()的介面綁死了「抓到
    today」，沒辦法餵「已經截好的歷史日線」進去。"""
    week_src = df_day.copy()
    week_src["date"] = pd.to_datetime(week_src["date"])
    df_week = (
        week_src.set_index("date")
        .resample("W")
        .agg({"open": "first", "max": "max", "min": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    if not df_week.empty and week_src["date"].iloc[-1].weekday() < 4:
        df_week = df_week.iloc[:-1]
    return df_week


def _evaluate_as_of(df_day_full: pd.DataFrame, as_of_date: str) -> str:
    """把已經算好日線指標的完整df_day_full截到as_of_date為止，重建對應的
    週線，跑長期(週+日)判讀，回傳分類後的結論桶（加碼/買進/觀望/賣出
    減碼）。資料不足回傳None，呼叫端據此略過這個日期。"""
    day_slice = df_day_full[df_day_full["date"] <= as_of_date]
    if day_slice.empty:
        return None
    week_slice = _build_weekly(day_slice)
    if week_slice.empty:
        return None
    week_slice = run_all_indicators(week_slice, "週")

    verdict_day = evaluate_timeframe(day_slice, "日")
    verdict_week = evaluate_timeframe(week_slice, "週")
    combined = combine_timeframes({"日": verdict_day, "週": verdict_week})
    return classify_final(combined["最終建議"])


def _backtest_one(name: str, code: str, market: str, api_token: str,
                   backtest_months: int) -> tuple:
    """回傳 (code, DataFrame或None, 錯誤訊息或None)。"""
    try:
        today = date.today()
        backtest_start = today - timedelta(days=backtest_months * 30)
        fetch_start = (backtest_start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

        df_day = get_stock_data(code, market, fetch_start, str(today), api_token)
        if df_day.empty:
            return code, None, "日線資料為空"
        df_day = run_all_indicators(df_day, "日")
        df_day = df_day.sort_values("date").reset_index(drop=True)
        df_day["date"] = df_day["date"].astype(str)

        backtest_start_str = backtest_start.strftime("%Y-%m-%d")
        as_of_dates = [d for d in df_day["date"] if d >= backtest_start_str]

        rows = []
        for as_of in as_of_dates:
            signal = _evaluate_as_of(df_day, as_of)
            if signal is None:
                continue
            close = float(df_day.loc[df_day["date"] == as_of, "close"].iloc[-1])
            row = {"date": as_of, "name": name, "code": code, "market": market,
                   "horizon": "長期", "close": close, "signal": signal}
            fwd = _forward_returns(df_day, as_of)
            for n in FORWARD_WINDOWS:
                row[f"fwd_ret_{n}d"] = fwd[n]
            rows.append(row)
        return code, pd.DataFrame(rows), None
    except Exception as exc:  # noqa: BLE001
        return code, None, str(exc)


def run_historical_backtest(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> pd.DataFrame:
    watchlist = unique_watchlist(STOCK_NAME_MAP)  # 只用現在的核心持股，見模組docstring

    all_frames, errors = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_backtest_one, name, code, market, api_token, backtest_months)
            for name, code, market in watchlist
        ]
        for future in concurrent.futures.as_completed(futures):
            code, df, err = future.result()
            if err is not None:
                errors[code] = err
            elif df is not None and not df.empty:
                all_frames.append(df)

    if errors:
        for code, err in errors.items():
            print(f"⚠️ {code}：{err}")

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _summary_to_plain(summary_df: pd.DataFrame) -> list:
    lines = []
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['結論']}：{row['樣本數(去重股票)']}檔/{row['訊號筆數']}筆　"
            f"5日 {row['5日_平均報酬%']}%(勝率{row['5日_正報酬勝率%']}%)　"
            f"10日 {row['10日_平均報酬%']}%(勝率{row['10日_正報酬勝率%']}%)　"
            f"20日 {row['20日_平均報酬%']}%(勝率{row['20日_正報酬勝率%']}%)"
        )
    return lines


def _summary_to_html(summary_df: pd.DataFrame) -> str:
    from notify_email import BUCKET_STYLE

    def _cell(row, n):
        ret, wr = row[f"{n}日_平均報酬%"], row[f"{n}日_正報酬勝率%"]
        if pd.isna(ret):
            return "—"
        return f'{ret:+.2f}%<br><span style="color:#999; font-size:11px;">勝率{wr}%</span>'

    header = (
        '<tr style="background:#f5f5f5; font-size:12px; color:#666;">'
        '<th style="text-align:left; padding:6px 8px;">結論</th>'
        '<th style="padding:6px 8px;">樣本</th>'
        '<th style="padding:6px 8px;">5日</th>'
        '<th style="padding:6px 8px;">10日</th>'
        '<th style="padding:6px 8px;">20日</th>'
        '</tr>'
    )
    rows_html = ""
    for _, row in summary_df.iterrows():
        style = BUCKET_STYLE.get(row["結論"], {"color": "#333"})
        rows_html += (
            '<tr style="border-top:1px solid #eee;">'
            f'<td style="padding:8px; font-weight:700; color:{style["color"]};">{row["結論"]}</td>'
            f'<td style="padding:8px; text-align:center; color:#999; font-size:12px;">'
            f'{row["樣本數(去重股票)"]}檔/{row["訊號筆數"]}筆</td>'
            f'<td style="padding:8px; text-align:center;">{_cell(row, 5)}</td>'
            f'<td style="padding:8px; text-align:center;">{_cell(row, 10)}</td>'
            f'<td style="padding:8px; text-align:center;">{_cell(row, 20)}</td>'
            '</tr>'
        )
    return f'<table style="width:100%; border-collapse:collapse; font-size:13px;">{header}{rows_html}</table>'


def build_backtest_email(today: str, months: int, summary_df: pd.DataFrame) -> tuple:
    from notify_email import _html_wrap

    caveat = ("提醒：這是對「現在的核心持股」往回回放，有存活者偏誤——這批股票是"
              "現在已經持有、多半已經賺錢的核心持股，不是當時隨機抽樣的股票池，"
              "所以不能當成「SOP整體選股準不準」的結論，只能看這批股票的訊號"
              "有沒有鑑別力（加碼有沒有贏賣出減碼）。")

    plain_lines = [f"核心持股長期SOP歷史回放回測（{today}，回溯{months}個月）", ""]
    plain_lines += _summary_to_plain(summary_df)
    plain_lines += ["", caveat]
    plain = "\n".join(plain_lines)

    html_body = _summary_to_html(summary_df)
    html_body += f'<p style="margin-top:16px; color:#999; font-size:12px;">{caveat}</p>'
    html = _html_wrap(f"核心持股長期SOP歷史回測（回溯{months}個月）", today, html_body)
    return plain, html


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    detail_df = run_historical_backtest(api_token, months_arg)
    if detail_df.empty:
        print("沒有算出任何歷史訊號，無法回測（可能是FinMind抓取全部失敗）。")
        sys.exit(0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"逐筆歷史回放明細已存到 {OUTPUT_FILE}（共 {len(detail_df)} 筆，回測窗口 {months_arg} 個月）")

    summary_df = summarize(detail_df)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print("\n=== 依結論分組的未來N個交易日報酬統計（歷史回放，核心持股） ===")
    print("（買進/加碼想看到正報酬才算對；賣出減碼想看到負報酬才算對，方向要自己對照看）")
    print(summary_df.to_string(index=False))

    gmail_address = os.environ.get("GMAIL_ADDRESS", "")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if gmail_address and gmail_app_password:
        from notify_email import send_email
        today_str = date.today().isoformat()
        plain, html = build_backtest_email(today_str, months_arg, summary_df)
        send_email(f"[台股SOP] 核心持股歷史回測（回溯{months_arg}個月）", plain, html,
                   gmail_address, gmail_app_password)
        print("\n已寄出回測摘要信。")
    else:
        print("\n未設定GMAIL_ADDRESS/GMAIL_APP_PASSWORD，只印出結果，不寄信。")
