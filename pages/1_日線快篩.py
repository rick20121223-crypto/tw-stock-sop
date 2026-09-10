"""
日線快篩 — 只看「日線」單一週期的簡化版總覽，速度最快（不算週線、
不需要 Fugle Key）。想看週+日整合判讀，請切到首頁「多週期整合總覽」。
"""

import os
import sys
from datetime import date, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from sop_decision import evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators

st.set_page_config(page_title="日線快篩", layout="wide", page_icon="⚡")

# 台股慣例：紅漲綠跌（跟美股相反）
UP_COLOR = "#e53935"
DOWN_COLOR = "#43a047"
FLAT_COLOR = "#9e9e9e"

CONCLUSION_ORDER = {"加碼": 0, "買進": 1, "觀望": 2, "賣出減碼": 3}
CONCLUSION_EMOJI = {"加碼": "🔺🔺", "買進": "🔺", "觀望": "⚪", "賣出減碼": "🔻"}


def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


# ------------------------------------------------------------------
# 側邊欄：金鑰設定（優先讀 secrets，沒有的話讓使用者手動輸入）
# ------------------------------------------------------------------
with st.sidebar:
    st.header("設定")
    secret_token = _get_secret("FINMIND_TOKEN")
    if secret_token:
        st.success("已使用雲端 Secrets 的 FinMind Token")
        api_token = secret_token
    else:
        api_token = st.text_input("FinMind API Token", type="password")

    st.divider()
    lookback_days = st.slider("回溯天數（需 ≥60 天才能算出 MA35）", 60, 365, 180, step=10)
    refresh = st.button("🔄 重新整理資料", use_container_width=True)

st.title("⚡ 日線快篩")
st.caption(
    "依但丁老師 SOP（四關價→均線→MACD→OBV）的位階否決邏輯，快篩清單內每檔股票的"
    "**日線**結論，方便一次掃過整份清單，速度最快、不需要 Fugle Key。"
    "僅供輔助判讀，不構成投資建議。"
)
st.info(
    "這是簡化版，只看日線單一週期。想看週+日整合過的完整判讀，"
    "請切到左側選單「streamlit app」首頁。",
    icon="ℹ️",
)

if not api_token:
    st.info("未設定 FinMind API Token，將使用免費額度（較容易觸發流量限制）。建議在左側輸入 Token，或部署時於 Secrets 設定 FINMIND_TOKEN。")

if refresh:
    st.cache_data.clear()


# ------------------------------------------------------------------
# 去重股票清單（0050／元大台灣50 是同一檔，只取第一個名稱）
# ------------------------------------------------------------------
def unique_watchlist() -> list:
    seen = set()
    result = []
    for name, (code, market) in STOCK_NAME_MAP.items():
        key = (code, market)
        if key in seen:
            continue
        seen.add(key)
        result.append((name, code, market))
    return result


@st.cache_data(ttl=900, show_spinner=False)
def analyze_one(name: str, code: str, market: str, token: str, days: int):
    start_date = str(date.today() - timedelta(days=days))
    end_date = str(date.today())
    try:
        df = get_stock_data(code, market, start_date, end_date, token)
        if df.empty or len(df) < 20:
            return {"名稱": name, "代碼": code, "狀態": "error", "訊息": "資料不足或查無資料"}
        df = run_all_indicators(df, "日")
        verdict = evaluate_timeframe(df, "日")
        latest = df.iloc[-1]
        return {
            "名稱": name,
            "代碼": code,
            "狀態": "ok",
            "收盤": float(latest["close"]) if pd.notna(latest["close"]) else None,
            "結論": verdict.conclusion,
            "信心": verdict.confidence,
            "分數": verdict.score,
            "理由": "；".join(verdict.reasons[:2]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"名稱": name, "代碼": code, "狀態": "error", "訊息": str(exc)}


with st.spinner("正在依序分析清單內所有股票（首次載入較久，之後 15 分鐘內會用快取）..."):
    rows = []
    errors = []
    for name, code, market in unique_watchlist():
        result = analyze_one(name, code, market, api_token, lookback_days)
        if result["狀態"] == "ok":
            rows.append(result)
        else:
            errors.append(result)

if not rows:
    st.error("所有股票都取得失敗，請確認 FinMind Token 是否正確。以下是實際錯誤原因：")
    for e in errors:
        st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")
    st.stop()

df_overview = pd.DataFrame(rows)
df_overview["排序"] = df_overview["結論"].map(CONCLUSION_ORDER)
df_overview = df_overview.sort_values(["排序", "分數"], ascending=[True, False]).drop(columns="排序")
df_overview["訊號"] = df_overview["結論"].map(CONCLUSION_EMOJI) + " " + df_overview["結論"]

# ------------------------------------------------------------------
# 摘要：買進/加碼 vs 賣出 各幾檔
# ------------------------------------------------------------------
buy_count = df_overview["結論"].isin(["買進", "加碼"]).sum()
sell_count = (df_overview["結論"] == "賣出減碼").sum()
watch_count = (df_overview["結論"] == "觀望").sum()

c1, c2, c3 = st.columns(3)
c1.metric("🔺 買進／加碼", f"{buy_count} 檔")
c2.metric("🔻 賣出減碼", f"{sell_count} 檔")
c3.metric("⚪ 觀望", f"{watch_count} 檔")

st.divider()


def highlight_conclusion(row):
    color = FLAT_COLOR
    if "賣出減碼" in row["訊號"]:
        color = DOWN_COLOR
    elif "買進" in row["訊號"] or "加碼" in row["訊號"]:
        color = UP_COLOR
    return [f"color: {color}; font-weight: 600" if col == "訊號" else "" for col in row.index]


display_cols = ["名稱", "代碼", "收盤", "訊號", "信心", "分數", "理由"]
styled = (
    df_overview[display_cols]
    .style.apply(highlight_conclusion, axis=1)
    .format({"收盤": "{:.2f}", "分數": "{:.1f}"})
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=min(80 + 35 * len(df_overview), 900))

if errors:
    with st.expander(f"⚠️ {len(errors)} 檔資料取得失敗"):
        for e in errors:
            st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")

st.caption("👉 想看某檔股票的四關價／K線／MACD／OBV 詳細圖表，請用左側選單切換到「個股詳細分析」頁面。")
