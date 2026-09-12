"""
日線快篩 — 只看「日線」單一週期的簡化版總覽，速度最快（不算週線、
不需要 Fugle Key）。想看週+日整合判讀，請切到首頁「多週期整合總覽」。
"""

import concurrent.futures
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
    with st.expander("🔑 API 金鑰狀態"):
        secret_token = _get_secret("FINMIND_TOKEN")
        if secret_token:
            st.success("已使用雲端 Secrets 的 FinMind Token")
            api_token = secret_token
        else:
            api_token = st.text_input("FinMind API Token", type="password")

    lookback_days = st.slider(
        "回溯天數（需 ≥60 天才能算出 MA35；≥170 天才能算出半年線 MA120）",
        60, 365, 180, step=10,
    )
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


with st.spinner("正在平行分析清單內所有股票（首次載入較久，之後 15 分鐘內會用快取）..."):
    rows = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(analyze_one, name, code, market, api_token, lookback_days)
            for name, code, market in unique_watchlist()
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
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


BUCKET_STYLE = {
    "加碼":     {"color": UP_COLOR,   "bg": "#fdecea"},
    "買進":     {"color": UP_COLOR,   "bg": "#fdecea"},
    "觀望":     {"color": FLAT_COLOR, "bg": "#f5f5f5"},
    "賣出減碼": {"color": DOWN_COLOR, "bg": "#eaf6ec"},
}


def _stock_card(row) -> str:
    style = BUCKET_STYLE[row["結論"]]
    price = f"{row['收盤']:.2f}" if pd.notna(row["收盤"]) else "—"
    reason = row["理由"] if row["理由"] else ""
    return f"""\
<div style="border-left:4px solid {style['color']}; background:{style['bg']};
            border-radius:6px; padding:10px 12px; height:100%;">
  <div style="font-weight:600; font-size:13px; color:#222;">{row['名稱']}（{row['代碼']}）</div>
  <div style="font-size:19px; font-weight:700; color:{style['color']}; margin:3px 0;">
    {row['訊號']}
  </div>
  <div style="font-size:12px; color:#666;">收盤 {price} ・ 信心 {row['信心']} ・ 分數 {row['分數']:+.1f}</div>
  <div style="font-size:11px; color:#888; margin-top:4px;">{reason}</div>
</div>
"""


cards_html = "".join(_stock_card(row) for _, row in df_overview.iterrows())
st.markdown(
    f'<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); '
    f'gap:10px;">{cards_html}</div>',
    unsafe_allow_html=True,
)

if errors:
    with st.expander(f"⚠️ {len(errors)} 檔資料取得失敗"):
        for e in errors:
            st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")

st.caption("👉 想看某檔股票的四關價／K線／MACD／OBV 詳細圖表，請用左側選單切換到「個股詳細分析」頁面。")
