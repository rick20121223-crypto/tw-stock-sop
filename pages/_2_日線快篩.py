"""
日線快篩 — 只看「日線」單一週期的簡化版總覽，速度最快（不算週線、
不需要 Fugle Key）。想看週+日整合判讀，請切到首頁「多週期整合總覽」。

檔名開頭底線是刻意的：依使用者指示把導覽列精簡成只留「短線進場／
長線留倉／多週期整合分析」3頁，這頁改成不顯示在側邊選單（但程式碼、
功能都還在，之後想恢復顯示，把檔名開頭的底線拿掉、恢復數字排序即可）。
"""

import concurrent.futures
import os
import sys
from datetime import date, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

import institutional_ranking as ir
from sop_decision import evaluate_timeframe
from stock_core import get_stock_data, run_all_indicators, unique_watchlist

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


def _stock_card(row, badge: str = "") -> str:
    style = BUCKET_STYLE[row["結論"]]
    price = f"{row['收盤']:.2f}" if pd.notna(row["收盤"]) else "—"
    reason = row["理由"] if row["理由"] else ""
    badge_html = f'<span style="font-size:11px; color:#999;">{badge}</span> ' if badge else ""
    return f"""\
<div style="border-left:4px solid {style['color']}; background:{style['bg']};
            border-radius:6px; padding:10px 12px; height:100%;">
  <div style="font-weight:600; font-size:13px; color:#222;">{badge_html}{row['名稱']}（{row['代碼']}）</div>
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

st.caption("👉 想看某檔股票的四關價／K線／MACD／OBV 詳細圖表，可直接開啟"
           "「個股詳細分析」頁面（已從導覽列隱藏，程式碼還在，網址是 "
           "pages/_4_個股詳細分析），或到「多週期整合分析」頁面看文字結論。")

# ------------------------------------------------------------------
# 🔄 本週法人排行：獨立區塊，跟上面的固定清單分開顯示。每週一由排程自動
# 算出「上週三大法人買賣超前十大（買超+賣超）」覆寫這份名單。
# ------------------------------------------------------------------
st.divider()
weekly_picks = ir.load_weekly_picks()

if not weekly_picks:
    st.caption("🔄 本週法人排行：尚未產生（要等排程第一次執行「每週一算排行」之後才會有資料）。")
else:
    st.subheader("🔄 本週法人排行（三大法人買賣超前十大，跟固定清單分開）")
    st.caption(
        f"資料範圍：{weekly_picks['week_start']} ~ {weekly_picks['week_end']}"
        "（上市+上櫃三大法人合計淨額，已排除ETF與固定清單，下週一會自動換成新一週的排行）。"
    )
    if weekly_picks.get("tpex_days_available", 0) == 0:
        st.caption("⚠️ 上櫃(TPEx)法人資料還在累積中（需要每天存檔滿一週），這次排行僅計算上市(TWSE)。")

    rotating_list = ir.rotating_watchlist()  # [(name, code, market, side)]
    side_map = {code: side for _, code, _market, side in rotating_list}

    rotating_rows, rotating_errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(analyze_one, name, code, market, api_token, lookback_days)
            for name, code, market, _side in rotating_list
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            (rotating_rows if result["狀態"] == "ok" else rotating_errors).append(result)

    if rotating_rows:
        for row in rotating_rows:
            row["訊號"] = CONCLUSION_EMOJI[row["結論"]] + " " + row["結論"]
        rotating_cards_html = "".join(
            _stock_card(row, badge=f"🔄 法人{side_map.get(row['代碼'], '')}")
            for row in rotating_rows
        )
        st.markdown(
            f'<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); '
            f'gap:10px;">{rotating_cards_html}</div>',
            unsafe_allow_html=True,
        )

    if rotating_errors:
        with st.expander(f"⚠️ {len(rotating_errors)} 檔本週法人排行股票資料取得失敗"):
            for e in rotating_errors:
                st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")
