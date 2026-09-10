"""
⚡ 短線進場
回答的問題是「現在適不適合短打進場」，只看 60分＋5分 兩個短週期的訊號，
不看日/週長線結構（那是首頁「長線留倉」的事）。依 SOP：兩個週期都同步
偏多才算真正的買進訊號，任一週期轉弱就判賣出減碼，避免對到雜訊追短。

只支援台股個股/ETF（美股/期貨用的 Fugle 不支援），且需要 Fugle 行情
API Key 才能查60分/5分資料。
"""

import concurrent.futures
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from sop_decision import classify_final, combine_timeframes, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_intraday_data, run_all_indicators

st.set_page_config(page_title="短線進場", layout="wide", page_icon="⚡")

# 台股慣例：紅漲綠跌（跟美股相反）
UP_COLOR = "#e53935"
DOWN_COLOR = "#43a047"
FLAT_COLOR = "#9e9e9e"

BUCKET_ORDER = {"加碼": 0, "買進": 1, "觀望": 2, "賣出減碼": 3}
BUCKET_STYLE = {
    "加碼":     {"color": UP_COLOR,   "bg": "#fdecea"},
    "買進":     {"color": UP_COLOR,   "bg": "#fdecea"},
    "觀望":     {"color": FLAT_COLOR, "bg": "#f5f5f5"},
    "賣出減碼": {"color": DOWN_COLOR, "bg": "#eaf6ec"},
}


def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


with st.sidebar:
    with st.expander("🔑 API 金鑰狀態"):
        secret_fugle = _get_secret("FUGLE_API_KEY")
        if secret_fugle:
            st.success("已使用雲端 Secrets 的 Fugle API Key")
            fugle_api_key = secret_fugle
        else:
            fugle_api_key = st.text_input("Fugle 行情 API Key（60分/5分線用）", type="password")

    refresh = st.button("🔄 重新整理資料", use_container_width=True)

st.title("⚡ 短線進場")
st.caption(
    "只看60分＋5分的短線訊號，找出「現在適合短打進場」的股票，不看日/週"
    "長線結構。依 SOP：兩個週期都同步偏多才判買進，任一週期轉弱就判賣出"
    "減碼，觀望代表訊號還不夠一致。只支援台股個股/ETF。僅供輔助判讀，"
    "不構成投資建議。"
)

if not fugle_api_key:
    st.warning("需要 Fugle 行情 API Key 才能查60分/5分資料。請在左側「🔑 API 金鑰狀態」輸入，"
               "或部署時於 Secrets 設定 FUGLE_API_KEY。")
    st.stop()

if refresh:
    st.cache_data.clear()


def unique_tw_watchlist() -> list:
    """短線進場只支援台股個股/ETF（Fugle不支援美股/期貨）"""
    seen, result = set(), []
    for name, (code, market) in STOCK_NAME_MAP.items():
        if market != "TW":
            continue
        key = (code, market)
        if key in seen:
            continue
        seen.add(key)
        result.append((name, code))
    return result


@st.cache_data(ttl=900, show_spinner=False)
def analyze_short(name: str, code: str, fugle_key: str):
    try:
        verdicts = {}
        latest_close = None

        df60 = get_intraday_data(code, "60", fugle_key)
        if not df60.empty:
            df60 = run_all_indicators(df60, "60分")
            verdicts["60分"] = evaluate_timeframe(df60, "60分")
            latest_close = float(df60.iloc[-1]["close"])

        df5 = get_intraday_data(code, "5", fugle_key)
        if not df5.empty:
            df5 = run_all_indicators(df5, "5分")
            verdicts["5分"] = evaluate_timeframe(df5, "5分")
            latest_close = float(df5.iloc[-1]["close"])

        if not verdicts:
            return {"名稱": name, "代碼": code, "狀態": "error", "訊息": "60分/5分皆無資料"}

        result = combine_timeframes(verdicts)
        bucket = classify_final(result["最終建議"])
        return {
            "名稱": name, "代碼": code, "狀態": "ok",
            "收盤": latest_close,
            "最終建議": result["最終建議"],
            "分類": bucket,
            "60分": verdicts["60分"].conclusion if "60分" in verdicts else "—",
            "5分": verdicts["5分"].conclusion if "5分" in verdicts else "—",
        }
    except Exception as exc:  # noqa: BLE001
        return {"名稱": name, "代碼": code, "狀態": "error", "訊息": str(exc)}


with st.spinner("正在平行分析清單內台股股票的60分/5分短線訊號（首次載入較久，之後15分鐘內會用快取）..."):
    rows, errors = [], []
    # Fugle 的頻率限制比 FinMind 嚴，這裡平行度故意調低（+程式碼裡的
    # 429 退避重試），避免一次查整份清單就撞到 Rate limit。
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(analyze_short, name, code, fugle_api_key)
            for name, code in unique_tw_watchlist()
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            (rows if result["狀態"] == "ok" else errors).append(result)

if not rows:
    st.error("所有股票都取得失敗，請確認 Fugle API Key 是否正確。以下是實際錯誤原因：")
    for e in errors:
        st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")
    st.stop()

df_overview = pd.DataFrame(rows)
df_overview["排序"] = df_overview["分類"].map(BUCKET_ORDER)
df_overview = df_overview.sort_values("排序").drop(columns="排序")

buy_count = df_overview["分類"].isin(["買進", "加碼"]).sum()
sell_count = (df_overview["分類"] == "賣出減碼").sum()
watch_count = (df_overview["分類"] == "觀望").sum()

c1, c2, c3 = st.columns(3)
c1.metric("🔺 短線可進場", f"{buy_count} 檔")
c2.metric("🔻 短線賣出減碼", f"{sell_count} 檔")
c3.metric("⚪ 觀望", f"{watch_count} 檔")

st.divider()


def _stock_card(row) -> str:
    style = BUCKET_STYLE[row["分類"]]
    price = f"{row['收盤']:.2f}" if pd.notna(row["收盤"]) else "—"
    return f"""\
<div style="border-left:4px solid {style['color']}; background:{style['bg']};
            border-radius:6px; padding:10px 12px; height:100%;">
  <div style="font-weight:600; font-size:13px; color:#222;">{row['名稱']}（{row['代碼']}）</div>
  <div style="font-size:17px; font-weight:700; color:{style['color']}; margin:3px 0;">
    {row['最終建議']}
  </div>
  <div style="font-size:12px; color:#666;">收盤 {price} ・ 60分：{row['60分']} ・ 5分：{row['5分']}</div>
</div>
"""


cards_html = "".join(_stock_card(row) for _, row in df_overview.iterrows())
st.markdown(
    f'<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); '
    f'gap:10px;">{cards_html}</div>',
    unsafe_allow_html=True,
)

if errors:
    with st.expander(f"⚠️ {len(errors)} 檔資料取得失敗"):
        for e in errors:
            st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")

st.caption("👉 SLS（美股）不支援60分/5分，未列入此頁。想看週+日的長線留倉判斷，請切到左側選單「長線留倉」首頁。")
