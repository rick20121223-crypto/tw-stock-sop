"""
多週期整合分析 — 日／週／60分／5分一次判讀，套用「為日線留倉，為五分出場」
的跨週期整合邏輯，是但丁老師 SOP 最完整的版本（總覽頁只看日線快篩，
這裡才是完整的多週期協同判讀）。
"""

import os
import sys
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from multi_timeframe_check import full_check
from stock_core import STOCK_NAME_MAP

st.set_page_config(page_title="多週期整合分析", layout="wide")


def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


with st.sidebar:
    with st.expander("🔑 API 金鑰狀態"):
        secret_token = _get_secret("FINMIND_TOKEN")
        secret_fugle = _get_secret("FUGLE_API_KEY")

        if secret_token:
            st.success("已使用雲端 Secrets 的 FinMind Token")
            api_token = secret_token
        else:
            api_token = st.text_input("FinMind API Token", type="password")

        if secret_fugle:
            st.success("已使用雲端 Secrets 的 Fugle API Key")
            fugle_api_key = secret_fugle
        else:
            fugle_api_key = st.text_input(
                "Fugle 行情 API Key（60分/5分線用，台股個股/ETF才需要）", type="password"
            )

    stock_names = list(STOCK_NAME_MAP.keys())
    options = ["自訂代碼"] + stock_names

    # 從首頁總覽表點「查看」帶過來的股票：預先填好選項，並自動執行一次。
    auto_run = False
    prefill = st.session_state.pop("prefill_stock", None)
    if prefill:
        match_name = next(
            (n for n, (c, m) in STOCK_NAME_MAP.items()
             if c == prefill["stock_id"] and m == prefill["market"]),
            None,
        )
        if match_name:
            st.session_state["mtf_choice"] = match_name
        else:
            st.session_state["mtf_choice"] = "自訂代碼"
            st.session_state["mtf_custom_code"] = prefill["stock_id"]
            st.session_state["mtf_market"] = prefill["market"]
        auto_run = True

    # 「自訂代碼」放第一個方便找（清單有45檔，放最後要滑很久），
    # 但預設還是選第一檔股票（index=1），不是一打開就跳自訂代碼。
    choice = st.selectbox("選擇股票", options, index=1, key="mtf_choice")
    if choice == "自訂代碼":
        custom_code = st.text_input("輸入股票代碼", value="2330", key="mtf_custom_code")
        market = st.radio("市場", ["TW", "INDEX", "US"], horizontal=True, key="mtf_market")
        stock_id, label = custom_code, custom_code
    else:
        stock_id, market = STOCK_NAME_MAP[choice]
        label = choice

    years_back = st.slider("回溯年數（週線需要夠長的歷史才能算出35週生死線）", 1, 5, 2)

    run_button = st.button("🔍 執行多週期整合分析", type="primary", use_container_width=True)


st.title("🧭 多週期整合分析")
st.caption(
    "依但丁老師 SOP：日/週線判斷長線結構，60分/5分判斷短線進出場時機。"
    "長天期結構轉弱會否決短天期的反彈買訊；短天期轉弱不會推翻長天期多頭結構，"
    "但會提示先讓短打部位出場（為日線留倉，為五分出場）。僅供輔助判讀，不構成投資建議。"
)

if not run_button and not auto_run:
    st.info("在左側選擇股票，按「執行多週期整合分析」開始。")
    st.stop()

if not api_token:
    st.warning("未設定 FinMind API Token，將使用免費額度（較容易觸發流量限制）。")

if market != "TW" and not fugle_api_key:
    pass  # 60分/5分本來就只支援TW，不需要fugle
elif market == "TW" and not fugle_api_key:
    st.info("未設定 Fugle API Key，本次只會跑日線／週線，60分／5分線步驟略過。")

start_date = str(date(date.today().year - years_back, date.today().month, date.today().day))

with st.spinner(f"正在分析 {label}（{stock_id}）的日/週/60分/5分資料..."):
    try:
        result = full_check(stock_id, market, api_token, fugle_api_key or "", start_date)
    except Exception as exc:  # noqa: BLE001
        st.error(f"分析失敗：{exc}")
        st.stop()

final = result["最終建議"]
color = "gray"
if "加碼" in final or ("買進" in final and "賣出" not in final):
    color = "red"
elif "賣出" in final or "減碼" in final:
    color = "green"

st.markdown(f"## :{color}[🎯 最終建議：{final}]")
st.caption(f"{label}（{stock_id}）— 綜合日/週/60分/5分的整合判讀")
st.divider()

st.markdown("### 各週期明細")
tf_order = ["週", "日", "60分", "5分"]
detail = result["各週期明細"]
present_tfs = [tf for tf in tf_order if tf in detail]

if not present_tfs:
    st.warning("沒有任何週期成功取得資料。")
else:
    cols = st.columns(len(present_tfs))
    for i, tf in enumerate(present_tfs):
        d = detail[tf]
        with cols[i]:
            st.markdown(f"**{tf}線**")
            tf_color = "gray"
            if d["結論"] in ("買進", "加碼"):
                tf_color = "red"
            elif d["結論"] == "賣出減碼":
                tf_color = "green"
            st.markdown(f":{tf_color}[**{d['結論']}**]（信心：{d['信心']}）")
            for r in d["理由"]:
                st.caption(f"• {r}")
            for c in d["但書"]:
                st.caption(f"⚠️ {c}")

missing_tfs = [tf for tf in tf_order if tf not in detail]
if missing_tfs:
    reason = "沒有 Fugle API Key 或非台股個股/ETF" if any(t in ("60分", "5分") for t in missing_tfs) else "資料不足"
    st.caption(f"缺少週期：{'、'.join(missing_tfs)}（{reason}），本次判讀僅依現有週期進行。")
