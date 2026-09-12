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

from chart_builder import build_price_chart
from holding_shares import align_to_trading_days, compute_consecutive_signals, fetch_major_holder_trend
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

    # 預設值：第一次載入這個 widget 還沒有 key 時才設，之後都由使用者
    # 輸入或「帶入上面欄位」按鈕決定，不能跟 text_input 的 value= 參數
    # 同時使用（兩邊都給值，Streamlit 會出警告）。
    if "mtf_query" not in st.session_state:
        st.session_state["mtf_query"] = "2330"

    # 從首頁總覽表點「查看」帶過來的股票：直接設定輸入框的值。
    auto_run = False
    prefill = st.session_state.pop("prefill_stock", None)
    if prefill:
        match_name = next(
            (n for n, (c, m) in STOCK_NAME_MAP.items()
             if c == prefill["stock_id"] and m == prefill["market"]),
            None,
        )
        st.session_state["mtf_query"] = match_name or prefill["stock_id"]
        if not match_name:
            st.session_state["mtf_market"] = prefill["market"]
        auto_run = True

    # 只有「一個」輸入框：打清單內的中文名稱、或任意股票代碼都可以，
    # 不用先猜要用哪一格（之前分成「清單搜尋」+「自訂代碼」兩格，
    # 使用者常常在只搜清單的那格打代碼、看到 No results 就以為查不到）。
    query = st.text_input(
        "股票代碼或名稱（清單內用中文名，清單外直接打代碼，例如 3481）",
        key="mtf_query",
    )

    def _apply_pick():
        # 一定要用 on_click callback 改 session_state，不能在按鈕的
        # if 區塊裡直接改：widget 一旦在這次 script run 建立過，
        # 同一輪就不能再改它的 session_state，會丟例外。callback 是在
        # 下一輪 rerun「開始前」執行，這時候還沒建立 widget，才能改。
        st.session_state["mtf_query"] = st.session_state["mtf_pick"]

    with st.expander("📋 從清單快速選擇"):
        st.selectbox("清單股票", stock_names, label_visibility="collapsed", key="mtf_pick")
        st.button("帶入上面欄位", key="mtf_pick_btn", on_click=_apply_pick, use_container_width=True)

    query = st.session_state["mtf_query"].strip()
    if query in STOCK_NAME_MAP:
        stock_id, market = STOCK_NAME_MAP[query]
        label = query
    else:
        stock_id = query
        label = query
        market = st.radio("市場（清單外的代碼才需要選）", ["TW", "INDEX", "US"],
                           horizontal=True, key="mtf_market")

    years_back = st.slider("回溯年數（週線需要夠長的歷史才能算出35週生死線）", 1, 5, 2)

    with st.expander("📊 股權分散表（大股東持股）設定"):
        holding_n = st.slider(
            "連續同方向週數門檻 N", min_value=2, max_value=10, value=3, step=1,
            help="大股東(>400張)持股比例連續N週同方向變化才在日線圖標訊號。"
                 "資料來自TDCC集保結算所，每週由排程自動累積歷史，剛上線或"
                 "剛加進觀察清單的股票可能還沒有足夠週數。",
        )

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
        result = full_check(stock_id, market, api_token, fugle_api_key or "", start_date,
                             include_dataframes=True)
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

# ------------------------------------------------------------------
# K線＋均線走勢圖：跟文字結論放在同一頁，不用再切去「個股詳細分析」頁
# （那頁目前為了精簡導覽列而隱藏，見 pages/_4_個股詳細分析.py 開頭說明）。
# 日線圖另外疊加「股權分散表（大股東持股）」連續同向訊號標記。
# ------------------------------------------------------------------
dataframes = result.get("原始資料", {})
if dataframes:
    st.divider()
    st.markdown("### 價格走勢＋均線")

    # 股權分散表只對日線有意義（TDCC週頻資料）。資料來自本地歷史檔（見
    # holding_shares.py：TDCC官方API免費但只給最新一週，歷史要靠每週
    # 排程自己累積），這裡保留try/except只是防呆，不是預期會查詢失敗。
    holding_signal_rows = pd.DataFrame()
    holding_error = None
    if "日" in dataframes and market == "TW":
        try:
            holding_trend = fetch_major_holder_trend(stock_id)
            if not holding_trend.empty:
                holding_signals = compute_consecutive_signals(holding_trend, n=holding_n)
                holding_signal_rows = align_to_trading_days(holding_signals, dataframes["日"])
        except Exception as exc:  # noqa: BLE001
            holding_error = str(exc)

    for tf in present_tfs:
        if tf not in dataframes:
            continue
        st.markdown(f"**{tf}線**")
        marks = holding_signal_rows if tf == "日" else None
        fig = build_price_chart(dataframes[tf], tf, holding_signals=marks)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{tf}")

    if market == "TW" and "日" in dataframes:
        if holding_error:
            st.warning(f"股權分散表（大股東持股）讀取失敗：{holding_error}\n\n不影響其他技術指標的判讀。")
        elif holding_signal_rows.empty:
            st.caption("🔸 股權分散表（大股東持股）：本地還沒有這檔股票的歷史資料"
                       "（要等每週排程累積，通常是新上線或剛加進觀察清單）。")
        elif len(holding_signal_rows) < holding_n:
            latest = holding_signal_rows.iloc[-1]
            st.caption(
                f"🔸 股權分散表（大股東持股）：目前只累積了 {len(holding_signal_rows)} 週資料"
                f"（要滿 {holding_n} 週才可能出現連續同向訊號），最新一週大股東持股比例 "
                f"{latest['percent']:.2f}%。"
            )
        else:
            marked = holding_signal_rows[holding_signal_rows["signal"] != ""]
            n_up = (marked["signal"] == "籌碼連續集中").sum()
            n_down = (marked["signal"] == "籌碼連續分散").sum()
            st.caption(
                f"🔺 大股東(>400張)持股比例連續 {holding_n} 週同向：籌碼連續集中 {n_up} 次、"
                f"籌碼連續分散 {n_down} 次（標記在日線圖上，滑鼠移過去可看詳細數值）。"
            )
