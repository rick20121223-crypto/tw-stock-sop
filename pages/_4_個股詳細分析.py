"""
台股技術指標 互動式 Dashboard — 個股詳細分析
單一股票的四關價／K線＋均線／MACD／OBV／MTM／CCI／KD 詳細圖表與 SOP 結論。
想看整份清單一次性的買賣總覽，請切回左側選單的「streamlit_app」首頁。

檔名開頭底線是刻意的：依使用者指示把導覽列精簡成只留「短線進場／
長線留倉／多週期整合分析」3頁，這頁改成不顯示在側邊選單（但程式碼、
功能都還在，之後想恢復顯示，把檔名開頭的底線拿掉、恢復數字排序即可）。
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from holding_shares import (
    align_to_trading_days,
    compute_consecutive_signals,
    fetch_major_holder_trend,
)
from sop_decision import evaluate_timeframe

from stock_core import (
    STOCK_NAME_MAP,
    TIMEFRAME_MA_PERIODS,
    FAST_MA,
    KEY_MA,
    HALF_YEAR_MA,
    get_stock_data,
    get_intraday_data,
    run_all_indicators,
    normalize_timeframe,
    ma_slope,
)

st.set_page_config(page_title="個股詳細分析", layout="wide")

# 台股慣例：紅漲綠跌（跟美股相反）
UP_COLOR = "#e53935"    # 紅
DOWN_COLOR = "#43a047"  # 綠
FLAT_COLOR = "#9e9e9e"  # 灰


def colored_text(value: float, label: str = "") -> str:
    color = FLAT_COLOR
    if value > 0:
        color = UP_COLOR
    elif value < 0:
        color = DOWN_COLOR
    return f"<span style='color:{color}; font-weight:600'>{label}{value:,.2f}</span>"


def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


# ------------------------------------------------------------------
# 側邊欄：設定區
# ------------------------------------------------------------------
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
            fugle_api_key = st.text_input("Fugle 行情 API Key（60分/5分線用）", type="password")

    stock_names = list(STOCK_NAME_MAP.keys())

    if "detail_query" not in st.session_state:
        st.session_state["detail_query"] = "2330"

    # 只有「一個」輸入框：打清單內的中文名稱、或任意股票代碼都可以，
    # 不用先猜要用哪一格。
    query = st.text_input(
        "股票代碼或名稱（清單內用中文名，清單外直接打代碼，例如 3481）",
        key="detail_query",
    )

    def _apply_pick():
        # 用 on_click callback 改 session_state：widget 一旦在這次
        # script run 建立過，同一輪就不能再改它的 session_state，
        # 要在下一輪 rerun「開始前」執行才行。
        st.session_state["detail_query"] = st.session_state["detail_pick"]

    with st.expander("📋 從清單快速選擇"):
        st.selectbox("清單股票", stock_names, label_visibility="collapsed", key="detail_pick")
        st.button("帶入上面欄位", key="detail_pick_btn", on_click=_apply_pick, use_container_width=True)

    query = st.session_state["detail_query"].strip()
    if query in STOCK_NAME_MAP:
        stock_id, market = STOCK_NAME_MAP[query]
        label = query
    else:
        stock_id = query
        label = query
        market = st.radio("市場（清單外的代碼才需要選）", ["TW", "INDEX", "US"], horizontal=True,
                           help="TW=台股個股/ETF，INDEX=大盤指數，US=美股")

    timeframe = st.radio("週期", ["日", "60", "5"], horizontal=True,
                          format_func=lambda x: {"日": "日線", "60": "60分線", "5": "5分線"}[x])

    with st.expander("📊 股權分散表（大股東持股）設定"):
        holding_n = st.slider(
            "連續同方向週數門檻 N", min_value=2, max_value=10, value=3, step=1,
            help="大股東(>400張)持股比例連續N週同方向變化才在K線圖標訊號。"
                 "只有台股個股/ETF的「日線」查詢才會顯示這個功能。資料來自"
                 "TDCC集保結算所，每週由排程自動累積歷史，剛上線或剛加進"
                 "觀察清單的股票可能還沒有足夠週數。",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("起始日期", value=pd.to_datetime("2025-01-01"))
    with col_b:
        end_date = st.date_input("結束日期", value=pd.to_datetime("today"))

    run_button = st.button("查詢", type="primary", use_container_width=True)


# ------------------------------------------------------------------
# 主畫面
# ------------------------------------------------------------------
st.title("📊 個股詳細分析")

if not run_button:
    st.info("在左側設定股票與週期，按「查詢」開始分析。")
    st.stop()

# --- 抓資料 ---
with st.spinner("資料抓取中..."):
    if timeframe == "日":
        df = get_stock_data(stock_id, market, str(start_date), str(end_date), api_token)
    else:
        if market != "TW":
            st.warning(f"目前分K資料只支援台股個股/ETF，{market} 類型暫不支援。")
            st.stop()
        if not fugle_api_key:
            st.warning("查詢分K線需要先在左側輸入 Fugle 行情 API Key。")
            st.stop()
        df = get_intraday_data(stock_id, timeframe, fugle_api_key)

if df.empty:
    st.error(f"查無資料，請確認代碼 {stock_id} 或日期區間是否正確。")
    st.stop()

df = run_all_indicators(df, timeframe)
latest = df.iloc[-1]

timeframe_label = {"日": "日線", "60": "60分線", "5": "5分線"}[timeframe]

# --- SOP自動判讀結論卡片（放在四關價區塊之前）---
verdict = evaluate_timeframe(df, timeframe_label)
conclusion_color = {"買進": "red", "加碼": "red", "賣出減碼": "green", "觀望": "gray"}
color = conclusion_color.get(verdict.conclusion.split("（")[0], "gray")
st.markdown(f"## :{color}[🎯 SOP結論：{verdict.conclusion}]（信心：{verdict.confidence}）")
st.markdown("**理由：** " + "；".join(verdict.reasons))
if verdict.caveats:
    st.warning("；".join(verdict.caveats))
st.divider()

st.subheader(f"{label}（{stock_id}）— {timeframe_label}")

# ------------------------------------------------------------------
# 區塊 1：四關價 + 摘要卡片
# ------------------------------------------------------------------
if timeframe == "日":
    st.markdown("### 四關價")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("今開", f"{latest['今開']:.2f}")
    c2.metric("昨高", f"{latest['昨高']:.2f}")
    c3.metric("昨低", f"{latest['昨低']:.2f}")
    c4.metric("昨收", f"{latest['昨收']:.2f}")

    if latest["今開"] < latest["昨低"]:
        st.markdown(f":green[**→ 今開跌破昨低，格局偏弱**]")
    elif latest["今開"] > latest["昨高"]:
        st.markdown(f":red[**→ 今開站上昨高，格局偏強**]")
    else:
        st.markdown("→ 今開落在昨日高低區間內，格局中性")
else:
    st.caption("四關價是「昨日 vs 今日」的日線概念，分K線暫不適用。")

st.divider()

# ------------------------------------------------------------------
# 區塊 2：K線 + 均線圖
# ------------------------------------------------------------------
st.markdown("### 價格走勢 + 均線")

fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(
    x=df["date"], open=df["open"], high=df["max"], low=df["min"], close=df["close"],
    increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
    name="K線",
))
tf_key = normalize_timeframe(timeframe)
ma_periods = TIMEFRAME_MA_PERIODS[tf_key]
key_ma_col = KEY_MA[tf_key]
half_year_ma_col = HALF_YEAR_MA.get(tf_key)
palette = ["#1e88e5", "#fb8c00", "#8e24aa", "#3949ab"]
ma_colors = {f"MA{p}": palette[i % len(palette)] for i, p in enumerate(ma_periods)}
for ma_col, color in ma_colors.items():
    if ma_col in df.columns:
        if ma_col == key_ma_col:
            label = f"{ma_col}（生死線）"
        elif ma_col == half_year_ma_col:
            label = f"{ma_col}（半年線）"
        else:
            label = ma_col
        fig_price.add_trace(go.Scatter(
            x=df["date"], y=df[ma_col], mode="lines",
            name=label,
            line=dict(color=color, width=1.3),
        ))

# ------------------------------------------------------------------
# 股權分散表（大股東持股比例）連續同向訊號，標註在K線圖上。只有台股
# 個股/ETF的日線適用（股權分散表是TDCC週頻資料，跟60分/5分、INDEX/US
# 沒有對應關係）。資料來自本地歷史檔（見 holding_shares.py 開頭說明：
# TDCC官方API免費但只給最新一週，歷史要靠每週排程自己累積），讀取
# 本身很單純，這裡保留try/except只是防呆（例如歷史檔案損毀），不是
# 因為預期會查詢失敗。
# ------------------------------------------------------------------
holding_signal_rows = pd.DataFrame()
holding_error = None
if timeframe == "日" and market == "TW":
    try:
        holding_trend = fetch_major_holder_trend(stock_id)
        if not holding_trend.empty:
            holding_signals = compute_consecutive_signals(holding_trend, n=holding_n)
            holding_signal_rows = align_to_trading_days(holding_signals, df)
    except Exception as exc:  # noqa: BLE001
        holding_error = str(exc)

if not holding_signal_rows.empty:
    marked = holding_signal_rows[holding_signal_rows["signal"] != ""]
    up_pts = marked[marked["signal"] == "籌碼連續集中"]
    down_pts = marked[marked["signal"] == "籌碼連續分散"]

    def _add_holding_markers(points: pd.DataFrame, color: str, symbol: str,
                              name: str, above: bool) -> None:
        if points.empty:
            return
        y = points["high"] * 1.02 if above else points["low"] * 0.98
        fig_price.add_trace(go.Scatter(
            x=points["trade_date"], y=y,
            mode="markers+text",
            marker=dict(symbol=symbol, size=13, color=color, line=dict(width=1, color="white")),
            text=[name] * len(points),
            textposition="top center" if above else "bottom center",
            textfont=dict(color=color, size=11),
            name=name,
            customdata=points[["percent", "diff", "people"]].values,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>大股東(>400張)持股 %{customdata[0]:.2f}%"
                "<br>較前週 %{customdata[1]:+.2f}%<br>合計人數 %{customdata[2]:,.0f}"
                f"<extra>{name}</extra>"
            ),
        ))

    _add_holding_markers(up_pts, UP_COLOR, "triangle-up", "籌碼連續集中", above=True)
    _add_holding_markers(down_pts, DOWN_COLOR, "triangle-down", "籌碼連續分散", above=False)

fig_price.update_layout(height=420, xaxis_rangeslider_visible=False,
                         margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_price, use_container_width=True)

if timeframe == "日" and market == "TW":
    if holding_error:
        st.warning(f"股權分散表（大股東持股）讀取失敗：{holding_error}\n\n"
                   "不影響其他技術指標的判讀。")
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
        n_up, n_down = len(up_pts), len(down_pts)
        st.caption(
            f"🔺 大股東(>400張)持股比例連續 {holding_n} 週同向：籌碼連續集中 {n_up} 次、"
            f"籌碼連續分散 {n_down} 次（三角形標記在K線圖上，滑鼠移過去可看詳細數值）。"
        )
        with st.expander("查看股權分散表訊號明細"):
            display_cols = holding_signal_rows[holding_signal_rows["signal"] != ""][
                ["trade_date", "percent", "diff", "run_length", "signal"]
            ].rename(columns={"trade_date": "對應交易日", "percent": "大股東持股%",
                               "diff": "較前週變化", "run_length": "連續週數"})
            if display_cols.empty:
                st.caption("尚無訊號")
            else:
                st.dataframe(display_cols, use_container_width=True, hide_index=True)

st.caption(f"本週期（{timeframe_label}）依 SOP 使用 {' / '.join(ma_colors.keys())}，"
           f"其中 **{key_ma_col}** 是生死線／多空分水嶺，斜率比價位本身更重要。")
ma_cols_display = st.columns(len(ma_periods))
for i, p in enumerate(ma_periods):
    col = f"MA{p}"
    if col in df.columns and pd.notna(latest.get(col)):
        slope = ma_slope(df, col)
        above = "站上" if latest["close"] >= latest[col] else "跌破"
        label = f"{col}⭐生死線" if col == key_ma_col else col
        ma_cols_display[i].metric(label, f"{latest[col]:.2f}", f"{above}｜{slope}")
    else:
        ma_cols_display[i].metric(col, "資料不足", "")

st.divider()

# ------------------------------------------------------------------
# 區塊 3：MACD
# ------------------------------------------------------------------
st.markdown("### MACD")
fig_macd = make_subplots(specs=[[{"secondary_y": False}]])
bar_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in df["MACD_hist"]]
fig_macd.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], name="柱狀圖", marker_color=bar_colors))
fig_macd.add_trace(go.Scatter(x=df["date"], y=df["DIF"], mode="lines", name="DIF", line=dict(color="#1e88e5")))
fig_macd.add_trace(go.Scatter(x=df["date"], y=df["MACD_signal"], mode="lines", name="訊號線", line=dict(color="#fb8c00")))
fig_macd.add_hline(y=0, line_dash="dot", line_color="gray")
fig_macd.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_macd, use_container_width=True)
zone = "零軸之上" if latest["DIF"] > 0 else "零軸之下"
st.markdown(f"DIF={latest['DIF']:.3f}，訊號線={latest['MACD_signal']:.3f} → **DIF 位於{zone}**")

st.divider()

# ------------------------------------------------------------------
# 區塊 4：OBV
# ------------------------------------------------------------------
st.markdown("### OBV（量價驗證）")
if pd.isna(latest.get("OBV")):
    st.caption("此標的無成交量資料（如加權指數），OBV 無法計算。")
else:
    fig_obv = go.Figure()
    fig_obv.add_trace(go.Scatter(x=df["date"], y=df["OBV"], mode="lines", name="OBV", line=dict(color="#3949ab")))
    fig_obv.add_trace(go.Scatter(x=df["date"], y=df["OBV_MA"], mode="lines", name="OBV_MA", line=dict(color="#fb8c00", dash="dot")))
    fig_obv.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_obv, use_container_width=True)
    st.markdown(f"近期趨勢：**{ma_slope(df, 'OBV_MA')}**")

st.divider()

# ------------------------------------------------------------------
# 區塊 5：MTM
# ------------------------------------------------------------------
st.markdown(f"### MTM（{timeframe_label}）")
fig_mtm = go.Figure()
fig_mtm.add_trace(go.Scatter(x=df["date"], y=df["MTM"], mode="lines", name="MTM", line=dict(color="#00897b")))
fig_mtm.add_trace(go.Scatter(x=df["date"], y=df["MTM_MA"], mode="lines", name="MTM_MA", line=dict(color="#fb8c00", dash="dot")))
fig_mtm.add_hline(y=0, line_dash="dot", line_color="gray")
fig_mtm.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_mtm, use_container_width=True)

df["MTM_signal"] = "-"
cross_up = (df["MTM"] > 0) & (df["MTM"].shift(1) <= 0)
cross_down = (df["MTM"] < 0) & (df["MTM"].shift(1) >= 0)
df.loc[cross_up, "MTM_signal"] = "翻多"
df.loc[cross_down, "MTM_signal"] = "翻空"
recent_signals = df[df["MTM_signal"] != "-"][["date", "close", "MTM", "MTM_signal"]].tail(5)
if not recent_signals.empty:
    st.markdown("近期翻多/翻空紀錄：")
    st.dataframe(recent_signals, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# 區塊 5.5：CCI（三代領先指標，配合MTM判斷零軸突破/跌破與背離）
# ------------------------------------------------------------------
st.markdown(f"### CCI（{timeframe_label}）")
fig_cci = go.Figure()
fig_cci.add_trace(go.Scatter(x=df["date"], y=df["CCI"], mode="lines", name="CCI", line=dict(color="#d81b60")))
fig_cci.add_hline(y=100, line_dash="dot", line_color="gray")
fig_cci.add_hline(y=-100, line_dash="dot", line_color="gray")
fig_cci.add_hline(y=0, line_dash="dot", line_color="lightgray")
fig_cci.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_cci, use_container_width=True)
if latest["CCI"] > 100:
    st.markdown(":red[**→ CCI 站上+100，強勢突破區**]")
elif latest["CCI"] < -100:
    st.markdown(":green[**→ CCI 跌破-100，弱勢跌破區**]")
else:
    st.markdown(f"CCI = {latest['CCI']:.2f}，介於±100之間，中性區")

st.divider()

# ------------------------------------------------------------------
# 區塊 6：KD
# ------------------------------------------------------------------
st.markdown("### KD（參考用）")
if market == "INDEX":
    st.caption("加權指數無開高低資料，K/D 僅供參考，不代表真實盤中高低點。")
fig_kd = go.Figure()
fig_kd.add_trace(go.Scatter(x=df["date"], y=df["K"], mode="lines", name="K", line=dict(color="#1e88e5")))
fig_kd.add_trace(go.Scatter(x=df["date"], y=df["D"], mode="lines", name="D", line=dict(color="#fb8c00")))
fig_kd.add_hline(y=80, line_dash="dot", line_color="gray")
fig_kd.add_hline(y=20, line_dash="dot", line_color="gray")
fig_kd.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_kd, use_container_width=True)

st.caption("以上資料可截圖或複製貼給但丁老師 SOP 技能，請它依此判讀買賣/加碼/觀望建議。")
