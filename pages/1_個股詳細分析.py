"""
台股技術指標 互動式 Dashboard — 個股詳細分析
單一股票的四關價／K線＋均線／MACD／OBV／MTM／CCI／KD 詳細圖表與 SOP 結論。
想看整份清單一次性的買賣總覽，請切回左側選單的「streamlit_app」首頁。
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sop_decision import evaluate_timeframe

from stock_core import (
    STOCK_NAME_MAP,
    TIMEFRAME_MA_PERIODS,
    FAST_MA,
    KEY_MA,
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
    st.header("設定")

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

    st.divider()

    stock_names = list(STOCK_NAME_MAP.keys())
    choice = st.selectbox("選擇股票", stock_names + ["自訂代碼"])

    if choice == "自訂代碼":
        custom_code = st.text_input("輸入股票代碼", value="2330")
        stock_id, market = custom_code, "TW"
        label = custom_code
    else:
        stock_id, market = STOCK_NAME_MAP[choice]
        label = choice

    timeframe = st.radio("週期", ["日", "60", "5"], horizontal=True,
                          format_func=lambda x: {"日": "日線", "60": "60分線", "5": "5分線"}[x])

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
palette = ["#1e88e5", "#fb8c00", "#8e24aa", "#3949ab"]
ma_colors = {f"MA{p}": palette[i % len(palette)] for i, p in enumerate(ma_periods)}
for ma_col, color in ma_colors.items():
    if ma_col in df.columns:
        fig_price.add_trace(go.Scatter(
            x=df["date"], y=df[ma_col], mode="lines",
            name=f"{ma_col}（生死線）" if ma_col == key_ma_col else ma_col,
            line=dict(color=color, width=1.3),
        ))
fig_price.update_layout(height=420, xaxis_rangeslider_visible=False,
                         margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_price, use_container_width=True)

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
