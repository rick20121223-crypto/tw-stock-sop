"""
共用的 K線＋均線 Plotly 圖表建構＋股權分散表訊號標註。

被 pages/3_多週期整合分析.py 用來把「原本只有文字結論」的多週期整合
分析頁補上K線圖（pages/_4_個股詳細分析.py 目前隱藏在導覽列外，使用者
看不到裡面的K線圖，所以在使用者還看得到的頁面上補一份）。

只抽「K線＋均線」這張主圖（技術上最常被稱為「線圖」的那張），MACD／
OBV／MTM／CCI／KD 那幾張細部圖表還是只有 pages/_4_個股詳細分析.py
有，這裡先不重複做一份，之後有需要再擴充。
"""

import plotly.graph_objects as go

from stock_core import HALF_YEAR_MA, KEY_MA, TIMEFRAME_MA_PERIODS, normalize_timeframe

# 台股慣例：紅漲綠跌（跟美股相反）
UP_COLOR = "#e53935"
DOWN_COLOR = "#43a047"

_MA_PALETTE = ["#1e88e5", "#fb8c00", "#8e24aa", "#3949ab"]


def build_price_chart(df, timeframe_label: str, holding_signals=None, height: int = 380) -> go.Figure:
    """
    df：已經跑過 run_all_indicators() 的 K線 DataFrame（含 date/open/max/
    min/close 與各條 MA 欄位）。
    timeframe_label：「週」「日」「60分」「5分」（或對應的「週線」「日線」
    等寫法），決定要畫哪組週期的均線。
    holding_signals：可選，align_to_trading_days() 的輸出（帶 signal 欄
    位），非空時會在圖上加「籌碼連續集中/分散」標記；只有日線適用。
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["max"], low=df["min"], close=df["close"],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
        name="K線",
    ))

    tf_key = normalize_timeframe(timeframe_label)
    ma_periods = TIMEFRAME_MA_PERIODS[tf_key]
    key_ma_col = KEY_MA[tf_key]
    half_year_ma_col = HALF_YEAR_MA.get(tf_key)
    ma_colors = {f"MA{p}": _MA_PALETTE[i % len(_MA_PALETTE)] for i, p in enumerate(ma_periods)}
    for ma_col, color in ma_colors.items():
        if ma_col in df.columns:
            if ma_col == key_ma_col:
                label = f"{ma_col}（生死線）"
            elif ma_col == half_year_ma_col:
                label = f"{ma_col}（半年線）"
            else:
                label = ma_col
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[ma_col], mode="lines",
                name=label, line=dict(color=color, width=1.3),
            ))

    if holding_signals is not None and not holding_signals.empty:
        _add_holding_markers(fig, holding_signals)

    fig.update_layout(height=height, xaxis_rangeslider_visible=False,
                       margin=dict(l=10, r=10, t=30, b=10))
    return fig


def _add_holding_markers(fig: go.Figure, holding_signals) -> None:
    marked = holding_signals[holding_signals["signal"] != ""]
    up_pts = marked[marked["signal"] == "籌碼連續集中"]
    down_pts = marked[marked["signal"] == "籌碼連續分散"]

    def _add(points, color, symbol, name, above):
        if points.empty:
            return
        y = points["high"] * 1.02 if above else points["low"] * 0.98
        fig.add_trace(go.Scatter(
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

    _add(up_pts, UP_COLOR, "triangle-up", "籌碼連續集中", above=True)
    _add(down_pts, DOWN_COLOR, "triangle-down", "籌碼連續分散", above=False)
