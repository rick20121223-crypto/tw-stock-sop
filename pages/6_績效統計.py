"""
績效統計 — 讀取「交易日誌」的成交(fills)紀錄，用performance.py配對成
完整交易，呈現勝率/盈虧比/期望值/R倍數/權益曲線/最大回撤/持有天數/與
0050對照等統計。支援多帳戶並排比較，可切換只看system單/manual單/全部。

限制：不處理除權息調整；只處理現股多頭（先買後賣），不支援融資融券/
放空；漲跌停判斷方式見fill_rules.py的簡化說明；與0050對照的報酬率是用
「總進場成本」當分母的近似值，不是嚴謹的時間加權報酬率(TWR)。

執行方式：streamlit run streamlit_app.py
"""

from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from ledger_store import CSVLedgerStore, get_secret, require_pin
from performance import compare_to_benchmark, compute_stats, match_fifo
from stock_core import get_stock_data

st.set_page_config(page_title="績效統計", layout="wide", page_icon="📊")

if not require_pin():
    st.stop()

st.title("📊 績效統計")
st.caption(
    "依「交易日誌」的成交紀錄，用FIFO配對成完整交易後計算績效。不處理"
    "除權息調整；只處理現股多頭（先買後賣），不支援融資融券/放空；與"
    "0050對照的報酬率是用「總進場成本」當分母的近似值，不是嚴謹的時間"
    "加權報酬率(TWR)。"
)

with st.sidebar:
    with st.expander("🔑 API 金鑰狀態"):
        secret_token = get_secret("FINMIND_TOKEN")
        if secret_token:
            st.success("已使用雲端 Secrets 的 FinMind Token")
            api_token = secret_token
        else:
            api_token = st.text_input("FinMind API Token", type="password")

store = CSVLedgerStore()
accounts = store.list_accounts()

if not accounts:
    st.info("目前沒有任何模擬帳戶，請先到「交易日誌」頁新增帳戶並記錄幾筆成交。")
    st.stop()

account_by_id = {a.account_id: a for a in accounts}
selected_ids = st.multiselect(
    "選擇要比較的帳戶", list(account_by_id.keys()),
    default=list(account_by_id.keys()),
    format_func=lambda i: f"{i}（{account_by_id[i].label}）",
)
source_choice = st.radio("來源篩選", ["全部", "只看system單", "只看manual單"], horizontal=True)
source_filter = {"全部": None, "只看system單": "system", "只看manual單": "manual"}[source_choice]
years_back = st.slider("與0050對照：FinMind回溯年數", 1, 5, 2)

if not selected_ids:
    st.info("請至少選一個帳戶。")
    st.stop()


def _render_account(account_id: str) -> None:
    account = account_by_id[account_id]
    st.subheader(f"{account_id}（{account.label}）")

    fills = store.list_fills(account_id=account_id)
    if not fills:
        st.info("這個帳戶目前沒有任何成交紀錄。")
        return

    closed = match_fifo(fills)
    if source_filter is not None:
        closed = [t for t in closed if t.source == source_filter]

    if not closed:
        st.info("這個帳戶（在目前的來源篩選下）沒有已平倉的交易。")
        return

    stats = compute_stats(closed)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總交易筆數", stats.total_trades)
    col2.metric("勝率", f"{stats.win_rate:.1%}")
    col3.metric("盈虧比", f"{stats.profit_factor:.2f}" if stats.profit_factor is not None else "—")
    col4.metric("每筆期望值", f"{stats.expectancy:,.0f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("平均獲利", f"{stats.avg_win:,.0f}")
    col6.metric("平均虧損", f"{stats.avg_loss:,.0f}")
    col7.metric(
        "平均R倍數",
        f"{stats.avg_r_multiple:.2f}" if stats.avg_r_multiple is not None else "—",
        help=f"{stats.trades_with_r}/{stats.total_trades} 筆有填停損價，僅這些筆數納入R倍數統計",
    )
    col8.metric("平均持有天數", f"{stats.avg_holding_days:.1f}")
    st.metric("最大回撤", f"{stats.max_drawdown:,.0f}")

    if stats.equity_curve:
        curve_df = pd.DataFrame(stats.equity_curve, columns=["日期", "累積損益"])
        st.plotly_chart(
            px.line(curve_df, x="日期", y="累積損益", title=f"{account_id} 累積損益曲線"),
            use_container_width=True,
        )

    st.markdown("**與 0050 買進持有對照**")
    start = min(t.entry_date for t in closed)
    end = max(t.exit_date for t in closed)
    fetch_start = start - timedelta(days=365 * years_back)
    price_df = get_stock_data("0050", "TW", fetch_start.isoformat(), end.isoformat(), api_token)
    if price_df.empty:
        st.warning("抓不到0050歷史價格，請確認FinMind Token設定正確。")
        return
    benchmark_prices = list(zip(pd.to_datetime(price_df["date"]).dt.date, price_df["close"]))
    comparison = compare_to_benchmark(closed, benchmark_prices)
    if comparison["portfolio_return"] is not None:
        col_b1, col_b2 = st.columns(2)
        col_b1.metric("帳戶報酬率（近似值）", f"{comparison['portfolio_return']:.1%}")
        col_b2.metric("0050同期買進持有報酬率", f"{comparison['benchmark_return']:.1%}")


if len(selected_ids) == 1:
    _render_account(selected_ids[0])
else:
    columns = st.columns(len(selected_ids))
    for col, account_id in zip(columns, selected_ids):
        with col:
            _render_account(account_id)
