"""
績效統計 — 讀取「交易日誌」的成交(fills)紀錄，用performance.py配對成
完整交易，呈現勝率/盈虧比/期望值/R倍數/權益曲線/最大回撤/持有天數/與
0050對照等統計。支援多帳戶並排比較，可切換只看system單/manual單/全部，
預設只看system單——放行實單標準不受這個篩選影響，一律固定只用system單
計算，避免練習用的manual單意外影響這個高風險結論。

限制：已平倉交易本身的損益計算不處理除權息調整（持有中的未平倉部位會
套用股票股利調整，見ledger_store.list_positions）；只處理現股多頭（先
買後賣），不支援融資融券/放空；漲跌停判斷方式見fill_rules.py的簡化說明；
與0050對照改用含息（總報酬）序列，報酬率仍是用「總進場成本」當分母的
近似值，不是嚴謹的時間加權報酬率(TWR)。

執行方式：streamlit run streamlit_app.py
"""

import calendar
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from backtest.data_fetch import detect_price_gaps, fetch_total_return_daily
from ledger_store import get_ledger_store, get_secret, require_pin
from performance import check_live_trading_gate, compare_to_benchmark, compute_stats, match_fifo

st.set_page_config(page_title="績效統計", layout="wide", page_icon="📊")

if not require_pin():
    st.stop()

st.title("📊 績效統計")
st.caption(
    "依「交易日誌」的成交紀錄，用FIFO配對成完整交易後計算績效（已平倉損益"
    "本身不處理除權息調整）；只處理現股多頭（先買後賣），不支援融資融券/"
    "放空；與0050對照已改用含息（總報酬）序列，報酬率仍是用「總進場成本」"
    "當分母的近似值，不是嚴謹的時間加權報酬率(TWR)。"
)

with st.sidebar:
    with st.expander("🔑 API 金鑰狀態"):
        secret_token = get_secret("FINMIND_TOKEN")
        if secret_token:
            st.success("已使用雲端 Secrets 的 FinMind Token")
            api_token = secret_token
        else:
            api_token = st.text_input("FinMind API Token", type="password")

store = get_ledger_store()
accounts = store.list_accounts()

if not accounts:
    st.info("目前沒有任何模擬帳戶，請先到「交易日誌」頁新增帳戶並記錄幾筆成交。")
    st.stop()

account_by_id = {a.account_id: a for a in accounts}
# rule_version_tag=="practice"的練習帳戶預設不顯示（不計入任何績效與
# 放行標準的檢視），要看的話手動從多選裡勾回來。
selected_ids = st.multiselect(
    "選擇要比較的帳戶", list(account_by_id.keys()),
    default=[i for i, a in account_by_id.items() if a.rule_version_tag != "practice"],
    format_func=lambda i: f"{i}（{account_by_id[i].label}）",
)
source_choice = st.radio("來源篩選", ["全部", "只看system單", "只看manual單"], index=1, horizontal=True)
source_filter = {"全部": None, "只看system單": "system", "只看manual單": "manual"}[source_choice]
disposition_choice = st.radio("處置股篩選", ["全部", "只看無處置股", "只看有處置股"], index=0, horizontal=True)
disposition_filter = {"全部": None, "只看無處置股": False, "只看有處置股": True}[disposition_choice]
slippage_choice = st.radio(
    "滑價口徑", ["有滑價（實際）", "無滑價（對照）"], index=0, horizontal=True,
    help="目前trading_costs.yaml的滑價pct預設0，兩個口徑數字會一樣；等W4訂出實際滑價數值後才會看出差異",
)
strip_slippage = slippage_choice == "無滑價（對照）"
years_back = st.slider("與0050對照：FinMind回溯年數", 1, 5, 2)

if not selected_ids:
    st.info("請至少選一個帳戶。")
    st.stop()


def _add_months(d: date, months: int) -> date:
    """單純的月份加法，不處理負數月份；日超出目標月份天數時自動取該月
    最後一天（例如1/31+6個月是7/31，8/31+6個月會是2/28或2/29）。"""
    total_month = d.month - 1 + months
    year = d.year + total_month // 12
    month = total_month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _render_account(account_id: str) -> None:
    account = account_by_id[account_id]
    st.subheader(f"{account_id}（{account.label}）")
    if account.go_live_date:
        decision_date = _add_months(account.go_live_date, 6)
        st.caption(f"上線日：{account.go_live_date}　決策日（上線日+6個月）：{decision_date}")
    else:
        st.caption("尚未上線（上線日未填，6個月觀察期還沒開始倒數）")

    fills = store.list_fills(account_id=account_id)
    if not fills:
        st.info("這個帳戶目前沒有任何成交紀錄。")
        return

    closed_all = match_fifo(fills)
    if account.go_live_date:
        closed_all = [t for t in closed_all if t.entry_date >= account.go_live_date]
    closed = closed_all if source_filter is None else [t for t in closed_all if t.source == source_filter]
    closed_system = [t for t in closed_all if t.source == "system"]

    if not closed_all:
        st.info("這個帳戶（上線日之後）沒有任何已平倉的交易。")
        return

    # 上半部的統計/曲線依UI來源篩選；下半部的放行實單標準固定用system單，
    # 不受篩選影響——所以就算目前篩選下沒有交易，也不能提早return，gate
    # 區塊還是要渲染。
    if not closed:
        st.info("這個帳戶（在目前的來源篩選下）沒有已平倉的交易。")
    else:
        stats = compute_stats(closed, disposition_filter=disposition_filter, strip_slippage=strip_slippage)
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

    st.markdown("**與 0050 買進持有對照（含息）**")
    start = min(t.entry_date for t in closed_all)
    end = max(t.exit_date for t in closed_all)
    fetch_start = start - timedelta(days=365 * years_back)
    price_df = fetch_total_return_daily("0050", "TW", fetch_start.isoformat(), end.isoformat(), api_token)
    if price_df.empty:
        st.warning("抓不到0050歷史價格，請確認FinMind Token設定正確。")
        comparison = {"portfolio_return": None, "benchmark_return": None}
        gate_comparison = {"portfolio_return": None, "benchmark_return": None}
    else:
        # TaiwanStockDividendResult只涵蓋除權息，不含股票分割（0050在
        # 2025-06有1拆4分割）——交易期間內若還有這種未被還原的價格斷層，
        # 對照結果會嚴重失真，用現成的detect_price_gaps()偵測並警告。
        in_period = price_df[pd.to_datetime(price_df["date"]).dt.date >= start]
        gaps = detect_price_gaps(in_period.reset_index(drop=True))
        if not gaps.empty:
            st.warning(
                f"0050在交易期間內有{len(gaps)}個未被除權息資料涵蓋的單日跳空"
                f"（例如股票分割），含息對照可能失真：{gaps['date'].tolist()}"
            )
        benchmark_prices = list(zip(pd.to_datetime(price_df["date"]).dt.date, price_df["close"]))
        comparison = compare_to_benchmark(closed, benchmark_prices)
        gate_comparison = compare_to_benchmark(closed_system, benchmark_prices)
        if comparison["portfolio_return"] is not None:
            col_b1, col_b2 = st.columns(2)
            col_b1.metric("帳戶報酬率（近似值）", f"{comparison['portfolio_return']:.1%}")
            col_b2.metric("0050同期買進持有報酬率（含息）", f"{comparison['benchmark_return']:.1%}")

    st.markdown("**放行實單標準**（固定只用system單計算，不受上方來源篩選影響）")
    if not closed_system:
        st.info("這個帳戶目前沒有system單的已平倉交易，無法計算放行標準。")
        return
    gate_stats = compute_stats(closed_system)
    gate = check_live_trading_gate(
        gate_stats, base_equity=account.initial_capital,
        portfolio_return=gate_comparison["portfolio_return"],
        benchmark_return=gate_comparison["benchmark_return"],
    )
    status_display = {"達標": "✅ 達標", "未達標": "❌ 未達標", "資料不足": "ℹ️ 資料不足"}
    (st.success if gate.status == "達標" else st.info if gate.status == "資料不足" else st.warning)(
        status_display[gate.status]
    )
    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric(
        "扣成本後優於0050",
        "—" if gate.beats_benchmark is None else ("是" if gate.beats_benchmark else "否"),
    )
    col_g2.metric(
        "最大回撤",
        "—" if gate.max_drawdown_pct is None else f"{gate.max_drawdown_pct:.1%}",
        help="門檻：≤15%",
    )
    col_g3.metric(
        "交易筆數（system單）",
        f"{gate_stats.total_trades}",
        help="門檻：≥30筆",
    )
    st.caption(f"前向結果與回測是否相近：{gate.forward_vs_backtest_note}")


if len(selected_ids) == 1:
    _render_account(selected_ids[0])
else:
    columns = st.columns(len(selected_ids))
    for col, account_id in zip(columns, selected_ids):
        with col:
            _render_account(account_id)
