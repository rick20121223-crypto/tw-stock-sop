"""
交易日誌 — 紙上交易記帳：管理模擬帳戶，記錄委託(orders)與成交(fills)，
檢視目前持倉(positions，由fills用FIFO算出的派生快照)。

這個月（第1個月）還沒有自動進出場引擎（排在第3個月），這一頁讓你手動
輸入委託/成交來驗證整套記帳結構跑得起來——之後引擎上線後會直接寫入
同一套 ledger_store，這一頁不用改。

限制：不處理除權息調整；只處理現股多頭（先買後賣），不支援融資融券/
放空；漲跌停判斷方式見 fill_rules.py 的簡化說明（這一頁手動輸入成交時
不會自動套用漲跌停順延，由你自己決定要不要真的記那一筆）。

執行方式：streamlit run streamlit_app.py
"""

from datetime import date

import pandas as pd
import streamlit as st

from ledger_store import Account, CSVLedgerStore, Order, require_pin
from performance import Fill, calc_fee, calc_sell_tax

st.set_page_config(page_title="交易日誌", layout="wide", page_icon="📝")

if not require_pin():
    st.stop()

st.title("📝 交易日誌")
st.caption(
    "紙上交易記帳：100萬模擬本金，依SOP訊號進出場（第3個月才會有自動引擎，"
    "這個月先用這頁手動輸入委託/成交來驗證記帳結構）。不處理除權息調整；"
    "只處理現股多頭（先買後賣），不支援融資融券/放空。"
)

store = CSVLedgerStore()
REASON_CATEGORIES = ["買進", "加碼", "觀望", "賣出減碼"]

# ------------------------------------------------------------------
# 帳戶管理
# ------------------------------------------------------------------
st.subheader("帳戶")
accounts = store.list_accounts()
if accounts:
    st.dataframe(
        pd.DataFrame([
            {
                "account_id": a.account_id, "名稱": a.label, "規則版本標籤": a.rule_version_tag,
                "初始本金": a.initial_capital, "單筆部位上限": f"{a.max_position_pct:.0%}",
                "持股檔數上限": a.max_holdings_count, "建立日期": a.created_date,
                "啟用中": a.is_active,
            }
            for a in accounts
        ]),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("目前還沒有任何模擬帳戶，請先在下方新增一個。")

with st.expander("➕ 新增帳戶", expanded=not accounts):
    with st.form("add_account_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_account_id = st.text_input("帳戶ID（例如 v1）")
            new_label = st.text_input("名稱")
            new_rule_tag = st.text_input("規則版本標籤（例如 sop_rules v1.3）")
        with col2:
            new_capital = st.number_input("初始本金", min_value=0.0, value=1_000_000.0, step=10_000.0)
            new_max_pct = st.slider("單筆部位上限（占目前帳戶淨值）", 0.01, 1.0, 0.2)
            new_max_holdings = st.number_input("持股檔數上限", min_value=1, value=10, step=1)
        if st.form_submit_button("新增帳戶"):
            if not new_account_id or not new_label:
                st.error("請填帳戶ID與名稱")
            else:
                try:
                    store.add_account(Account(
                        account_id=new_account_id.strip(), label=new_label,
                        rule_version_tag=new_rule_tag, initial_capital=new_capital,
                        max_position_pct=new_max_pct, max_holdings_count=int(new_max_holdings),
                        created_date=date.today(),
                    ))
                    st.success("已新增帳戶")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

if not accounts:
    st.stop()

account_ids = [a.account_id for a in accounts]
view_account = st.selectbox("目前檢視帳戶", ["全部"] + account_ids)
filter_account_id = None if view_account == "全部" else view_account

tab_orders, tab_fills, tab_positions = st.tabs(["委託 Orders", "成交 Fills", "持倉 Positions"])

# ------------------------------------------------------------------
# 委託 Orders
# ------------------------------------------------------------------
with tab_orders:
    with st.expander("➕ 新增委託", expanded=False):
        with st.form("add_order_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                o_account = st.selectbox("帳戶", account_ids, key="order_account")
                o_code = st.text_input("股票代號", key="order_code")
                o_side = st.radio("買或賣", ["買", "賣"], horizontal=True, key="order_side")
            with col2:
                o_source = st.radio("來源", ["manual", "system"], horizontal=True, key="order_source")
                o_signal_date = st.date_input("訊號日期", value=date.today(), key="order_signal_date")
                o_shares = st.number_input("委託股數", min_value=0, step=1, key="order_shares")
            with col3:
                o_reason = st.selectbox("進場理由（SOP結論）", [""] + REASON_CATEGORIES, key="order_reason")
                o_stop_loss = st.number_input("預設停損價（可不填）", min_value=0.0, step=0.01, value=0.0, key="order_sl")
                o_target = st.number_input("預設目標價（可不填）", min_value=0.0, step=0.01, value=0.0, key="order_tp")
            o_detail = st.text_area("進場理由補充（reasons/caveats文字快照，自由輸入）", key="order_detail")
            if st.form_submit_button("新增委託"):
                if not o_code or o_shares <= 0:
                    st.error("請填代號與委託股數")
                else:
                    store.add_order(Order(
                        account_id=o_account, code=o_code.strip(), side=o_side, source=o_source,
                        signal_date=o_signal_date, entry_reason=o_reason or None,
                        entry_reason_detail=o_detail, planned_stop_loss=o_stop_loss or None,
                        planned_target=o_target or None, requested_shares=int(o_shares),
                    ))
                    st.success("已新增委託")
                    st.rerun()

    orders = store.list_orders(account_id=filter_account_id)
    st.subheader(f"委託紀錄（{len(orders)} 筆）")
    if orders:
        st.dataframe(
            pd.DataFrame([
                {
                    "id": o.id, "帳戶": o.account_id, "訊號日期": o.signal_date, "代號": o.code,
                    "買賣": o.side, "來源": o.source, "進場理由": o.entry_reason or "",
                    "理由補充": o.entry_reason_detail, "委託股數": o.requested_shares,
                    "預設停損價": o.planned_stop_loss, "預設目標價": o.planned_target,
                    "狀態": o.status,
                }
                for o in sorted(orders, key=lambda o: o.signal_date or date.min, reverse=True)
            ]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("目前沒有委託紀錄。")

# ------------------------------------------------------------------
# 成交 Fills
# ------------------------------------------------------------------
with tab_fills:
    with st.expander("➕ 新增成交", expanded=False):
        pending_orders = [o for o in store.list_orders(account_id=filter_account_id) if o.status == "pending"]
        with st.form("add_fill_form", clear_on_submit=True):
            linked_order_id = st.selectbox(
                "關聯委託（可不選，直接記成交）", [""] + [o.id for o in pending_orders],
                format_func=lambda i: "" if i == "" else next(
                    f"{o.account_id} {o.code} {o.side} {o.signal_date}" for o in pending_orders if o.id == i
                ),
            )
            linked_order = next((o for o in pending_orders if o.id == linked_order_id), None)

            col1, col2, col3 = st.columns(3)
            with col1:
                f_account = st.selectbox(
                    "帳戶", account_ids,
                    index=account_ids.index(linked_order.account_id) if linked_order else 0,
                )
                f_code = st.text_input("股票代號", value=linked_order.code if linked_order else "")
                f_side = st.radio("買或賣", ["買", "賣"], horizontal=True,
                                   index=0 if not linked_order or linked_order.side == "買" else 1)
            with col2:
                f_price = st.number_input("成交價", min_value=0.0, step=0.01)
                f_shares = st.number_input(
                    "成交股數（支援零股）", min_value=0, step=1,
                    value=linked_order.requested_shares if linked_order else 0,
                )
                f_fill_date = st.date_input("成交日期", value=date.today())
            with col3:
                f_is_etf = st.checkbox("是ETF")
                f_is_day_trade = st.checkbox("當沖（買賣同一天）")
                f_source = st.selectbox(
                    "來源", ["manual", "system"],
                    index=0 if not linked_order or linked_order.source == "manual" else 1,
                )

            f_stop_loss = st.number_input(
                "預設停損價（買方才有意義，可不填）", min_value=0.0, step=0.01,
                value=(linked_order.planned_stop_loss or 0.0) if linked_order else 0.0,
            )
            f_exit_reason = st.text_input("出場理由（賣方才需要，可不填）")

            if f_price > 0 and f_shares > 0:
                preview_fee = calc_fee(f_price, f_shares)
                preview_tax = calc_sell_tax(f_price, f_shares, f_is_etf, f_is_day_trade) if f_side == "賣" else 0.0
                st.caption(f"預覽：手續費 {preview_fee:,.0f} 元"
                           + (f"、證交稅 {preview_tax:,.0f} 元" if f_side == "賣" else ""))

            if st.form_submit_button("新增成交"):
                if not f_code or f_shares <= 0 or f_price <= 0:
                    st.error("請填代號、成交價、成交股數")
                else:
                    fee = calc_fee(f_price, f_shares)
                    tax = calc_sell_tax(f_price, f_shares, f_is_etf, f_is_day_trade) if f_side == "賣" else 0.0
                    store.add_fill(Fill(
                        account_id=f_account, code=f_code.strip(), side=f_side, price=f_price,
                        shares=int(f_shares), fill_date=f_fill_date, fee=fee, tax=tax,
                        is_day_trade=f_is_day_trade, source=f_source,
                        planned_stop_loss=f_stop_loss or None, exit_reason=f_exit_reason,
                        order_id=linked_order_id or "",
                    ))
                    if linked_order_id:
                        store.update_order_status(linked_order_id, "filled")
                    st.success("已新增成交")
                    st.rerun()

    fills = store.list_fills(account_id=filter_account_id)
    st.subheader(f"成交紀錄（{len(fills)} 筆）")
    if fills:
        st.dataframe(
            pd.DataFrame([
                {
                    "id": f.id, "帳戶": f.account_id, "成交日期": f.fill_date, "代號": f.code,
                    "買賣": f.side, "成交價": f.price, "股數": f.shares,
                    "張數": round(f.shares / 1000, 3), "手續費": f.fee, "證交稅": f.tax,
                    "當沖": f.is_day_trade, "來源": f.source, "預設停損價": f.planned_stop_loss,
                    "出場理由": f.exit_reason,
                }
                for f in sorted(fills, key=lambda f: f.fill_date, reverse=True)
            ]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("目前沒有成交紀錄。")

# ------------------------------------------------------------------
# 持倉 Positions（派生快照，不能在這裡編輯）
# ------------------------------------------------------------------
with tab_positions:
    positions = store.list_positions(account_id=filter_account_id)
    st.subheader(f"目前持倉（{len(positions)} 檔，由成交紀錄用FIFO算出，不能直接編輯）")
    if positions:
        st.dataframe(
            pd.DataFrame([
                {"帳戶": p.account_id, "代號": p.code, "股數": p.shares,
                 "張數": round(p.shares / 1000, 3), "平均成本": round(p.avg_cost, 2)}
                for p in positions
            ]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("目前沒有未平倉的持倉。")
