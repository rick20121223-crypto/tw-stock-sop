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

import institutional_ranking as ir
from sop_decision import classify_final, combine_timeframes, evaluate_timeframe
from stock_core import get_intraday_data, run_all_indicators, unique_watchlist

st.set_page_config(page_title="短線進場", layout="wide", page_icon="⚡")


def _resolve_theme() -> str:
    """回傳 Streamlit 目前實際套用的主題（'light'／'dark'）。偵測不到（舊版
    Streamlit、或還沒有真正的前端連線）一律當 light，不假設使用者在深色模式。"""
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None
    return theme_type if theme_type in ("light", "dark") else "light"


# 台股慣例：紅漲綠跌（跟美股相反）。淺色/深色主題各自準備一套通過 WCAG AA
# （文字對底色 ≥4.5:1）的顏色組合——不是同一份色票套兩種主題再算了事，
# 深色主題預設是「跟隨系統」，手機本來就常是深色，兩套都要能看清楚。
_LIGHT_BUCKET_STYLE = {
    "加碼":     {"color": "#c62828", "bg": "#fdecea", "name": "#222222", "meta": "#555555"},
    "買進":     {"color": "#c62828", "bg": "#fdecea", "name": "#222222", "meta": "#555555"},
    "觀望":     {"color": "#616161", "bg": "#f5f5f5", "name": "#222222", "meta": "#555555"},
    "賣出減碼": {"color": "#2e7d32", "bg": "#eaf6ec", "name": "#222222", "meta": "#555555"},
}
_DARK_BUCKET_STYLE = {
    "加碼":     {"color": "#ff6659", "bg": "#3a1f1f", "name": "#e8e8ea", "meta": "#b3b3b8"},
    "買進":     {"color": "#ff6659", "bg": "#3a1f1f", "name": "#e8e8ea", "meta": "#b3b3b8"},
    "觀望":     {"color": "#b3b3b8", "bg": "#232326", "name": "#e8e8ea", "meta": "#b3b3b8"},
    "賣出減碼": {"color": "#66bb6a", "bg": "#14241a", "name": "#e8e8ea", "meta": "#b3b3b8"},
}
BUCKET_STYLE = _DARK_BUCKET_STYLE if _resolve_theme() == "dark" else _LIGHT_BUCKET_STYLE

BUCKET_ORDER = {"加碼": 0, "買進": 1, "觀望": 2, "賣出減碼": 3}
# 跟 home.py 既有的 BUCKET_EMOJI 是同一套慣例：買賣結論不能只靠顏色傳達
# （色盲、黑白列印、螢幕閱讀器都需要這個非色彩線索）。
BUCKET_EMOJI = {"加碼": "🔺🔺", "買進": "🔺", "觀望": "⚪", "賣出減碼": "🔻"}


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
    return [(name, code) for name, code, market in unique_watchlist() if market == "TW"]


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


def _stock_card(row, badge: str = "") -> str:
    """單張股票卡片。`badge` 給「本週法人排行」區塊標註🔄來源用，固定清單不傳。"""
    style = BUCKET_STYLE[row["分類"]]
    price = f"{row['收盤']:.2f}" if pd.notna(row["收盤"]) else "—"
    icon = BUCKET_EMOJI[row["分類"]]
    badge_html = f'<span style="font-size:11px; color:{style["meta"]};">{badge}</span> ' if badge else ""
    # aria-label：整張卡是純裝飾用 div，screen reader 預設會把它當成一串沒有
    # 邊界的文字唸出來，補上 role/aria-label 才聽得出「這是一項、內容是什麼」。
    aria_label = f"{row['名稱']}（{row['代碼']}）：{row['最終建議']}"
    return f"""\
<div role="listitem" aria-label="{aria_label}"
     style="border-left:1px solid {style['color']}; background:{style['bg']};
            border-radius:6px; padding:10px 12px; height:100%;">
  <div style="font-weight:600; font-size:14px; color:{style['name']};">{badge_html}{row['名稱']}（{row['代碼']}）</div>
  <div style="font-size:17px; font-weight:700; color:{style['color']}; margin:3px 0;">
    {icon} {row['最終建議']}
  </div>
  <div style="font-size:14px; color:{style['meta']};">收盤 {price} ・ 60分：{row['60分']} ・ 5分：{row['5分']}</div>
</div>
"""


cards_html = "".join(_stock_card(row) for _, row in df_overview.iterrows())
st.markdown(
    f'<div role="list" style="display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); '
    f'gap:10px;">{cards_html}</div>',
    unsafe_allow_html=True,
)

if errors:
    with st.expander(f"⚠️ {len(errors)} 檔資料取得失敗"):
        for e in errors:
            st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")

st.caption("👉 SLS（美股）不支援60分/5分，未列入此頁。想看週+日的長線留倉判斷，請切到左側選單「長線留倉」首頁。")

# ------------------------------------------------------------------
# 🔄 本週法人排行：獨立區塊，跟上面的固定清單分開顯示。每週一由排程自動
# 算出「上週三大法人買賣超前十大（買超+賣超）」覆寫這份名單，這裡一律
# 是台股個股，不用再篩市場。
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(analyze_short, name, code, fugle_api_key)
            for name, code, _market, _side in rotating_list
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            (rotating_rows if result["狀態"] == "ok" else rotating_errors).append(result)

    if rotating_rows:
        rotating_cards_html = "".join(
            _stock_card(row, badge=f"🔄 法人{side_map.get(row['代碼'], '')}")
            for row in rotating_rows
        )
        st.markdown(
            f'<div role="list" style="display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); '
            f'gap:10px;">{rotating_cards_html}</div>',
            unsafe_allow_html=True,
        )

    if rotating_errors:
        with st.expander(f"⚠️ {len(rotating_errors)} 檔本週法人排行股票資料取得失敗"):
            for e in rotating_errors:
                st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")
