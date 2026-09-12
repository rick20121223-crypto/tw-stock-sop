"""
📅 長線留倉 — 首頁
一打開網站就能看到整份清單「週線＋日線」整合後的 SOP 結論，回答的問題是
「哪些股票的中長期結構值得留倉/加碼，哪些該減碼」。想看「現在適不適合
短打進場」，請切到左側選單「短線進場」頁面（用60分/5分判讀）。

60分/5分沒有放進這份總覽：那需要額外的 Fugle API 呼叫、且只支援台股
個股/ETF，整份清單一起跑會太慢，所以維持「點進單一股票才查」。
但週+日已經是 SOP 裡份量最重的長天期結構判斷，多數情況下已經足夠先篩出
「哪些值得再點進去細看」。

執行方式：
    streamlit run streamlit_app.py

安裝套件：
    pip3 install streamlit plotly requests pandas fugle-marketdata --user
    (Mac 若出現 --break-system-packages 相關訊息，把 --user 拿掉改用 --break-system-packages 即可)

API 金鑰設定（部署到 Streamlit Community Cloud 時）：
    在 App 的 Settings → Secrets 貼入：
        FINMIND_TOKEN = "你的 FinMind token"
        FUGLE_API_KEY = "你的 Fugle 行情 API Key"
    本機執行時若沒有設定 secrets.toml，會改用側邊欄手動輸入。
"""

import concurrent.futures
from datetime import date

import streamlit as st

import institutional_ranking as ir
from multi_timeframe_check import full_check
from sop_decision import classify_final
from stock_core import unique_watchlist

st.set_page_config(page_title="長線留倉", layout="wide", page_icon="📅")

# 台股慣例：紅漲綠跌（跟美股相反）
UP_COLOR = "#e53935"
DOWN_COLOR = "#43a047"
FLAT_COLOR = "#9e9e9e"

BUCKET_ORDER = {"加碼": 0, "買進": 1, "觀望": 2, "賣出減碼": 3}
BUCKET_EMOJI = {"加碼": "🔺🔺", "買進": "🔺", "觀望": "⚪", "賣出減碼": "🔻"}
BUCKET_COLOR = {"加碼": UP_COLOR, "買進": UP_COLOR, "觀望": FLAT_COLOR, "賣出減碼": DOWN_COLOR}


def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


with st.sidebar:
    with st.expander("🔑 API 金鑰狀態"):
        secret_token = _get_secret("FINMIND_TOKEN")
        if secret_token:
            st.success("已使用雲端 Secrets 的 FinMind Token")
            api_token = secret_token
        else:
            api_token = st.text_input("FinMind API Token", type="password")

    years_back = st.slider("週線回溯年數（需夠長才能算出35週生死線）", 1, 5, 2)
    refresh = st.button("🔄 重新整理資料", use_container_width=True)

st.title("📅 長線留倉")
st.caption(
    "回答「哪些股票的中長期結構值得留倉/加碼、哪些該減碼」。依但丁老師"
    "完整 SOP，整合「週線＋日線」的結構判讀給出買進/加碼/觀望/賣出減碼"
    "結論。想看「現在適不適合短打進場」，請切到左側選單「短線進場」頁面"
    "（60分/5分）。對哪檔有興趣，按「查看完整分析」直接跳過去看日/週/60分"
    "/5分的完整明細。僅供輔助判讀，不構成投資建議。"
)

if not api_token:
    st.info("未設定 FinMind API Token，將使用免費額度（較容易觸發流量限制）。建議在左側輸入 Token，或部署時於 Secrets 設定 FINMIND_TOKEN。")

if refresh:
    st.cache_data.clear()


@st.cache_data(ttl=900, show_spinner=False)
def analyze_stock(name: str, code: str, market: str, token: str, start_date: str):
    try:
        # 首頁只做「週＋日」整合，不帶 Fugle Key，避免整份清單刷新太久。
        result = full_check(code, market, token, "", start_date)
        detail = result["各週期明細"]
        return {
            "名稱": name, "代碼": code, "market": market, "狀態": "ok",
            "收盤": result.get("收盤"),
            "最終建議": result["最終建議"],
            "週線": detail.get("週", {}).get("結論", "—"),
            "日線": detail.get("日", {}).get("結論", "—"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"名稱": name, "代碼": code, "market": market, "狀態": "error", "訊息": str(exc)}


start_date = str(date(date.today().year - years_back, date.today().month, date.today().day))

with st.spinner("正在平行分析清單內所有股票的週+日整合結論（首次載入較久，之後15分鐘內會用快取）..."):
    rows, errors = [], []
    watchlist = unique_watchlist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(analyze_stock, name, code, market, api_token, start_date)
            for name, code, market in watchlist
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            (rows if result["狀態"] == "ok" else errors).append(result)

if not rows:
    st.error("所有股票都取得失敗，請確認 FinMind Token 是否正確。以下是實際錯誤原因：")
    for e in errors:
        st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")
    st.stop()

for r in rows:
    r["分類"] = classify_final(r["最終建議"])
    r["排序"] = BUCKET_ORDER[r["分類"]]

rows.sort(key=lambda r: r["排序"])

buy_count = sum(1 for r in rows if r["分類"] in ("買進", "加碼"))
sell_count = sum(1 for r in rows if r["分類"] == "賣出減碼")
watch_count = sum(1 for r in rows if r["分類"] == "觀望")

c1, c2, c3 = st.columns(3)
c1.metric("🔺 買進／加碼", f"{buy_count} 檔")
c2.metric("🔻 賣出減碼", f"{sell_count} 檔")
c3.metric("⚪ 觀望", f"{watch_count} 檔")

st.divider()

header = st.columns([2, 1, 1, 3, 2, 2, 1.2])
for col, text in zip(header, ["名稱", "代碼", "收盤", "最終建議（週+日整合）", "週線", "日線", ""]):
    col.markdown(f"**{text}**")

for r in rows:
    c = st.columns([2, 1, 1, 3, 2, 2, 1.2])
    c[0].write(r["名稱"])
    c[1].write(r["代碼"])
    c[2].write(f"{r['收盤']:.2f}" if r["收盤"] is not None else "—")
    color = BUCKET_COLOR[r["分類"]]
    emoji = BUCKET_EMOJI[r["分類"]]
    c[3].markdown(f"<span style='color:{color}; font-weight:600'>{emoji} {r['最終建議']}</span>",
                   unsafe_allow_html=True)
    c[4].write(r["週線"])
    c[5].write(r["日線"])
    if c[6].button("查看", key=f"view_{r['代碼']}_{r['market']}", use_container_width=True):
        st.session_state["prefill_stock"] = {
            "label": r["名稱"], "stock_id": r["代碼"], "market": r["market"],
        }
        st.switch_page("pages/3_多週期整合分析.py")

if errors:
    with st.expander(f"⚠️ {len(errors)} 檔資料取得失敗"):
        for e in errors:
            st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")

st.caption("👉 想一次看整份清單的短線進場訊號，請到左側選單「短線進場」頁面；"
           "想查清單外的股票，請到「多週期整合分析」頁面選「自訂代碼」。")

# ------------------------------------------------------------------
# 🔄 本週法人排行：獨立區塊，跟上面的固定清單分開顯示。每週一由排程自動
# 算出「上週三大法人買賣超前十大（買超+賣超）」覆寫這份名單，這裡只是
# 讀取現有結果來顯示，不會在網頁瀏覽時重新計算排行本身。
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

    with st.spinner("正在分析本週法人排行股票的週+日整合結論..."):
        rotating_rows, rotating_errors = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(analyze_stock, name, code, market, api_token, start_date)
                for name, code, market, _side in rotating_list
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                (rotating_rows if result["狀態"] == "ok" else rotating_errors).append(result)

    if rotating_rows:
        for r in rotating_rows:
            r["法人方向"] = side_map.get(r["代碼"], "—")
            r["分類"] = classify_final(r["最終建議"])
        rotating_rows.sort(key=lambda r: (0 if r["法人方向"] == "買超" else 1, BUCKET_ORDER[r["分類"]]))

        header2 = st.columns([2, 1, 1, 3, 1.3, 2, 2])
        for col, text in zip(header2, ["名稱", "代碼", "收盤", "最終建議（週+日整合）", "法人方向", "週線", "日線"]):
            col.markdown(f"**{text}**")
        for r in rotating_rows:
            c = st.columns([2, 1, 1, 3, 1.3, 2, 2])
            c[0].write(r["名稱"])
            c[1].write(r["代碼"])
            c[2].write(f"{r['收盤']:.2f}" if r["收盤"] is not None else "—")
            color = BUCKET_COLOR[r["分類"]]
            emoji = BUCKET_EMOJI[r["分類"]]
            c[3].markdown(f"<span style='color:{color}; font-weight:600'>{emoji} {r['最終建議']}</span>",
                           unsafe_allow_html=True)
            c[4].write(f"{'🟥' if r['法人方向']=='買超' else '🟩'} {r['法人方向']}")
            c[5].write(r["週線"])
            c[6].write(r["日線"])

    if rotating_errors:
        with st.expander(f"⚠️ {len(rotating_errors)} 檔本週法人排行股票資料取得失敗"):
            for e in rotating_errors:
                st.write(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")
