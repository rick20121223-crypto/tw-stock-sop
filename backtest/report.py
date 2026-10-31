"""
第三輪驗證報告：6830 DIF校驗釐清、W底W0~W4規則測試、大盤環境控制。

用法：python3 -m backtest.report
"""
import pickle
from pathlib import Path

import pandas as pd

from backtest.summarize import summarize_w_bottom

OUTPUT_DIR = Path(__file__).parent / "output"
REPORT_FILE = OUTPUT_DIR / "report.md"
ROUND3_W_CACHE = OUTPUT_DIR / "round3_w_cache"

SPLIT_DATE = pd.Timestamp("2025-04-15")
LOW_SAMPLE_THRESHOLD = 30

W_SIGNALS = ["W0_型態本身", "W1_斜率上揚且MACD零軸上", "W2_剛金叉(完整規則)",
             "W3_持續K大於D非剛交叉", "W4_K小於D尚未金叉"]


def df_to_md(df: pd.DataFrame, flag_col: str = None, label_col: str = None) -> str:
    if df.empty:
        return "_（無資料）_\n"
    df = df.copy()
    if flag_col and label_col and flag_col in df.columns and label_col in df.columns:
        low = df[flag_col] < LOW_SAMPLE_THRESHOLD
        df.loc[low, label_col] = df.loc[low, label_col].astype(str) + f" ⚠️樣本數<{LOW_SAMPLE_THRESHOLD}"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    col_arrays = {c: df[c].to_numpy() for c in cols}
    for i in range(len(df)):
        cells = []
        for c in cols:
            v = col_arrays[c][i]
            cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_report() -> str:
    calib = pd.read_csv(OUTPUT_DIR / "step1_calibration.csv")
    w_raw = pd.read_csv(OUTPUT_DIR / "w_bottom_events.csv", dtype={"code": str})
    w = w_raw[~w_raw["near_unadjusted_gap"]].copy()
    w["date"] = pd.to_datetime(w["date"])
    w_first, w_second = w[w["date"] < SPLIT_DATE], w[w["date"] >= SPLIT_DATE]

    sample10 = pd.read_csv(OUTPUT_DIR / "w_bottom_sample10.csv")

    md = []
    md.append("# 指標搭配回測報告（第三輪驗證）：6830 DIF釐清、W底規則、大盤環境控制\n")
    md.append(
        "沿用同一批108檔股票、3年還原權息日線、2025-04-15為前後兩段切分點。"
        "本輪新增W底型態偵測模組(backtest/w_bottom.py)與TAIEX大盤環境模組"
        "(backtest/market_regime.py)，皆為獨立模組，未修改sop_rules.yaml或"
        "既有判讀邏輯。\n"
    )

    # ================= 項目1 =================
    md.append("## 1. 項目1：6830 DIF校驗釐清\n")
    md.append(
        "上一輪報告的說法確實不夠精確，造成表面上的矛盾，這裡重新完整列出、"
        "逐項澄清：\n"
    )
    md.append(df_to_md(calib))
    md.append(
        "\n**精確結論（不是上一輪籠統講的「兩檔所有查核日誤差<0.03%」）**：\n"
        "- 13筆查核值中，只有2筆未通過（誤差>1%門檻）：6830在2026-04-23的DIF"
        "（誤差2.766%）、6830在2026-05-06的DIF（誤差1.791%）。\n"
        "- 6830在這兩天的其餘指標（MACD訊號線、36K、36D、BBI）**全數通過**，"
        "但誤差並不是一致地<0.03%：36K/36D/BBI確實都在0.000%~0.004%這個"
        "極小範圍；MACD訊號線的誤差明顯大一些，是0.527%(04-23)/0.799%(05-06)，"
        "仍在1%門檻內但不是「幾乎零誤差」。上一輪把這兩種誤差水準混著講成"
        "一句「誤差多在0.00%~0.03%」，才會讓人誤以為DIF也該一樣準，這是"
        "上一輪報告的表達問題，不是數字本身矛盾。\n"
        "- 2308在2026-10-13的全部4個指標（DIF/MACD訊號線/36K/36D）都通過，"
        "誤差0.000%~0.016%。\n\n"
        "**新增測試：還原權息 vs 未還原，對MACD死叉判定日期的影響**\n"
        "直接比較6830全部3年歷史，用「未還原」與「還原權息」兩種收盤價序列"
        "各自算出來的MACD死叉/金叉日期清單：兩者在絕大多數日期完全一致，"
        "只有4處不同，而且全部發生在6830自己的除權息日附近（2024-07-11/12、"
        "2024-07-23/26、2024-08-19/20、2025-08-20/21），每一處都剛好差"
        "**1個交易日**。本次有爭議的2026-04-23、2026-05-06附近完全沒有出現"
        "在這4處差異清單裡——換句話說，不管用未還原或還原權息口徑，這兩個"
        "查核日附近的MACD死叉判定日期都是一樣的，DIF數值上的誤差不會讓"
        "回測實際用的訊號日期跟著偏移。\n\n"
        "**還未排除的問題，具體範圍**：只剩「6830在2026-04-23/05-06這兩天"
        "的DIF數值」本身為什麼跟券商顯示值差1.8%~2.8%，原因只能縮小到"
        "「這檔股票在這段期間（04-23當天盤中從946高點殺到803收盤，單日"
        "跌幅逾15%）的資料供應商之間可能存在細微的收盤價記錄差異」，"
        "沒有拿到券商原始逐筆成交資料無法100%證實。但已經確認：(a)不是"
        "公式寫錯（2308完全吻合、6830的KD/BBI/MACD訊號線也吻合）、"
        "(b)不影響回測實際使用的MACD死叉判定日期（上面的新測試）。\n\n"
        "**判斷**：這是一個範圍侷限（僅2筆查核值、僅DIF欄位、僅6830一檔"
        "股票的2個特定日期）、已排除公式錯誤、已確認不會傳播到回測訊號"
        "日期的殘留疑點。依照使用者規格「先修正或說明原因，再繼續」的"
        "要求，這裡選擇後者——以上的說明與影響評估取代了「數值上修正」"
        "（沒辦法在不知道券商原始資料的情況下強行修正DIF數值去貼合），"
        "繼續往下做項目2、3；若使用者認為這個解釋不夠充分，請直接告知，"
        "我會停下來進一步追查。\n"
    )

    # ================= 項目2 =================
    md.append("\n## 2. 項目2：W底 W0~W4 比較\n")
    md.append(
        "W0=型態本身；W1=W0+均線(20MA)斜率上揚+MACD在零軸上；W2=W1+KD365近3"
        "個交易日內剛金叉（老師的完整規則）；W3=W1+KD365持續K>D但不是剛"
        "交叉（應降級但K已經>D的情況）；W4=W1+KD365 K<D（尚未金叉）。"
        "W2/W3/W4互斥（同一個W1訊號依KD365狀態三選一）。\n"
    )
    w_full = summarize_w_bottom(w)
    w_full_first = summarize_w_bottom(w_first)
    w_full_second = summarize_w_bottom(w_second)
    md.append("### 2.1 全期間\n")
    md.append(df_to_md(w_full, flag_col="樣本數", label_col="signal"))
    md.append("\n### 2.2 分期間穩健性\n")
    md.append("**前半段**：\n")
    md.append(df_to_md(w_full_first, flag_col="樣本數", label_col="signal"))
    md.append("\n**後半段**：\n")
    md.append(df_to_md(w_full_second, flag_col="樣本數", label_col="signal"))
    md.append(
        "\n**重點回答：W2是否真的比W3、W4表現更好？**\n"
        "**沒有**。全期間、以及前後兩段，W2(完整規則)的20日平均報酬、中位數"
        "報酬、勝率都沒有系統性高於W3；W4(尚未金叉，照老師規則本該最不"
        "可信)在全期間甚至是20日報酬最高、勝率最高的一組。但分期間看更"
        "關鍵：前半段W4表現明顯最差（勝率35.9%、假突破率84.6%最高），"
        "方向上支持老師規則；後半段W4卻反過來表現最好（勝率73.8%）。"
        "這個反轉在第3節的大盤環境控制後會進一步檢視是不是單純被後半段"
        "的多頭環境放大。**KD365「剛交叉」這個限制，從目前資料看不出"
        "穩定地有「過濾摸底過早」的效果**——W2跟W3的假突破率（59.5% vs "
        "57.9%全期間）差異很小，不是W2明顯更低。\n"
    )

    md.append("\n### 2.3 參數敏感度\n")
    variant_rows = []
    for label, desc in [
        ("baseline_gap10-40_close_n3_ma20", "baseline(右底10~40/頸線收盤/N=3/20MA)"),
        ("gap5-60", "右底間隔5~60"), ("neckline_high", "頸線用最高點"),
        ("n5", "N=5"), ("n8", "N=8"), ("ma35", "均線用35MA"),
    ]:
        cache_file = ROUND3_W_CACHE / f"{label}.pkl"
        if not cache_file.exists():
            continue
        with open(cache_file, "rb") as f:
            w_variant = pickle.load(f)["w"]
        w_variant = w_variant[~w_variant["near_unadjusted_gap"]]
        s = summarize_w_bottom(w_variant)
        present = [x for x in W_SIGNALS if x in set(s["signal"])]
        s = s.set_index("signal").loc[present].reset_index()
        for _, r in s.iterrows():
            variant_rows.append({"變體": desc, "訊號": r["signal"], "樣本數": r["樣本數"],
                                  "假突破率%": r["假突破率%(10日內跌破頸線)"],
                                  "20日報酬%": r["20日_平均報酬%"], "20日勝率%": r["20日_勝率%(正報酬)"]})
    variant_df = pd.DataFrame(variant_rows)
    md.append(df_to_md(variant_df))
    md.append(
        "\n**敏感度結論**：右底間隔放寬(5~60)、頸線改用最高點、N改成5/8、"
        "均線改用35MA，6種組合下W2始終沒有穩定地優於W3/W4——这個「KD365"
        "剛交叉沒有看出明顯額外效益」的方向在所有測試參數下一致。\n"
    )

    md.append("\n### 2.4 10個W底樣本（供人工核對）\n")
    md.append(
        "隨機抽樣（非精選），neckline_price/signal_close欄位是**還原權息後**的價格，跟券商App"
        "看到的原始股價可能有小幅落差（尤其除權息次數多的股票），核對"
        "左底/右底/頸線的「相對位置、時間間隔」會比核對絕對價位更準確。\n"
    )
    md.append(df_to_md(sample10))

    # ================= 項目3：大盤環境控制（真正的TAIEX） =================
    md.append("\n## 3. 項目3：大盤環境控制（TWSE限制解除後，改用真正的TAIEX）\n")
    md.append(
        "**更新**：TWSE加權指數端點、FinMind的IP限制後來都解除了"
        "（第一次是因為一次性併發打太多次觸發bot防護）。這次改成"
        "**序列化+固定0.6秒節流**重新抓取3年TAIEX歷史，第一輪抓到"
        "660天就又卡住一段很長的缺口(2025-10~2026-04近半年幾乎全空)，"
        "重跑一次補齊缺口後拿到888個交易日，只剩4個6~12天的小缺口"
        "（對照農曆年/國定連假，屬正常現象），資料品質足夠支撐這節"
        "60日均線/60日報酬的計算。以下全部用**真正的TAIEX**（發行量"
        "加權股價指數，價格指數版），不是上一版報告用的0050代理指標。\n\n"
        "**0050代理指標回頭驗證**：拿真正TAIEX跟上一版用的0050代理"
        "指標比較，兩者收盤價相關係數**0.9988**，regime_ma60分類"
        "一致率**95.9%**、regime_ret60一致率**96.8%**——0050當時"
        "是個相當準確的代理，上一版報告用0050算出來的方向性結論"
        "（問題1/2的答案）在這裡用真正TAIEX重算後**沒有改變**，細節"
        "數字略有差異但不影響任何結論，詳見r3_item3_taiex_vs_0050.csv。\n"
    )

    buy_period_regime = pd.read_csv(OUTPUT_DIR / "r3_item3_taiex_buy_period_regime.csv")
    w_period_regime = pd.read_csv(OUTPUT_DIR / "r3_item3_taiex_w_period_regime.csv")
    regime_dist = pd.read_csv(OUTPUT_DIR / "r3_item3_taiex_regime_distribution.csv")

    md.append("\n### 3.1 大盤環境分布（regime_ma60：TAIEX站上且60MA上揚=多頭）\n")
    md.append(df_to_md(regime_dist))
    md.append(
        "\n前半段55.4%的交易日是多頭（regime_ret60分類是66.2%），"
        "後半段多頭比例跳到82.7%（regime_ret60是85.9%)——後半段大盤"
        "環境確實系統性地比前半段偏多頭，這是前後兩段表現落差的其中"
        "一個真實成因（但不是全部，見下方3.2）。\n"
    )

    md.append("\n### 3.2 B1~B4、W0~W4 依「期間 x 大盤環境」交叉拆分（20日報酬/超額報酬）\n")
    md.append(
        "超額報酬 = 個股20日報酬 − 同期間TAIEX 20日報酬。"
        "這張表把「期間」跟「大盤環境」交叉拆開，用來回答「前後兩段的"
        "落差，是不是只因為後半段比較常處於多頭」——如果真的只是這個"
        "原因，同一個環境格裡前後兩段的數字應該會很接近。\n"
    )
    md.append("**買進側(B1~B4)**：\n")
    md.append(df_to_md(buy_period_regime))
    md.append("\n**W底(W0~W4)**：\n")
    md.append(df_to_md(w_period_regime, flag_col="樣本數", label_col="訊號"))
    md.append(
        "\n**回答問題1「前後兩段的勝率差，有多少可以用大盤環境解釋？」**："
        "**部分解釋，不是全部**。以B2為例：原始20日勝率前半46.4%、後半"
        "64.6%，差18.2個百分點；拿掉大盤環境影響後看超額勝率，前半"
        "42.0%、後半50.8%，差距縮小到約8.8個百分點——降幅約52%，表示"
        "一半左右的前後段落差可以用「後半段大盤環境更多頭」解釋，但"
        "**剩下的一半左右不能**，即使同樣被歸類為「多頭」，後半段的多頭"
        "期間表現還是明顯優於前半段的多頭期間（例如B2「多頭」這個環境"
        "格本身，前半段20日報酬只有0.98%，後半段卻有9.55%）——這代表"
        "regime_ma60這種二元分類（多頭/非多頭）太粗，沒辦法完全捕捉"
        "「後半段的多頭力道比前半段強很多」這件事，**這次是用真正的"
        "TAIEX算出來的，不是0050代理指標的侷限，是二元分類方法本身的"
        "侷限**。\n"
    )

    md.append("\n### 3.3 只依大盤環境拆分（不分期間）\n")
    regime_only_buy = pd.read_csv(OUTPUT_DIR / "r3_item3_taiex_regime_only_buy.csv")
    regime_only_w = pd.read_csv(OUTPUT_DIR / "r3_item3_taiex_regime_only_w.csv")
    md.append("**買進側**：\n")
    md.append(df_to_md(regime_only_buy))
    md.append("\n**W底**：\n")
    md.append(df_to_md(regime_only_w, flag_col="樣本數", label_col="訊號"))
    md.append(
        "\n**回答問題2「控制大盤後，B4與W2相對其他組別還有沒有差異？」**："
        "**沒有，而且方向跟原本一樣**。只看「多頭」這個環境格內部"
        "（樣本數最多、最穩健）：B4(超額報酬3.16%、超額勝率48.3%)不比"
        "B1(3.54%、49.4%)突出，四個買進訊號在同一個環境格內部差異都"
        "很小；W2(超額報酬4.41%、超額勝率50.4%)依然落後W3(4.87%、"
        "52.1%)、W4(5.80%、56.4%)——這個排名跟"
        "第2節「不控制環境」時看到的一樣，等於**控制大盤環境之後，"
        "W2沒有額外優勢這個結論被進一步印證**，不是被推翻或被解釋掉。\n"
    )

    # ================= 第4節：結論（三個項目都涵蓋） =================
    md.append("\n## 4. 結論：穩定 vs 不穩定的發現\n")
    md.append(
        "**穩定**：\n"
        "1. 6830的DIF校驗誤差（2.766%/1.791%）已確認範圍侷限，且不影響"
        "回測實際使用的MACD死叉判定日期（還原權息 vs 未還原在這兩天附近"
        "完全一致）。\n"
        "2. **W2(KD365剛金叉的完整規則)沒有穩定地優於W3(持續K>D非剛交叉)"
        "或W4(K<D尚未金叉)**——這個方向在6種型態參數變體下一致，"
        "**控制大盤環境後依然一致、沒有改變**（見3.3）。\n"
        "3. **B4(三者同步)相對B1~B3沒有看出額外優勢**，控制大盤環境後"
        "同樣一致（見3.3）。\n"
        "4. 前後兩段期間的表現落差，**大盤環境能解釋一部分（約一半，"
        "以B2為例）但不是全部**——這個「部分解釋」的結論本身是穩定的"
        "（買進側、W底側都看到類似模式）。\n\n"
        "**不穩定 / 尚無法完全解釋**：\n"
        "1. 即使同樣歸類為「多頭」環境，後半段的多頭期間表現仍系統性"
        "優於前半段的多頭期間——這是用**真正TAIEX**算出來的，所以已經"
        "排除是0050代理指標的誤差，確認是regime_ma60這種二元分類"
        "（多頭/非多頭）本身的方法侷限：真正的原因可能是多頭力道強弱"
        "的連續性差異（不是簡單二分法能處理的），**本輪未進一步確認**。\n"
    )

    # ================= 第5節：限制 =================
    md.append("\n## 5. 限制\n")
    md.append(
        "- 項目3改用真正的TAIEX後，已經驗證0050代理指標在這3年期間"
        "跟真正TAIEX高度一致（價格相關係數0.9988、regime分類一致率"
        "95.9%~96.8%），上一版報告用0050算出來的方向性結論沒有改變。\n"
        "- regime_ma60/regime_ret60都是**二元**分類（多頭/非多頭），"
        "3.2節已經看到這種二分法無法完全捕捉「多頭力道強弱」的連續性"
        "差異，是本輪一個已知但未解決的方法侷限。\n"
        "- **W底偵測是演算法近似，不等於人工看圖判定**，已知會不一致的"
        "地方（詳見2.4樣本與w_bottom.py模組docstring）：(a)區域低點用"
        "收盤價判斷，不是最低價；(b)每個左底只取最早一個符合條件的右底，"
        "不會窮舉所有候選；(c)突破後只要回測期間內破過頸線就整次判定"
        "失敗，不會在同一組左右底之間再找下一次突破機會；(d)左底候選"
        "彼此沒有去重機制，同一段走勢可能被偵測成好幾個略有位移的"
        "「W底」（已用signal_idx去重同一訊號日的重複計算，但左右底本身"
        "的選取仍可能跟人工判讀抓的不是同一對低點）；(e)老師實際判讀"
        "W底可能還會考慮成交量、型態的視覺對稱性等本模組沒有量化的條件。\n"
        "- 2.4節的10個樣本是隨機抽樣，不是我手動挑選「看起來正確」的"
        "案例，核對結果可能發現相當比例的「演算法判定」跟人工判讀不"
        "一致，這是預期中的事，需要你核對後回饋。\n"
        "- 真頭部/W底相關的既有限制（選股偏差、流動性對照組口徑、"
        "還原權息不含減資等）沿用前兩輪報告，本輪未重新列出。\n"
    )

    # ================= 第6節：完成前自我檢查 =================
    md.append("\n## 6. 完成前自我檢查\n")
    md.append(
        "**做的假設**：\n"
        "- W底訊號日定義為「回測不破且收紅K」的第一個符合日，若突破後"
        "retest_window天內曾經跌破頸線就整次判定失敗，不另外搜尋下一次"
        "突破。\n"
        "- W2/W3/W4用KD365的K/D「狀態」（K>D或K<D）搭配「近N日內是否"
        "剛交叉」來三分類，W1本身（均線斜率上揚+MACD零軸上）是W2/W3/W4"
        "共同的必要條件，不滿足W1的W0型態不會被分到任何W2/3/4。\n"
        "- 假突破率定義為「訊號後10日內收盤跌破頸線」，頸線本身在整個"
        "判斷窗口內固定不變（不會因為之後價格創新高而調整頸線位置）。\n"
        "- 超額報酬用「同一天」的TAIEX報酬相減，沒有做beta調整"
        "（例如個股對大盤的敏感度不是1:1這件事沒有處理）。\n"
        "- TAIEX資料本身有4處6~12天的小缺口（對照農曆年/國定連假），"
        "缺口附近的60日均線/60日報酬會吃到一點點不連續，但缺口都不大，"
        "判斷影響有限，沒有逐一排除。\n\n"
        "**無法確認的地方**：\n"
        "- 6830兩筆DIF誤差的根本原因仍未100%確認，只排除到「資料供應商"
        "在極端波動期間的細微差異」這個範圍（詳見第1節）。\n"
        "- 3.2節發現「同一環境格內前後兩段仍有落差」，已確認不是0050"
        "代理指標的問題（這次用真正TAIEX重算，結論一樣），但無法確認"
        "是二元分類的侷限、還是真的存在規模更大的結構性轉變（例如"
        "AI/半導體供應鏈在後半段的一次性重估），本輪未進一步拆解。\n\n"
        "**樣本不足的結果**：本輪W0~W4各表格裡樣本數<30的列都已經用"
        "⚠️標註；3.2節的W底期間x環境交叉表裡，W2/W4在「後半段x非多頭」"
        "這一格樣本數仍偏小（分別15筆、13筆），已標註樣本不足，解讀"
        "時要留意。\n"
    )

    w_full_tagged = w_full.copy(); w_full_tagged.insert(0, "期間", "全期間")
    w_first_tagged = w_full_first.copy(); w_first_tagged.insert(0, "期間", "前半段")
    w_second_tagged = w_full_second.copy(); w_second_tagged.insert(0, "期間", "後半段")

    tables = {
        "r3_item1_calibration": calib,
        "r3_item2_w_comparison": pd.concat([w_full_tagged, w_first_tagged, w_second_tagged], ignore_index=True),
        "r3_item2_sensitivity": variant_df,
        "r3_item2_sample10": sample10,
        "r3_item3_taiex_regime_distribution": regime_dist,
        "r3_item3_taiex_buy_period_regime": buy_period_regime,
        "r3_item3_taiex_w_period_regime": w_period_regime,
        "r3_item3_taiex_regime_only_buy": regime_only_buy,
        "r3_item3_taiex_regime_only_w": regime_only_w,
    }
    return "\n".join(md), tables


if __name__ == "__main__":
    report_text, report_tables = build_report()
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    for name, table in report_tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    print(f"報告已寫入 {REPORT_FILE}（{len(report_text)} 字元），另存 {len(report_tables)} 份CSV")
