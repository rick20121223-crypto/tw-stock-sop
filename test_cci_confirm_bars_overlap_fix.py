"""
修正CCI折返偵測(confirm_bars>1分支)的視窗重疊bug：sop_rules.yaml正式
config裡indicators.cci.lookback_bars跟confirm_bars都是3，但原始程式碼
的「prior」視窗(lookback_bars天，不含今天)跟「last_n」視窗(confirm_bars
天，含今天)在lookback_bars<=confirm_bars時會重疊——具體來說prior=
recent.tail(lookback_bars).iloc[:-1]，跟last_n=df.tail(confirm_bars)
在lookback_bars==confirm_bars==3時完全是同一組位置的子集關係，導致
「曾經站上門檻(was_above，只需prior裡任一根)」跟「現在已經連續N根都
站回門檻內(confirmed_below，要求last_n裡每一根)」在數學上互斥（同一根
K棒不可能同時>=門檻又<門檻），bear_foldback/bull_foldback在這組
config下永遠不可能成立——不是罕見，是數學上不可能，已經實測確認過。

這代表當初用test_short_term_whipsaw_fix_compare.py驗證「confirm_bars=3
比1/2更能降低雜訊反轉」時，candidate(cb=3/3)這組設定下CCI示警可能根本
沒有真正觸發過（完全死掉），降低雜訊反轉的「效果」有可能只是訊號整個
消失、不是真的過濾掉雜訊，這點已經回報給使用者，待他決定要不要重新
用Fugle資料跑一次這支backtest驗證修好以後的confirm_bars=3是否仍是
最佳選擇。

修法：confirm_bars>1分支改用獨立的「touch視窗」——往前推
confirm_bars+lookback_bars根裡，取緊接在confirm視窗前面的lookback_bars
根（不含confirm視窗本身），跟last_n(confirm視窗)完全不重疊，不管
lookback_bars跟confirm_bars的相對大小都成立。confirm_bars<=1的舊版
分支完全不受影響（繼續用原本的recent/prior定義）。

用合成資料，不需要API，執行：
    python3 test_cci_confirm_bars_overlap_fix.py
"""
import pandas as pd

from sop_decision import RULES, _step5b_cci


def _build_cci_df(cci_values):
    n = len(cci_values)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="h"),
        "open": [100] * n, "max": [101] * n, "min": [99] * n,
        "close": [100] * n, "volume": [1000] * n,
        "CCI": cci_values,
    })


def test_bear_foldback_reachable_with_production_default_lookback_equals_confirm():
    """正式config現在是lookback_bars=3, confirm_bars=3——修好之前這個
    組合下bear_foldback數學上不可能成立，這支測試直接驗證修好之後真的
    可以成立（不是只測邊界case，是驗證正式生產config本身真的有效）。"""
    cfg = RULES["indicators"]["cci"]
    assert cfg["lookback_bars"] == cfg["confirm_bars"] == 3, \
        "這支測試假設正式config的lookback_bars/confirm_bars都是3，若已經改了請同步更新測試"

    # touch視窗在confirm視窗「之前」：位置-6,-5,-4曾經站上+100，
    # 位置-3,-2,-1(含今天)連續3根都已折返回到+100以內
    values = [50] * 5 + [120, 120, 120] + [50, 50, 50]
    df = _build_cci_df(values)
    reasons, caveats = [], []
    cci_exit_alert, cci_bias = _step5b_cci(df, "60分", df.iloc[-1], reasons, caveats)
    assert cci_exit_alert is True, "正式config下，touch視窗在confirm視窗之前時，bear_foldback應該可以成立"
    assert cci_bias == -1, f"熊市折返應給cci_bias=-1，實際={cci_bias}"
    print("✅ 正式config(lookback_bars=confirm_bars=3)下，bear_foldback現在真的可以觸發")


def test_bull_foldback_reachable_with_production_default_lookback_equals_confirm():
    cfg = RULES["indicators"]["cci"]
    values = [-50] * 5 + [-120, -120, -120] + [-50, -50, -50]
    df = _build_cci_df(values)
    reasons, caveats = [], []
    cci_exit_alert, cci_bias = _step5b_cci(df, "60分", df.iloc[-1], reasons, caveats)
    assert cci_exit_alert is False, "牛市折返不改變cci_exit_alert行為（仍只是caveat提示）"
    assert cci_bias == 1, f"牛市折返應給cci_bias=1，實際={cci_bias}"
    print("✅ 正式config下，bull_foldback現在真的可以觸發")


def test_no_foldback_when_extreme_value_still_inside_confirm_window():
    """極端值如果落在confirm視窗「裡面」（代表還沒真的連續3根都折返回
    門檻內），不該誤判成foldback——confirm_bars的本意就是要連續N根都
    已經回到門檻內，不是「最近隨便一根」都算。"""
    values = [50] * 8 + [120, 50, 50]  # 120落在confirm視窗(最後3根)的第一根，confirmed_below應該為False
    df = _build_cci_df(values)
    reasons, caveats = [], []
    cci_exit_alert, cci_bias = _step5b_cci(df, "60分", df.iloc[-1], reasons, caveats)
    assert cci_exit_alert is False, "極端值還在confirm視窗內（尚未連續3根都折返）時不該觸發foldback"
    assert cci_bias == 0.0
    print("✅ 極端值還在confirm視窗內（尚未連續N根都折返）時，不會誤判成foldback")


def test_confirm_bars_1_legacy_path_unaffected():
    """confirm_bars<=1的舊版單根觸發邏輯完全不受這次修改影響。"""
    cci_cfg = RULES["indicators"]["cci"]
    orig_confirm = cci_cfg["confirm_bars"]
    cci_cfg["confirm_bars"] = 1
    try:
        values = [120, 120, 50]  # 單根折返就立刻觸發(舊行為)
        df = _build_cci_df(values)
        reasons, caveats = [], []
        cci_exit_alert, cci_bias = _step5b_cci(df, "60分", df.iloc[-1], reasons, caveats)
    finally:
        cci_cfg["confirm_bars"] = orig_confirm
    assert cci_exit_alert is True, "confirm_bars=1時應維持舊版單根即觸發的行為"
    print("✅ confirm_bars<=1的舊版行為不受這次修改影響")


if __name__ == "__main__":
    test_bear_foldback_reachable_with_production_default_lookback_equals_confirm()
    test_bull_foldback_reachable_with_production_default_lookback_equals_confirm()
    test_no_foldback_when_extreme_value_still_inside_confirm_window()
    test_confirm_bars_1_legacy_path_unaffected()
    print()
    print("全部通過。")
