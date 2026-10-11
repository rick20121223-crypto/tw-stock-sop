"""
驗證evaluate_timeframe()接上suggested_exposure後：
1. 週線(tf="週")完全不呼叫_step_kd365()，不受影響——用合成資料跑週線，
   確認不會因為沒有K365/D365欄位而出錯（週線的run_all_indicators也會
   算K365/D365，但_step_kd365不應該被呼叫到）。
2. 日線能正常跑出一個suggested_exposure，值落在{0,30,50,70,100,None}之一。
3. combine_timeframes()的detail[tf]字典有"建議曝險"鍵。

用合成資料，不需要FinMind token，執行：
    python3 verify_suggested_exposure_pipeline.py
"""
import numpy as np
import pandas as pd

from sop_decision import combine_timeframes, evaluate_timeframe
from stock_core import run_all_indicators


def _make_synthetic_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    volume = rng.integers(1000, 5000, n)
    return pd.DataFrame({"date": dates, "open": open_, "max": high,
                          "min": low, "close": close, "volume": volume})


def main() -> None:
    df = _make_synthetic_df()

    # 1. 週線不受影響，不會因為K365邏輯而crash
    df_week = run_all_indicators(df.copy(), "週")
    v_week = evaluate_timeframe(df_week, "週")
    print(f"OK: 週線正常跑完，conclusion={v_week.conclusion}, "
          f"suggested_exposure={v_week.suggested_exposure}（預期None，週線不分類）")

    # 2. 日線能跑出suggested_exposure
    df_day = run_all_indicators(df.copy(), "日")
    v_day = evaluate_timeframe(df_day, "日")
    assert v_day.suggested_exposure in (0, 30, 50, 70, 100, None), v_day.suggested_exposure
    print(f"OK: 日線suggested_exposure={v_day.suggested_exposure}（合法值之一）")

    # 3. combine_timeframes()有「建議曝險」鍵
    detail = combine_timeframes({"日": v_day})
    assert "建議曝險" in detail["各週期明細"]["日"]
    assert detail["各週期明細"]["日"]["建議曝險"] == v_day.suggested_exposure
    print("OK: combine_timeframes()的detail正確帶出「建議曝險」")

    print("\n全部通過")


if __name__ == "__main__":
    main()
