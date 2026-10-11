"""
驗證evaluate_timeframe()接上suggested_exposure後：
1. 週線(tf="週")完全不呼叫_step_kd365()，而且suggested_exposure固定是
   None（跟「週線與月線完全不動」的範圍限制一致，不要讓部位建議附掛在
   週線輸出上）。這個不變量是chain裡的第一個分支、不會被底下任何條件
   覆蓋，可以嚴格assert。
2. 日線在多組隨機種子下，suggested_exposure都落在合法值集合內，而且
   符合「有明確結論就該有明確曝險建議」的不變量：conclusion是買進/加碼/
   賣出減碼(不是觀望)時，suggested_exposure不該是None——
   注意：這裡刻意不assert「買進/加碼一定等於100」，因為
   kd365_bearish_divergence是獨立訊號、不會否決conclusion，所以conclusion
   ="加碼"但suggested_exposure因為KD365示警而降到50是合法、刻意的設計
   （曝險建議本來就是比conclusion更謹慎的獨立風險層，不是conclusion的
   另一種編碼方式）。
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
    # 1. 週線固定None，不管哪個種子
    for seed in range(5):
        df_week = run_all_indicators(_make_synthetic_df(seed=seed), "週")
        v_week = evaluate_timeframe(df_week, "週")
        assert v_week.suggested_exposure is None, \
            f"seed={seed}: 週線suggested_exposure應為None，實際={v_week.suggested_exposure}"
    print("OK: 週線在多組種子下suggested_exposure固定是None")

    # 2. 日線在多組種子下跑出合法值，且「有明確結論就有明確曝險建議」成立
    seen_values = set()
    for seed in range(60):
        df_day = run_all_indicators(_make_synthetic_df(seed=seed), "日")
        v_day = evaluate_timeframe(df_day, "日")
        assert v_day.suggested_exposure in (0, 30, 50, 70, 100, None), \
            f"seed={seed}: 不合法的suggested_exposure={v_day.suggested_exposure}"
        seen_values.add(v_day.suggested_exposure)

        if v_day.conclusion in ("買進", "加碼", "賣出減碼"):
            assert v_day.suggested_exposure is not None, \
                f"seed={seed}: conclusion={v_day.conclusion}（明確結論）但suggested_exposure=None（不該發生）"

    print(f"OK: 日線60組種子suggested_exposure都是合法值，且「買進/加碼/賣出減碼"
          f"都有明確曝險建議、不會是None」成立"
          f"（涵蓋到的值：{sorted(v for v in seen_values if v is not None)}"
          f"{'、None' if None in seen_values else ''}）")

    # 3. combine_timeframes()有「建議曝險」鍵
    df_day = run_all_indicators(_make_synthetic_df(seed=0), "日")
    v_day = evaluate_timeframe(df_day, "日")
    detail = combine_timeframes({"日": v_day})
    assert "建議曝險" in detail["各週期明細"]["日"]
    assert detail["各週期明細"]["日"]["建議曝險"] == v_day.suggested_exposure
    print("OK: combine_timeframes()的detail正確帶出「建議曝險」")

    print("\n全部通過")


if __name__ == "__main__":
    main()
