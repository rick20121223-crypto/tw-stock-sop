"""
用真正的TAIEX重跑項目3，並跟0050代理指標比較一致性。

用法：python3 -m backtest.round3_taiex_compare
"""
from pathlib import Path

import pandas as pd

from backtest.market_regime import (attach_regime_and_excess_return,
                                     build_benchmark_regime, fetch_0050_proxy)
from backtest.run_backtest import _fetch_start
from backtest.summarize import summarize_by_regime

OUTPUT_DIR = Path(__file__).parent / "output"
SPLIT_DATE = pd.Timestamp("2025-04-15")


def main(api_token: str = ""):
    from datetime import date
    start = _fetch_start(3)
    today = date.today().isoformat()

    taiex = pd.read_csv(OUTPUT_DIR / "taiex_daily.csv")
    taiex_regime = build_benchmark_regime(taiex)

    proxy = fetch_0050_proxy(start, today, api_token)
    proxy_regime = build_benchmark_regime(proxy)

    # --- 一致性比較 ---
    cmp = taiex_regime[["date", "close", "regime_ma60", "regime_ret60"]].merge(
        proxy_regime[["date", "close", "regime_ma60", "regime_ret60"]],
        on="date", suffixes=("_taiex", "_0050"))
    corr = cmp["close_taiex"].corr(cmp["close_0050"])
    agree_ma60 = (cmp["regime_ma60_taiex"] == cmp["regime_ma60_0050"]).mean()
    agree_ret60 = (cmp["regime_ret60_taiex"] == cmp["regime_ret60_0050"]).mean()
    print(f"TAIEX vs 0050代理：價格相關係數={corr:.4f}，"
          f"regime_ma60一致率={agree_ma60*100:.1f}%，regime_ret60一致率={agree_ret60*100:.1f}%")
    pd.DataFrame([{"相關係數": round(corr, 4), "regime_ma60一致率%": round(agree_ma60*100, 1),
                    "regime_ret60一致率%": round(agree_ret60*100, 1), "比較天數": len(cmp)}]) \
        .to_csv(OUTPUT_DIR / "r3_item3_taiex_vs_0050.csv", index=False)

    # --- 用真正TAIEX重算 ---
    buy = pd.read_csv(OUTPUT_DIR / "buy_events.csv", dtype={"code": str})
    buy = buy[~buy["near_unadjusted_gap"]]
    w = pd.read_csv(OUTPUT_DIR / "w_bottom_events.csv", dtype={"code": str})
    w = w[~w["near_unadjusted_gap"]]

    buy_r = attach_regime_and_excess_return(buy, taiex_regime, "regime_ma60")
    w_r = attach_regime_and_excess_return(w, taiex_regime, "regime_ma60")

    regime_only_buy = summarize_by_regime(buy_r)
    regime_only_w = summarize_by_regime(w_r)
    regime_only_buy.to_csv(OUTPUT_DIR / "r3_item3_taiex_regime_only_buy.csv", index=False)
    regime_only_w.to_csv(OUTPUT_DIR / "r3_item3_taiex_regime_only_w.csv", index=False)

    buy_r["date_dt"] = pd.to_datetime(buy_r["date"])
    w_r["date_dt"] = pd.to_datetime(w_r["date"])
    buy_r["period"] = (buy_r["date_dt"] >= SPLIT_DATE).map({True: "後半", False: "前半"})
    w_r["period"] = (w_r["date_dt"] >= SPLIT_DATE).map({True: "後半", False: "前半"})

    def _period_regime(df, name_col="signal"):
        rows = []
        for (sig, period, regime), g in df.dropna(subset=["regime"]).groupby([name_col, "period", "regime"]):
            ret = g["fwd_ret_20d"].dropna()
            exc = g["excess_ret_20d"].dropna()
            n = len(g)
            rows.append({name_col: sig, "期間": period, "環境": regime, "樣本數": n,
                         "樣本不足": "是" if n < 30 else "",
                         "20日報酬%": round(ret.mean() * 100, 2) if len(ret) else None,
                         "20日勝率%": round((ret > 0).mean() * 100, 1) if len(ret) else None,
                         "20日超額報酬%": round(exc.mean() * 100, 2) if len(exc) else None,
                         "20日超額勝率%": round((exc > 0).mean() * 100, 1) if len(exc) else None})
        return pd.DataFrame(rows).sort_values([name_col, "期間", "環境"])

    buy_period_regime = _period_regime(buy_r)
    w_period_regime = _period_regime(w_r)
    buy_period_regime.to_csv(OUTPUT_DIR / "r3_item3_taiex_buy_period_regime.csv", index=False)
    w_period_regime.to_csv(OUTPUT_DIR / "r3_item3_taiex_w_period_regime.csv", index=False)

    taiex_regime["date_dt"] = pd.to_datetime(taiex_regime["date"])
    rows = []
    for label, sub in [("全期間", taiex_regime), ("前半段", taiex_regime[taiex_regime["date_dt"] < SPLIT_DATE]),
                        ("後半段", taiex_regime[taiex_regime["date_dt"] >= SPLIT_DATE])]:
        vc = sub["regime_ma60"].value_counts(normalize=True) * 100
        vc2 = sub["regime_ret60"].value_counts(normalize=True) * 100
        rows.append({"期間": label, "交易日數": len(sub),
                     "regime_ma60多頭%": round(vc.get("多頭", 0), 1),
                     "regime_ret60多頭%": round(vc2.get("多頭", 0), 1)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "r3_item3_taiex_regime_distribution.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n真正TAIEX - regime_only買進側:")
    print(regime_only_buy.to_string(index=False))
    print("\n真正TAIEX - regime_only W底:")
    print(regime_only_w.to_string(index=False))


if __name__ == "__main__":
    main()
