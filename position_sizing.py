"""
部位規模計算：用「進場價到停損價的距離」反推股數，同時受單檔部位上限、
現金、持股檔數上限三個限制，取最小值——呼應performance.py「沒有停損價
就不計入R倍數」的同一套設計語言：沒有有效停損價就直接拒絕，不會算出一個
沒有風險依據的股數。純函式，不依賴Streamlit，供第3個月的自動引擎直接
呼叫。

限制：`shares_by_cash`沒有預先扣掉手續費（只用`cash_available / entry_price`
粗算），實際下單金額會比這裡算出的再多一點點手續費，這是已知的簡化。
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

_CONSTRAINT_LABELS = {
    "risk": "單筆最大虧損",
    "position_cap": "單檔部位上限",
    "cash": "現金",
}


@dataclass
class PositionSizeResult:
    shares: int
    approved: bool
    binding_constraint: Optional[str]  # "risk" / "position_cap" / "cash" / "holdings_count" / "invalid_stop_loss" / None
    reasons: List[str]


def calc_position_size(equity: float, entry_price: float, stop_loss: Optional[float],
                        cash_available: float, current_holdings_count: int,
                        account) -> PositionSizeResult:
    """account需要有`max_loss_pct`/`max_position_pct`/`max_holdings_count`
    三個屬性（即`ledger_store.Account`）。假設這是「開新倉」的試算（不是
    對既有持股加碼）——加碼情境的持股檔數上限判斷不適用，由呼叫端自行
    跳過這個函式或另外處理。"""
    if current_holdings_count >= account.max_holdings_count:
        return PositionSizeResult(
            shares=0, approved=False, binding_constraint="holdings_count",
            reasons=[f"已達持股檔數上限（{account.max_holdings_count}檔）"],
        )

    if stop_loss is None or stop_loss >= entry_price:
        return PositionSizeResult(
            shares=0, approved=False, binding_constraint="invalid_stop_loss",
            reasons=["沒有有效停損價（必須低於進場價），無法用風險反推股數"],
        )

    risk_per_share = entry_price - stop_loss
    candidates: Dict[str, int] = {
        "risk": math.floor(equity * account.max_loss_pct / risk_per_share),
        "position_cap": math.floor(equity * account.max_position_pct / entry_price),
        "cash": math.floor(cash_available / entry_price),
    }
    binding_constraint = min(candidates, key=candidates.get)
    final_shares = candidates[binding_constraint]
    label = _CONSTRAINT_LABELS[binding_constraint]

    if final_shares <= 0:
        return PositionSizeResult(
            shares=0, approved=False, binding_constraint=binding_constraint,
            reasons=[f"依{label}限制，可買股數為0"],
        )

    return PositionSizeResult(
        shares=final_shares, approved=True, binding_constraint=binding_constraint,
        reasons=[f"依{label}限制，核准{final_shares}股"],
    )


def prioritize_candidates(candidates: List[Dict], sort_key: str = "avg_trade_value_20d",
                           descending: bool = True) -> List[Dict]:
    """同一天訊號數超過空位/現金時的排序。candidates的每個元素至少要有
    `sort_key`指定的那個欄位；回傳一份新的list（不修改原始dict），依
    `sort_key`排序後，每個元素多加`priority_rank`（1起算）跟
    `priority_reason`兩個欄位，供呼叫端寫進`Order.priority_rank`/
    `priority_reason`留下稽核紀錄。"""
    ordered = sorted(candidates, key=lambda c: c[sort_key], reverse=descending)
    direction = "由高到低" if descending else "由低到高"
    result = []
    for i, c in enumerate(ordered, start=1):
        updated = dict(c)
        updated["priority_rank"] = i
        updated["priority_reason"] = f"依{sort_key}{direction}排序，第{i}名"
        result.append(updated)
    return result
