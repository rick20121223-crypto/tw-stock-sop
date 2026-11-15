"""
合格性篩選：決定某檔股票在某個日期是否可以交易（排除流動性不足、處置股、
全額交割股、新股）。純函式，不打API——這個月只做判斷邏輯本身，實際抓取
處置股/全額交割股/上市日期/20日均成交值的即時資料留到第3個月接自動引擎
時再做（呼叫端自己準備好這些輸入值傳進來）。

缺資料時一律判定「不合格」（fail closed）——這是風險控管用的篩選器，
查不到資料時假設合格反而危險，寧可保守排除。

第3個月要接的官方資料來源（已查好，待實作）：
- 處置股（TWSE）：https://openapi.twse.com.tw/v1/announcement/punish
- 全額交割股/變更交易（TWSE）：https://openapi.twse.com.tw/v1/exchangeReport/TWT85U
- 最近上市公司（TWSE）：https://openapi.twse.com.tw/v1/company/newlisting
- TPEx（上櫃）有對應的處置/變更交易公告頁面，確切JSON端點待實作時依
  holding_shares.py既有的「免金鑰公開API」模式確認。
- 20日均成交值：backtest/data_fetch.py的fetch_twse_daily_turnover()/
  fetch_tpex_latest_turnover()可重用，累加20個交易日。
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_RULES_PATH = Path(__file__).with_name("trading_rules.yaml")


def _load_rules() -> Dict:
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


TRADING_RULES = _load_rules()


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: List[str] = field(default_factory=list)


def is_eligible(code: str, as_of_date: date, listing_date: Optional[date],
                 avg_trade_value_20d: Optional[float], is_disposition: bool,
                 is_full_delivery: bool, rules: Dict = TRADING_RULES) -> EligibilityResult:
    """listing_date/avg_trade_value_20d為None時視為查無資料，fail closed
    （判定不合格）。is_disposition/is_full_delivery由呼叫端自行查好傳入，
    這裡只套用trading_rules.yaml裡的開關跟門檻。"""
    reasons: List[str] = []
    elig_cfg = rules["eligibility"]
    liq_cfg = rules["liquidity"]

    if elig_cfg.get("exclude_disposition", True) and is_disposition:
        reasons.append(f"{code}是處置股")

    if elig_cfg.get("exclude_full_delivery", True) and is_full_delivery:
        reasons.append(f"{code}是全額交割股")

    if elig_cfg.get("exclude_new_listing", True):
        min_days = elig_cfg.get("new_listing_min_trading_days", 60)
        if listing_date is None:
            reasons.append(f"{code}查無上市日期，無法確認是否未滿{min_days}天新股門檻")
        else:
            days_listed = (as_of_date - listing_date).days
            if days_listed < min_days:
                reasons.append(f"{code}上市僅{days_listed}天（日曆天數近似），未滿{min_days}天門檻")

    min_value = liq_cfg.get("min_avg_trade_value_20d", 0)
    if avg_trade_value_20d is None:
        reasons.append(f"{code}查無20日均成交值，無法確認流動性")
    elif avg_trade_value_20d < min_value:
        reasons.append(f"{code}20日均成交值{avg_trade_value_20d:,.0f}低於門檻{min_value:,.0f}")

    return EligibilityResult(eligible=(len(reasons) == 0), reasons=reasons)
