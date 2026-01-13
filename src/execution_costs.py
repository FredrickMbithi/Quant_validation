from __future__ import annotations

"""
Execution cost utilities for FX backtests.

This module provides simple, configurable estimates for spreads and slippage
so mid-only strategies can be stress-tested with more realistic fills.

Conventions
- Prices are spot (e.g., GBPUSD ~ 1.2500)
- Pips are standard: 1 pip = 1e-4 for most USD majors, 1e-2 for JPY crosses
- Costs can be converted to return space (decimal) for integration with
  return-based backtests.

Note: These are heuristics. Calibrate with your broker/LP data if available.
"""

from dataclasses import dataclass
from typing import Literal

Session = Literal["asia", "london", "newyork"]
Regime = Literal["low", "normal", "high"]


def pip_value(pair: str) -> float:
    """Return decimal value per pip for a given pair.

    Examples
    - GBPUSD: 1 pip = 0.0001
    - USDJPY: 1 pip = 0.01
    """
    pair = pair.upper()
    if pair.endswith("JPY"):
        return 1e-2
    return 1e-4


def estimate_spread_pips(session: Session, regime: Regime, pair: str = "GBPUSD") -> float:
    """Estimate spread in pips by session and volatility regime.

    Defaults are conservative for retail-like conditions. Tune as needed.
    """
    base = {
        "GBPUSD": {"asia": 1.6, "london": 1.0, "newyork": 1.1},
        "EURUSD": {"asia": 1.2, "london": 0.8, "newyork": 0.9},
        "USDJPY": {"asia": 1.3, "london": 0.9, "newyork": 1.0},
    }
    pair = pair.upper()
    sess = session.lower()
    reg = regime.lower()

    spread = base.get(pair, base["GBPUSD"]).get(sess, 1.2)

    # Volatility surcharge
    if reg == "high":
        spread += 0.5
    elif reg == "low":
        spread -= 0.2

    return max(spread, 0.1)


def estimate_slippage_pips(regime: Regime, event: bool = False) -> float:
    """Estimate one-way slippage in pips.

    - Normal regime: ~0.2 pip
    - High vol: ~0.5 pip
    - Event spikes: add +0.5–1.0 pips
    """
    reg = regime.lower()
    slip = 0.2 if reg in ("low", "normal") else 0.5
    if event:
        slip += 0.7
    return slip


def round_trip_cost_return(
    price: float,
    spread_pips: float,
    slippage_entry_pips: float,
    slippage_exit_pips: float,
    pair: str = "GBPUSD",
) -> float:
    """Approximate round-trip cost as a fraction of price (return space).

    For a long entered at mid, you effectively pay half-spread on entry
    and half-spread on exit, plus slippage on both sides.
    """
    pip = pip_value(pair)
    total_pips = spread_pips + slippage_entry_pips + slippage_exit_pips
    # Convert pips to price delta and then to return fraction
    price_delta = total_pips * pip
    return price_delta / max(price, 1e-12)


@dataclass
class CostModel:
    pair: str = "GBPUSD"

    def per_trade_cost_return(
        self,
        price: float,
        session: Session,
        regime: Regime,
        is_event: bool = False,
    ) -> float:
        """Convenience wrapper to compute a round-trip cost in return space."""
        spread = estimate_spread_pips(session, regime, self.pair)
        slip_in = estimate_slippage_pips(regime, event=is_event)
        slip_out = estimate_slippage_pips(regime, event=is_event)
        return round_trip_cost_return(price, spread, slip_in, slip_out, self.pair)
