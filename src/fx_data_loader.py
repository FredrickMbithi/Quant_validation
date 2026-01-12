"""
FX Data Loader - Stage 2: Data Acquisition & Integrity

PARANOID FX DATA HANDLING
=========================
This module assumes ALL FX data is:
- Broker-specific (not neutral)
- Timezone-corrupted until proven UTC
- Spread-hostile (mid-price fills are fantasy)
- Volume-meaningless (tick != traded)
- Gap-ridden (weekends, holidays, sessions)

NO CLEANING. NO RESAMPLING. EXPOSE EVERYTHING.

Data Source Acknowledgment
--------------------------
This loader is designed for:
- Primary: HistData.com (free, bid prices, 1-min bars, UTC)
- Secondary: Dukascopy (tick data, requires conversion)
- Fallback: Broker exports (OANDA, IC Markets - MUST state which)

NEVER use Yahoo Finance for FX intraday - it's garbage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
import json


class FXDataIntegrityError(Exception):
    """Raised when FX data fails critical integrity checks."""
    pass


class FXDataWarning(UserWarning):
    """Warning for non-fatal FX data issues that MUST be acknowledged."""
    pass


class FXDataMetadata:
    """
    Explicit metadata about FX data source.
    If you can't fill this out, you don't understand your data.
    """
    
    def __init__(
        self,
        source: str,
        broker: Optional[str],
        price_type: str,  # "bid", "ask", "mid" - NO OTHER OPTIONS
        timezone: str,  # Must be "UTC" or explicit offset
        volume_type: str,  # "tick", "real", "none"
        spread_available: bool,
        bar_interval: str,  # "1min", "5min", etc.
        dst_handling: str,  # "utc_native", "local_shifted", "unknown"
    ):
        self.source = source
        self.broker = broker
        self.price_type = self._validate_price_type(price_type)
        self.timezone = self._validate_timezone(timezone)
        self.volume_type = self._validate_volume_type(volume_type)
        self.spread_available = spread_available
        self.bar_interval = bar_interval
        self.dst_handling = dst_handling
        self.acknowledged_limitations = []
        
    def _validate_price_type(self, price_type: str) -> str:
        valid = {"bid", "ask", "mid"}
        if price_type.lower() not in valid:
            raise ValueError(f"price_type must be one of {valid}, got '{price_type}'")
        return price_type.lower()
    
    def _validate_timezone(self, tz: str) -> str:
        if tz.upper() != "UTC" and not tz.startswith("UTC"):
            raise FXDataIntegrityError(
                f"FX data MUST be in UTC. Got '{tz}'. "
                "Convert to UTC BEFORE loading or acknowledge timezone risk."
            )
        return tz.upper()
    
    def _validate_volume_type(self, vol_type: str) -> str:
        valid = {"tick", "real", "none"}
        if vol_type.lower() not in valid:
            raise ValueError(f"volume_type must be one of {valid}")
        return vol_type.lower()
    
    def acknowledge_limitation(self, limitation: str):
        """Explicitly acknowledge a data limitation."""
        self.acknowledged_limitations.append(limitation)
        
    def get_execution_realism_warning(self) -> str:
        """Return warnings about execution realism based on metadata."""
        warnings_list = []
        
        if self.price_type == "mid":
            warnings_list.append(
                "⚠️ MID-PRICE DATA: You cannot execute at mid. "
                "All backtest fills are fantasy without spread model."
            )
        elif self.price_type == "bid":
            warnings_list.append(
                "⚠️ BID-PRICE DATA: Sells execute here, buys need ask estimate."
            )
        elif self.price_type == "ask":
            warnings_list.append(
                "⚠️ ASK-PRICE DATA: Buys execute here, sells need bid estimate."
            )
            
        if self.volume_type == "tick":
            warnings_list.append(
                "⚠️ TICK VOLUME: This is NOT traded volume. "
                "It's price update frequency. Liquidity is unknowable."
            )
        elif self.volume_type == "none":
            warnings_list.append(
                "⚠️ NO VOLUME: Market depth is completely unknown."
            )
            
        if not self.spread_available:
            warnings_list.append(
                "⚠️ NO SPREAD DATA: Must model spreads. "
                "Typical EUR/USD: 0.5-2 pips. During news: 5-20+ pips."
            )
            
        if self.broker:
            warnings_list.append(
                f"⚠️ BROKER-SPECIFIC DATA ({self.broker}): "
                "Prices may differ from other brokers by 1-5 pips."
            )
            
        return "\n".join(warnings_list)
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "broker": self.broker,
            "price_type": self.price_type,
            "timezone": self.timezone,
            "volume_type": self.volume_type,
            "spread_available": self.spread_available,
            "bar_interval": self.bar_interval,
            "dst_handling": self.dst_handling,
            "acknowledged_limitations": self.acknowledged_limitations,
        }


class FXSessionLabeler:
    """
    Label FX data by trading session.
    
    Sessions are APPROXIMATE and broker-dependent:
    - Sydney: 21:00 - 06:00 UTC (varies with DST)
    - Tokyo:  00:00 - 09:00 UTC
    - London: 07:00 - 16:00 UTC (08:00 - 17:00 during DST)
    - New York: 12:00 - 21:00 UTC (13:00 - 22:00 during DST)
    
    OVERLAP PERIODS are highest liquidity (and tightest spreads).
    """
    
    # UTC hours (approximate, DST shifts these)
    SESSIONS = {
        "sydney": (21, 6),    # Wraps midnight
        "tokyo": (0, 9),
        "london": (7, 16),
        "new_york": (12, 21),
    }
    
    @classmethod
    def label_session(cls, timestamp: pd.Timestamp) -> List[str]:
        """
        Return list of active sessions for a timestamp.
        Multiple sessions can be active (overlaps).
        """
        hour = timestamp.hour
        active = []
        
        for session, (start, end) in cls.SESSIONS.items():
            if start > end:  # Wraps midnight (Sydney)
                if hour >= start or hour < end:
                    active.append(session)
            else:
                if start <= hour < end:
                    active.append(session)
                    
        if not active:
            active.append("off_hours")  # Weekend or gap
            
        return active
    
    @classmethod
    def add_session_labels(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add session labels to dataframe."""
        df = df.copy()
        df["sessions"] = df.index.map(lambda x: cls.label_session(x))
        df["primary_session"] = df["sessions"].apply(lambda x: x[0] if x else "unknown")
        df["is_overlap"] = df["sessions"].apply(lambda x: len(x) > 1)
        return df


class FXDataLoader:
    """
    Paranoid FX Data Loader.
    
    This loader does ONE thing: load raw FX data and EXPOSE its problems.
    
    It does NOT:
    - Clean data
    - Resample data
    - Fill gaps
    - Interpolate spreads
    - Assume anything is correct
    
    Default assumption: HistData.com format
    - CSV with columns: timestamp, open, high, low, close, volume
    - Bid prices (NOT mid, NOT ask)
    - 1-minute bars
    - UTC timezone
    - Tick volume (NOT real volume)
    """
    
    def __init__(
        self, 
        data_dir: str = "data/raw/fx",
        metadata: Optional[FXDataMetadata] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default metadata for HistData.com
        self.metadata = metadata or FXDataMetadata(
            source="HistData.com",
            broker=None,  # HistData aggregates from multiple sources
            price_type="bid",
            timezone="UTC",
            volume_type="tick",
            spread_available=False,
            bar_interval="1min",
            dst_handling="utc_native",
        )
        
        self.integrity_report: Dict = {}
        
    def load_csv(
        self, 
        pair: str,
        timestamp_col: str = "timestamp",
        verify: bool = True,
        fail_on_warning: bool = False
    ) -> pd.DataFrame:
        """
        Load raw FX intraday data.
        
        NO CLEANING. NO RESAMPLING. RAW DATA ONLY.
        
        Args:
            pair: Currency pair (e.g., "EURUSD")
            timestamp_col: Name of timestamp column
            verify: Run integrity checks
            fail_on_warning: Raise exception on warnings (paranoid mode)
            
        Returns:
            DataFrame with verified raw data
            
        Raises:
            FileNotFoundError: Data file doesn't exist
            FXDataIntegrityError: Critical integrity failure
        """
        filepath = self.data_dir / f"{pair}.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"No data for {pair} at {filepath}\n"
                f"Download from {self.metadata.source} and place CSV here."
            )
        
        # Load raw - minimal parsing
        df = pd.read_csv(filepath)
        
        # Normalize column names (lowercase)
        df.columns = df.columns.str.lower().str.strip()
        
        # ---------- REQUIRED COLUMNS ----------
        required_cols = {"open", "high", "low", "close"}
        
        # Find timestamp column (various names)
        timestamp_candidates = ["timestamp", "datetime", "date", "time", "date time"]
        found_ts = None
        for col in timestamp_candidates:
            if col in df.columns:
                found_ts = col
                break
                
        if found_ts is None:
            raise FXDataIntegrityError(
                f"{pair}: No timestamp column found. "
                f"Expected one of {timestamp_candidates}, got {list(df.columns)}"
            )
        
        missing = required_cols - set(df.columns)
        if missing:
            raise FXDataIntegrityError(f"{pair} missing required columns: {missing}")
        
        # ---------- TIMEZONE ENFORCEMENT ----------
        try:
            df[found_ts] = pd.to_datetime(df[found_ts], utc=True)
        except Exception as e:
            raise FXDataIntegrityError(
                f"{pair}: Cannot parse timestamps as UTC. Error: {e}\n"
                "FX data MUST be in UTC. Fix source data or convert before loading."
            )
        
        df.set_index(found_ts, inplace=True)
        df.index.name = "timestamp"
        df = df.sort_index()
        
        # ---------- INTEGRITY VERIFICATION ----------
        if verify:
            self.integrity_report[pair] = self._verify_fx_integrity(
                df, pair, fail_on_warning
            )
            
        # Add session labels
        df = FXSessionLabeler.add_session_labels(df)
        
        return df
    
    def _verify_fx_integrity(
        self, 
        df: pd.DataFrame, 
        pair: str,
        fail_on_warning: bool = False
    ) -> Dict:
        """
        FX-specific data integrity checks.
        
        This function is PARANOID. It assumes data is lying until proven otherwise.
        
        Returns:
            Dictionary with integrity report
        """
        report = {
            "pair": pair,
            "rows": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "warnings": [],
            "errors": [],
            "statistics": {},
        }
        
        def warn(msg: str):
            report["warnings"].append(msg)
            warnings.warn(f"{pair}: {msg}", FXDataWarning)
            if fail_on_warning:
                raise FXDataIntegrityError(f"{pair}: {msg}")
        
        # ========== 1. TIME GAP ANALYSIS ==========
        diffs = df.index.to_series().diff()
        expected_gap = pd.Timedelta(self.metadata.bar_interval)
        
        # Regular gaps (market closed)
        weekend_gaps = diffs[diffs > pd.Timedelta(hours=24)]
        report["statistics"]["weekend_gaps"] = len(weekend_gaps)
        
        # Suspicious gaps (data missing)
        suspicious_gaps = diffs[
            (diffs > expected_gap * 5) & 
            (diffs <= pd.Timedelta(hours=24))
        ]
        
        if len(suspicious_gaps) > 0:
            warn(
                f"SUSPICIOUS GAPS: {len(suspicious_gaps)} gaps > 5 bars but < 24h. "
                f"Possible missing data or DST issues.\n"
                f"First 5: {suspicious_gaps.head().to_dict()}"
            )
        report["statistics"]["suspicious_gaps"] = len(suspicious_gaps)
        
        # Micro gaps (bars too close - possible duplicates or bad source)
        micro_gaps = diffs[diffs < expected_gap * 0.9]
        if len(micro_gaps) > 1:  # First diff is NaT
            warn(
                f"MICRO GAPS: {len(micro_gaps)-1} bars closer than expected. "
                "Possible duplicates or inconsistent source."
            )
        report["statistics"]["micro_gaps"] = max(0, len(micro_gaps) - 1)
        
        # ========== 2. PRICE SANITY ==========
        # Zero-range candles (bad ticks or illiquid periods)
        zero_range = (df["high"] == df["low"]).sum()
        zero_pct = zero_range / len(df) * 100
        
        if zero_pct > 1:  # More than 1% zero-range
            warn(
                f"ZERO-RANGE CANDLES: {zero_range} ({zero_pct:.2f}%) bars have high==low. "
                "Either illiquid periods or bad data."
            )
        report["statistics"]["zero_range_candles"] = zero_range
        report["statistics"]["zero_range_pct"] = zero_pct
        
        # OHLC consistency (high >= low, high >= open/close, low <= open/close)
        ohlc_violations = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ).sum()
        
        if ohlc_violations > 0:
            warn(
                f"OHLC VIOLATIONS: {ohlc_violations} bars have impossible OHLC relationships. "
                "Data is CORRUPT."
            )
            report["errors"].append(f"OHLC violations: {ohlc_violations}")
        report["statistics"]["ohlc_violations"] = ohlc_violations
        
        # Negative prices (should never happen in FX)
        negative_prices = (
            (df["open"] <= 0) | 
            (df["high"] <= 0) | 
            (df["low"] <= 0) | 
            (df["close"] <= 0)
        ).sum()
        
        if negative_prices > 0:
            raise FXDataIntegrityError(
                f"{pair}: {negative_prices} bars have zero or negative prices. "
                "Data is FUNDAMENTALLY BROKEN."
            )
        
        # ========== 3. EXTREME MOVES (FLASH CRASHES / BAD DATA) ==========
        returns = df["close"].pct_change()
        
        # Different thresholds for different pairs
        if "JPY" in pair:
            extreme_threshold = 0.005  # 0.5% for JPY pairs (smaller moves)
        else:
            extreme_threshold = 0.01  # 1% for major pairs
            
        extreme_moves = returns[abs(returns) > extreme_threshold]
        
        if len(extreme_moves) > 0:
            warn(
                f"EXTREME MOVES: {len(extreme_moves)} bars with >{extreme_threshold*100}% moves.\n"
                f"Could be: flash crash, rollover, news, or BAD DATA.\n"
                f"Largest: {extreme_moves.abs().max()*100:.2f}% at {extreme_moves.abs().idxmax()}\n"
                f"First 5:\n{extreme_moves.head()}"
            )
        report["statistics"]["extreme_moves"] = len(extreme_moves)
        
        # ========== 4. DUPLICATE TIMESTAMPS ==========
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            warn(
                f"DUPLICATE TIMESTAMPS: {duplicates} duplicate timestamps found. "
                "Data source is unreliable."
            )
        report["statistics"]["duplicate_timestamps"] = duplicates
        
        # ========== 5. STALE PRICES ==========
        # Consecutive identical closes (market frozen or bad feed)
        stale_mask = df["close"] == df["close"].shift(1)
        stale_runs = (stale_mask != stale_mask.shift()).cumsum()
        stale_lengths = stale_mask.groupby(stale_runs).sum()
        long_stale = stale_lengths[stale_lengths > 10]  # More than 10 consecutive
        
        if len(long_stale) > 0:
            warn(
                f"STALE PRICES: {len(long_stale)} periods with >10 consecutive identical closes. "
                "Market was frozen or data feed died."
            )
        report["statistics"]["long_stale_periods"] = len(long_stale)
        
        # ========== 6. VOLUME SANITY (if available) ==========
        if "volume" in df.columns:
            zero_volume = (df["volume"] == 0).sum()
            zero_vol_pct = zero_volume / len(df) * 100
            
            if zero_vol_pct > 10:
                warn(
                    f"ZERO VOLUME: {zero_volume} ({zero_vol_pct:.2f}%) bars have zero volume. "
                    f"Remember: this is TICK volume, not real volume."
                )
            report["statistics"]["zero_volume_bars"] = zero_volume
            
            # Volume is always questionable in FX
            warn(
                "TICK VOLUME WARNING: Volume column is tick count, NOT traded volume. "
                "Real FX volume is unknowable without prime broker data."
            )
        else:
            report["warnings"].append("No volume column - cannot assess tick activity")
        
        # ========== 7. DST TRANSITION CHECK ==========
        # Look for hour skips in March/November (common DST months)
        march_data = df[df.index.month == 3]
        november_data = df[df.index.month == 11]
        
        for month_name, month_data in [("March", march_data), ("November", november_data)]:
            if len(month_data) > 0:
                month_diffs = month_data.index.to_series().diff()
                hour_skips = month_diffs[month_diffs == pd.Timedelta(hours=1)]
                if len(hour_skips) > 0:
                    warn(
                        f"POSSIBLE DST ISSUE in {month_name}: Found {len(hour_skips)} 1-hour gaps. "
                        "Verify data is truly UTC, not local time with DST shifts."
                    )
        
        # ========== 8. SPREAD ESTIMATION (if bid/ask not available) ==========
        if not self.metadata.spread_available:
            # Estimate spread from high-low range
            typical_range = (df["high"] - df["low"]).median()
            typical_range_pips = typical_range * 10000 if "JPY" not in pair else typical_range * 100
            
            report["statistics"]["estimated_median_range_pips"] = typical_range_pips
            warn(
                f"NO SPREAD DATA: Estimated median bar range: {typical_range_pips:.1f} pips. "
                f"Actual spread is UNKNOWN. For {pair}, expect 0.5-3 pips normal, 5-20+ during news."
            )
        
        # ========== 9. PRICE PRECISION CHECK ==========
        # FX prices should have 4-5 decimal places (2-3 for JPY)
        sample_price = df["close"].iloc[0]
        decimals = len(str(sample_price).split(".")[-1]) if "." in str(sample_price) else 0
        
        expected_decimals = 3 if "JPY" in pair else 5
        if decimals < expected_decimals - 1:
            warn(
                f"LOW PRECISION: Prices have {decimals} decimals, expected {expected_decimals}. "
                "Data may be rounded, affecting precision."
            )
        report["statistics"]["price_decimals"] = decimals
        
        # ========== FINAL REPORT ==========
        report["passed"] = len(report["errors"]) == 0
        report["warning_count"] = len(report["warnings"])
        
        return report
    
    def print_integrity_report(self, pair: str):
        """Print formatted integrity report for a pair."""
        if pair not in self.integrity_report:
            print(f"No integrity report for {pair}. Load data first.")
            return
            
        report = self.integrity_report[pair]
        
        print("=" * 60)
        print(f"FX DATA INTEGRITY REPORT: {pair}")
        print("=" * 60)
        print(f"Source: {self.metadata.source}")
        print(f"Price Type: {self.metadata.price_type.upper()}")
        print(f"Timezone: {self.metadata.timezone}")
        print(f"Volume Type: {self.metadata.volume_type}")
        print("-" * 60)
        print(f"Rows: {report['rows']:,}")
        print(f"Period: {report['start']} to {report['end']}")
        print("-" * 60)
        
        print("\n📊 STATISTICS:")
        for key, value in report["statistics"].items():
            print(f"  {key}: {value}")
        
        if report["warnings"]:
            print(f"\n⚠️ WARNINGS ({len(report['warnings'])}):")
            for w in report["warnings"]:
                print(f"  - {w}")
                
        if report["errors"]:
            print(f"\n❌ ERRORS ({len(report['errors'])}):")
            for e in report["errors"]:
                print(f"  - {e}")
                
        status = "✅ PASSED" if report["passed"] else "❌ FAILED"
        print(f"\n{status} (with {report['warning_count']} warnings)")
        print("=" * 60)
        
        # Print execution realism warning
        print("\n" + self.metadata.get_execution_realism_warning())
    
    def get_data_summary(self, df: pd.DataFrame, pair: str) -> Dict:
        """
        Get summary statistics for loaded data.
        
        This is DESCRIPTIVE only - no cleaning or modification.
        """
        summary = {
            "pair": pair,
            "total_bars": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "trading_days": df.index.normalize().nunique(),
            },
            "price_stats": {
                "min": df["low"].min(),
                "max": df["high"].max(),
                "mean_close": df["close"].mean(),
                "std_close": df["close"].std(),
            },
            "sessions": {
                "london_bars": (df["primary_session"] == "london").sum(),
                "new_york_bars": (df["primary_session"] == "new_york").sum(),
                "tokyo_bars": (df["primary_session"] == "tokyo").sum(),
                "sydney_bars": (df["primary_session"] == "sydney").sum(),
                "overlap_bars": df["is_overlap"].sum(),
            },
            "metadata": self.metadata.to_dict(),
        }
        
        if "volume" in df.columns:
            summary["volume_stats"] = {
                "total": df["volume"].sum(),
                "mean": df["volume"].mean(),
                "median": df["volume"].median(),
                "zero_count": (df["volume"] == 0).sum(),
            }
            
        return summary
    
    def save_integrity_report(self, pair: str, output_dir: str = "results/metrics"):
        """Save integrity report to JSON."""
        if pair not in self.integrity_report:
            raise ValueError(f"No integrity report for {pair}")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{pair}_integrity_report.json"
        
        with open(filepath, "w") as f:
            json.dump(self.integrity_report[pair], f, indent=2, default=str)
            
        print(f"Saved integrity report to {filepath}")


def create_sample_fx_data(
    pair: str,
    output_dir: str = "data/raw/fx",
    days: int = 30,
    bar_interval_minutes: int = 1
) -> str:
    """
    Create sample FX data for testing.
    
    THIS IS SYNTHETIC DATA - NOT FOR REAL ANALYSIS.
    Use only for testing the pipeline.
    
    The data intentionally includes:
    - Weekend gaps
    - Some zero-range candles
    - A few extreme moves
    - Realistic price levels
    """
    import numpy as np
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Realistic starting prices
    start_prices = {
        "EURUSD": 1.0850,
        "GBPUSD": 1.2650,
        "USDJPY": 149.50,
        "AUDUSD": 0.6550,
        "USDCAD": 1.3550,
        "USDCHF": 0.8850,
    }
    
    base_price = start_prices.get(pair, 1.0000)
    
    # Generate timestamps (skip weekends)
    start_date = pd.Timestamp("2024-01-01", tz="UTC")
    end_date = start_date + pd.Timedelta(days=days)
    
    timestamps = []
    current = start_date
    
    while current < end_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.dayofweek < 5:
            timestamps.append(current)
        current += pd.Timedelta(minutes=bar_interval_minutes)
    
    n_bars = len(timestamps)
    
    # Generate realistic price movements
    np.random.seed(42)  # Reproducibility
    
    # Base volatility (annualized ~10%, scaled to 1-min)
    vol = 0.10 / np.sqrt(252 * 24 * 60)
    
    returns = np.random.normal(0, vol, n_bars)
    
    # Add a few extreme moves (flash crashes, news)
    extreme_indices = np.random.choice(n_bars, size=5, replace=False)
    returns[extreme_indices] = np.random.choice([-1, 1], size=5) * np.random.uniform(0.005, 0.015, size=5)
    
    # Generate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    noise = np.random.uniform(0.0001, 0.0005, n_bars)
    
    opens = prices
    closes = prices * (1 + np.random.normal(0, vol/2, n_bars))
    highs = np.maximum(opens, closes) * (1 + noise)
    lows = np.minimum(opens, closes) * (1 - noise)
    
    # Add some zero-range candles (illiquid periods)
    zero_range_indices = np.random.choice(n_bars, size=int(n_bars * 0.005), replace=False)
    highs[zero_range_indices] = opens[zero_range_indices]
    lows[zero_range_indices] = opens[zero_range_indices]
    closes[zero_range_indices] = opens[zero_range_indices]
    
    # Generate tick volume (NOT real volume)
    volume = np.random.poisson(100, n_bars)
    volume[zero_range_indices] = 0  # Zero volume on zero-range candles
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.round(opens, 5 if "JPY" not in pair else 3),
        "high": np.round(highs, 5 if "JPY" not in pair else 3),
        "low": np.round(lows, 5 if "JPY" not in pair else 3),
        "close": np.round(closes, 5 if "JPY" not in pair else 3),
        "volume": volume,
    })
    
    # Save to CSV
    filepath = output_path / f"{pair}.csv"
    df.to_csv(filepath, index=False)
    
    print(f"Created sample data: {filepath}")
    print(f"  Bars: {n_bars:,}")
    print(f"  Period: {timestamps[0]} to {timestamps[-1]}")
    print(f"  ⚠️ THIS IS SYNTHETIC DATA - NOT FOR REAL ANALYSIS")
    
    return str(filepath)


# ==============================================================================
# MANDATORY FX VERIFICATION QUESTIONS
# ==============================================================================
"""
Before using ANY FX data, you MUST answer these questions:

1. WHAT IS THE DATA SOURCE AND BROKER?
   - Source: HistData.com (default), Dukascopy, TrueFX, or broker export
   - If broker export: Which broker? OANDA? IC Markets? FXCM?
   - Different brokers have DIFFERENT prices (1-5 pip differences are normal)

2. IS THIS BID, ASK, OR MID?
   - HistData.com: BID prices
   - You CANNOT execute at bid on a buy order
   - If mid: ALL fills are fantasy without spread model

3. WHAT TIMEZONE IS THE DATA IN?
   - Must be UTC (or explicit offset)
   - Many broker exports are in broker's local time (EST, GMT, etc.)
   - If not UTC: CONVERT BEFORE LOADING

4. HOW IS DST HANDLED?
   - UTC-native: No DST shifts (correct)
   - Local-shifted: DST transitions cause 1-hour gaps/overlaps (problematic)
   - HistData.com: UTC-native (correct)

5. IS VOLUME REAL OR TICK VOLUME?
   - FX has NO centralized exchange = NO real volume
   - Tick volume = number of price updates (NOT traded volume)
   - Tick volume is BROKER-SPECIFIC

6. WHAT DOES TICK VOLUME IMPLY?
   - High tick volume ≠ high traded volume
   - It means price updated frequently (active market)
   - It says NOTHING about actual liquidity

7. ARE SPREADS AVAILABLE?
   - If no: Must model spreads
   - Typical EUR/USD: 0.5-2 pips (normal), 5-20+ pips (news/illiquid)
   - Without spreads: backtest fills are fantasy

8. WHAT FX-SPECIFIC BIASES EXIST?
   - Rollover: Daily swap points at 5PM EST (21:00 UTC)
   - Session gaps: Lower liquidity = wider spreads in Asia for EUR/USD
   - Broker filtering: Some brokers filter extreme ticks (missing flash crashes)
   - Weekend gaps: Market closed Fri 21:00 - Sun 21:00 UTC

If ANY answer is vague → DO NOT PROCEED TO MODELING
"""
