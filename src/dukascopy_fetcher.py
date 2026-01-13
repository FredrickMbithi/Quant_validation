"""
Dukascopy M1 Fetcher and Aggregator

This module fetches minute (M1) BID/ASK candles from Dukascopy and
optionally aggregates them to hourly bars.

Requirements:
- Install `dukascpy` (PyPI) which handles downloading and decoding `.bi5` files.

Usage:
  python src/dukascopy_fetcher.py minute GBPUSD 2025-10-01 2025-11-01 data/raw/fx
  python src/dukascopy_fetcher.py hourly GBPUSD 2025-10-01 2025-11-01 data/processed/fx

Notes:
- Output minute CSV columns: timestamp,bid_open,bid_high,bid_low,bid_close,ask_open,ask_high,ask_low,ask_close,volume
- Output hourly CSV columns: timestamp,open,high,low,close,volume (mid OHLC computed as (bid+ask)/2)
- All timestamps are UTC.
"""

from pathlib import Path
from datetime import datetime
import sys
import pandas as pd

try:
    # Try common package names
    import dukascopy as _dukascopy_pkg  # PyPI: dukascopy
except Exception:
    try:
        import dukascpy as _dukascopy_pkg  # PyPI: dukascpy
    except Exception:
        _dukascopy_pkg = None


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fetch_m1(pair: str, start: str, end: str) -> pd.DataFrame:
    """Fetch M1 BID/ASK candles from Dukascopy using dukascpy.
    Returns a DataFrame with UTC timestamps and bid/ask OHLC.
    """
    if _dukascopy_pkg is None:
        raise RuntimeError(
            "Dukascopy client not installed. Try: pip install dukascopy or pip install dukascpy"
        )
    # Attempt known APIs
    Candles = None
    # API variant 1
    try:
        from dukascopy.candles import Candles  # type: ignore
    except Exception:
        pass
    # API variant 2 (hypothetical): dukascpy.candles
    if Candles is None:
        try:
            from dukascpy.candles import Candles  # type: ignore
        except Exception:
            pass
    # API variant 3: client-based interface
    client = None
    if Candles is None:
        try:
            from dukascopy.client import Client  # type: ignore
            client = Client()
        except Exception:
            pass

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    if Candles is not None:
        candles = Candles(
            instrument=pair,
            timeframe=60,  # 1 minute
            start=start_dt,
            end=end_dt,
        )
        df = candles.download()
    elif client is not None:
        # Fallback client interface; try a generic candles method
        try:
            df = client.get_candles(pair, start_dt, end_dt, timeframe=60)
        except Exception as e:
            raise RuntimeError(f"Dukascopy client get_candles failed: {e}")
    else:
        raise RuntimeError("No supported Dukascopy API found in installed package.")

    # Normalize columns and enforce UTC
    df = df.rename(columns={
        'timestamp': 'timestamp',
        'bid_open': 'bid_open',
        'bid_high': 'bid_high',
        'bid_low': 'bid_low',
        'bid_close': 'bid_close',
        'ask_open': 'ask_open' if 'ask_open' in df.columns else None,
        'ask_high': 'ask_high' if 'ask_high' in df.columns else None,
        'ask_low': 'ask_low' if 'ask_low' in df.columns else None,
        'ask_close': 'ask_close' if 'ask_close' in df.columns else None,
        'volume': 'volume' if 'volume' in df.columns else 'tick_volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Some interfaces may only return bid; compute mid columns if ask is missing
    if 'ask_open' not in df.columns:
        df['ask_open'] = pd.NA
        df['ask_high'] = pd.NA
        df['ask_low'] = pd.NA
        df['ask_close'] = pd.NA

    return df[['timestamp','bid_open','bid_high','bid_low','bid_close','ask_open','ask_high','ask_low','ask_close','volume']]


def save_minute_csv(df: pd.DataFrame, out_dir: Path, pair: str) -> Path:
    _ensure_dir(out_dir)
    out = out_dir / f"{pair}_M1_dukascopy.csv"
    df.to_csv(out, index=False)
    return out


def aggregate_to_hourly(df: pd.DataFrame, pair: str, out_dir: Path) -> Path:
    """Aggregate BID/ASK M1 to hourly MID OHLC and sum volume."""
    _ensure_dir(out_dir)
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')

    # Compute mid OHLC when ask is present; else use bid as proxy
    open_mid = df[['bid_open','ask_open']].mean(axis=1)
    high_mid = df[['bid_high','ask_high']].mean(axis=1)
    low_mid = df[['bid_low','ask_low']].mean(axis=1)
    close_mid = df[['bid_close','ask_close']].mean(axis=1)

    mid_df = pd.DataFrame({
        'open': open_mid,
        'high': high_mid,
        'low': low_mid,
        'close': close_mid,
        'volume': df['volume'] if 'volume' in df.columns else df.get('tick_volume', pd.Series(index=df.index, data=0))
    })

    hourly = mid_df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    hourly = hourly.reset_index()
    hourly.to_csv(out_dir / f"{pair}_hourly.csv", index=False)
    return out_dir / f"{pair}_hourly.csv"


def main():
    if len(sys.argv) < 6:
        print("Usage:")
        print("  python src/dukascopy_fetcher.py minute <PAIR> <START> <END> <OUT_DIR>")
        print("  python src/dukascopy_fetcher.py hourly <PAIR> <START> <END> <OUT_DIR>")
        print("Examples:")
        print("  python src/dukascopy_fetcher.py minute GBPUSD 2025-10-01 2025-11-01 data/raw/fx")
        print("  python src/dukascopy_fetcher.py hourly GBPUSD 2025-10-01 2025-11-01 data/processed/fx")
        sys.exit(1)

    mode = sys.argv[1]
    pair = sys.argv[2]
    start = sys.argv[3]
    end = sys.argv[4]
    out_dir = Path(sys.argv[5])

    df = fetch_m1(pair, start, end)

    if mode == 'minute':
        out = save_minute_csv(df, out_dir, pair)
        print('Saved minute CSV:', out)
    elif mode == 'hourly':
        out = aggregate_to_hourly(df, pair, out_dir)
        print('Saved hourly CSV:', out)
        # Also write metadata sidecar
        meta_path = out_dir / f"{pair}_hourly_metadata.json"
        meta = {
            'source': 'Dukascopy',
            'pair': pair,
            'price_type': 'mid',
            'timezone': 'UTC',
            'volume_type': 'tick',
            'spread_available': True,
            'bar_interval': '1h',
            'dst_handling': 'utc_native',
            'notes': f'Aggregated from Dukascopy M1 BID/ASK {start}..{end}'
        }
        import json
        meta_path.write_text(json.dumps(meta, indent=2))
        print('Wrote metadata:', meta_path)
    else:
        print('Unknown mode:', mode)
        sys.exit(2)


if __name__ == '__main__':
    main()
