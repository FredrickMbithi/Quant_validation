"""
HistData.com FX Data Downloader & Converter

CRITICAL DATA SOURCE FACTS:
===========================
- Source: HistData.com (aggregated from multiple liquidity providers)
- Price Type: BID prices ONLY (not mid, not ask)
- Timezone: EST (Eastern Standard Time) WITHOUT DST adjustments
- Volume: Tick volume (NOT real traded volume)
- Format: Semicolon-separated (not comma)

TIMEZONE WARNING:
=================
HistData uses EST WITHOUT DST. This means:
- EST = UTC-5 year-round
- This is NOT the same as US Eastern Time (which has DST)
- Must convert to UTC for consistency

CONVERSION: EST (no DST) -> UTC = Add 5 hours

Usage:
------
1. Download manually from https://www.histdata.com/download-free-forex-data/
2. Select: ASCII -> 1-Minute Bar Quotes -> GBPUSD -> Year
3. Extract ZIP to get DAT_ASCII_GBPUSD_M1_YYYY.csv
4. Run this converter to transform to UTC format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import os
from typing import Optional, List


class HistDataConverter:
    """
    Convert HistData.com CSV files to standardized UTC format.
    
    HistData Format (Generic ASCII M1):
    - Delimiter: semicolon (;)
    - Columns: DateTime;Open;High;Low;Close;Volume
    - DateTime format: YYYYMMDD HHMMSS
    - Timezone: EST (UTC-5, no DST)
    - Prices: BID only
    """
    
    # EST without DST = UTC-5 always
    EST_TO_UTC_HOURS = 5
    
    def __init__(self, raw_dir: str = "data/raw/fx", output_dir: str = "data/raw/fx"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_histdata_file(
        self, 
        input_file: str,
        pair: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert a single HistData CSV file to UTC format.
        
        Args:
            input_file: Path to HistData CSV (DAT_ASCII_*_M1_*.csv)
            pair: Currency pair (e.g., "GBPUSD")
            output_file: Output path (default: data/raw/fx/{pair}.csv)
            
        Returns:
            Converted DataFrame
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Converting {input_path.name}...")
        print(f"  Source timezone: EST (UTC-5, no DST)")
        print(f"  Target timezone: UTC")
        
        # Read HistData format (semicolon delimiter, no header)
        df = pd.read_csv(
            input_path,
            delimiter=';',
            header=None,
            names=['datetime_est', 'open', 'high', 'low', 'close', 'volume'],
            dtype={
                'datetime_est': str,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': int
            }
        )
        
        print(f"  Loaded {len(df):,} rows")
        
        # Parse datetime (format: YYYYMMDD HHMMSS)
        df['datetime_est'] = pd.to_datetime(
            df['datetime_est'], 
            format='%Y%m%d %H%M%S'
        )
        
        # Convert EST (UTC-5) to UTC by adding 5 hours
        df['timestamp'] = df['datetime_est'] + pd.Timedelta(hours=self.EST_TO_UTC_HOURS)
        
        # Make timezone-aware
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Drop the EST column
        df = df.drop(columns=['datetime_est'])
        
        # Reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Verify conversion
        print(f"  Date range (UTC): {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Save to output
        if output_file is None:
            output_file = self.output_dir / f"{pair}.csv"
        else:
            output_file = Path(output_file)
            
        # Save with proper format
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        
        return df
    
    def convert_multiple_files(
        self,
        input_files: List[str],
        pair: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert and merge multiple HistData CSV files (e.g., multiple months/years).
        
        Args:
            input_files: List of paths to HistData CSVs
            pair: Currency pair
            output_file: Output path
            
        Returns:
            Merged UTC minute DataFrame
        """
        converted = []
        for f in input_files:
            df = pd.read_csv(
                f,
                delimiter=';',
                header=None,
                names=['datetime_est', 'open', 'high', 'low', 'close', 'volume'],
            )
            df['datetime_est'] = pd.to_datetime(df['datetime_est'], format='%Y%m%d %H%M%S')
            df['timestamp'] = df['datetime_est'] + pd.Timedelta(hours=self.EST_TO_UTC_HOURS)
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df = df.drop(columns=['datetime_est'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            converted.append(df)
        
        combined = pd.concat(converted, ignore_index=True)
        combined = combined.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
        
        if output_file is None:
            output_file = self.output_dir / f"{pair}_minute_UTC.csv"
        
        combined.to_csv(output_file, index=False)
        print(f"Combined {len(input_files)} files -> {output_file}")
        print(f"Total rows: {len(combined):,}")
        
        return combined

    def resample_to_hourly(self, df_minute: pd.DataFrame, pair: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Resample minute OHLC (UTC) to hourly OHLC without altering raw files.
        
        - Open: first open
        - High: max high
        - Low: min low
        - Close: last close
        - Volume: sum (tick count)
        """
        df = df_minute.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        rule = '1H'
        ohlc = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }
        agg = df.resample(rule).agg({**ohlc, 'volume': 'sum'}).dropna()
        agg = agg.reset_index()
        
        if output_file is None:
            # Save to processed/fx
            output_dir = Path('data/processed/fx')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{pair}_hourly.csv"
        
        agg.to_csv(output_file, index=False)
        print(f"Saved hourly resample: {output_file}")
        return agg

    def build_hourly_from_csvs(self, csv_files: List[str], pair: str) -> str:
        """Convert multiple HistData minute CSVs and save hourly dataset."""
        minute_df = self.convert_multiple_files(csv_files, pair)
        hourly_df = self.resample_to_hourly(minute_df, pair)
        return str(Path('data/processed/fx') / f"{pair}_hourly.csv")
    
    def extract_zip(self, zip_path: str, extract_to: Optional[str] = None) -> List[str]:
        """
        Extract HistData ZIP file.
        
        Args:
            zip_path: Path to downloaded ZIP file
            extract_to: Extraction directory (default: same as ZIP)
            
        Returns:
            List of extracted CSV file paths
        """
        zip_path = Path(zip_path)
        
        if extract_to is None:
            extract_to = zip_path.parent
        else:
            extract_to = Path(extract_to)
            
        extract_to.mkdir(parents=True, exist_ok=True)
        
        extracted = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.csv'):
                    zf.extract(name, extract_to)
                    extracted.append(str(extract_to / name))
                    print(f"Extracted: {name}")
                    
        return extracted


def download_instructions():
    """Print instructions for downloading HistData files."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          HISTDATA.COM DOWNLOAD INSTRUCTIONS                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to: https://www.histdata.com/download-free-forex-data/   ║
║                                                                  ║
║  2. Select:                                                      ║
║     - Platform: ASCII (Generic CSV)                              ║
║     - Timeframe: M1 (1 Minute Bars)                             ║
║     - Pair: GBP/USD (or other)                                  ║
║                                                                  ║
║  3. Download the year(s) you need (e.g., 2024, 2025)            ║
║                                                                  ║
║  4. Save ZIP files to: data/raw/fx/downloads/                   ║
║                                                                  ║
║  5. Run this script to convert EST -> UTC                       ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  CRITICAL DATA FACTS:                                        ║
║      - Prices are BID only (not mid, not ask)                   ║
║      - Timezone is EST (UTC-5) WITHOUT DST                      ║
║      - Volume is TICK count (NOT traded volume)                 ║
║      - Different from live broker feeds                         ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        download_instructions()
        print("\nUsage:")
        print("  python histdata_converter.py <zip_or_csv_file> <pair>")
        print("  python histdata_converter.py data/raw/fx/downloads/HISTDATA_COM_ASCII_GBPUSD_M1_2024.zip GBPUSD")
        print("  python histdata_converter.py hourly <pair> <csv1> <csv2> ...  # build hourly from CSVs")
        sys.exit(0)
    
    input_file = sys.argv[1]
    pair = sys.argv[2] if len(sys.argv) > 2 else "GBPUSD"
    
    converter = HistDataConverter()
    
    if input_file == 'hourly':
        # Build hourly from provided CSV files
        csv_files = sys.argv[3:]
        if not csv_files:
            print("Provide minute CSV files from HistData to build hourly dataset.")
            sys.exit(1)
        converter.build_hourly_from_csvs(csv_files, pair)
    elif input_file.endswith('.zip'):
        # Extract and convert
        csv_files = converter.extract_zip(input_file)
        if csv_files:
            converter.convert_histdata_file(csv_files[0], pair)
    else:
        # Direct CSV conversion
        converter.convert_histdata_file(input_file, pair)
