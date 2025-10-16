"""
Data Loader
===========

Handles loading market data from various sources and formats.
Supports CSV, Parquet, and other common data formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads market data from various file formats and sources.
    
    Supports:
    - CSV files with various column formats
    - Parquet files (preferred for performance)
    - Multiple timeframes and instruments
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, file_path: Union[str, Path], 
                 columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            columns: Optional column mapping for standardization
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
            
            if columns:
                df = df.rename(columns=columns)
                
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # Convert to UTC if timezone-naive
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz != timezone.utc:
                df.index = df.index.tz_convert('UTC')
                
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise
            
    def load_parquet(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_parquet(file_path)
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # Convert to UTC if timezone-naive
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz != timezone.utc:
                df.index = df.index.tz_convert('UTC')
                
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Parquet {file_path}: {e}")
            raise
            
    def load_tick_data(self, instrument: str, date: str) -> pd.DataFrame:
        """
        Load tick data for specific instrument and date.
        
        Args:
            instrument: Instrument symbol (e.g., 'EURUSD')
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with tick data
        """
        file_path = self.data_path / "tick" / instrument / f"{date}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No tick data found for {instrument} on {date}")
            
        return self.load_parquet(file_path)
        
    def load_ohlcv(self, instrument: str, timeframe: str, 
                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load OHLCV data for specific instrument and timeframe.
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.data_path / "ohlcv" / instrument / f"{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No OHLCV data found for {instrument} {timeframe}")
            
        df = self.load_parquet(file_path)
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        
        return df[(df.index >= start_dt) & (df.index <= end_dt)]
        
    def list_instruments(self) -> List[str]:
        """
        List available instruments.
        
        Returns:
            List of instrument symbols
        """
        instruments = []
        
        # Check OHLCV data
        ohlcv_path = self.data_path / "ohlcv"
        if ohlcv_path.exists():
            instruments.extend([d.name for d in ohlcv_path.iterdir() if d.is_dir()])
            
        # Check tick data
        tick_path = self.data_path / "tick"
        if tick_path.exists():
            tick_instruments = [d.name for d in tick_path.iterdir() if d.is_dir()]
            instruments.extend(tick_instruments)
            
        return sorted(list(set(instruments)))
        
    def list_timeframes(self, instrument: str) -> List[str]:
        """
        List available timeframes for instrument.
        
        Args:
            instrument: Instrument symbol
            
        Returns:
            List of available timeframes
        """
        ohlcv_path = self.data_path / "ohlcv" / instrument
        
        if not ohlcv_path.exists():
            return []
            
        timeframes = []
        for file_path in ohlcv_path.glob("*.parquet"):
            timeframes.append(file_path.stem)
            
        return sorted(timeframes)