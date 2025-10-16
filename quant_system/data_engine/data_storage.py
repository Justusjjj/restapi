"""
Data Storage
============

Manages partitioned storage of market data for efficient access.
Uses Parquet format for optimal performance and compression.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Manages partitioned storage of market data.
    
    Features:
    - Parquet format for fast I/O and compression
    - Partitioned by instrument and timeframe
    - Efficient time-based filtering
    - Metadata tracking and versioning
    """
    
    def __init__(self, storage_path: Union[str, Path]):
        """
        Initialize DataStorage.
        
        Args:
            storage_path: Base path for data storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "ohlcv").mkdir(exist_ok=True)
        (self.storage_path / "tick").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        
    def save_ohlcv(self, df: pd.DataFrame, instrument: str, timeframe: str,
                   overwrite: bool = False) -> Path:
        """
        Save OHLCV data to partitioned storage.
        
        Args:
            df: OHLCV DataFrame
            instrument: Instrument symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved file
        """
        # Create instrument directory
        instrument_path = self.storage_path / "ohlcv" / instrument
        instrument_path.mkdir(parents=True, exist_ok=True)
        
        # File path
        file_path = instrument_path / f"{timeframe}.parquet"
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists. Use overwrite=True to replace.")
            
        # Ensure UTC timezone
        if df.index.tz != timezone.utc:
            df = df.copy()
            df.index = df.index.tz_convert('UTC')
            
        # Save as Parquet
        df.to_parquet(file_path, engine='pyarrow', compression='snappy')
        
        # Save metadata
        self._save_metadata(df, file_path, 'ohlcv', instrument, timeframe)
        
        logger.info(f"Saved OHLCV data: {instrument} {timeframe} ({len(df)} rows)")
        return file_path
        
    def save_tick_data(self, df: pd.DataFrame, instrument: str, date: str,
                      overwrite: bool = False) -> Path:
        """
        Save tick data to partitioned storage.
        
        Args:
            df: Tick DataFrame
            instrument: Instrument symbol
            date: Date in YYYY-MM-DD format
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved file
        """
        # Create instrument directory
        instrument_path = self.storage_path / "tick" / instrument
        instrument_path.mkdir(parents=True, exist_ok=True)
        
        # File path
        file_path = instrument_path / f"{date}.parquet"
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists. Use overwrite=True to replace.")
            
        # Ensure UTC timezone
        if df.index.tz != timezone.utc:
            df = df.copy()
            df.index = df.index.tz_convert('UTC')
            
        # Save as Parquet
        df.to_parquet(file_path, engine='pyarrow', compression='snappy')
        
        # Save metadata
        self._save_metadata(df, file_path, 'tick', instrument, date)
        
        logger.info(f"Saved tick data: {instrument} {date} ({len(df)} rows)")
        return file_path
        
    def load_ohlcv(self, instrument: str, timeframe: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data with optional date filtering.
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            OHLCV DataFrame
        """
        file_path = self.storage_path / "ohlcv" / instrument / f"{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No OHLCV data found: {instrument} {timeframe}")
            
        # Load data
        df = pd.read_parquet(file_path)
        
        # Apply date filtering if specified
        if start_date or end_date:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC') if start_date else None
            end_dt = pd.to_datetime(end_date).tz_localize('UTC') if end_date else None
            
            if start_dt:
                df = df[df.index >= start_dt]
            if end_dt:
                df = df[df.index <= end_dt]
                
        logger.debug(f"Loaded OHLCV data: {instrument} {timeframe} ({len(df)} rows)")
        return df
        
    def load_tick_data(self, instrument: str, date: str) -> pd.DataFrame:
        """
        Load tick data for specific date.
        
        Args:
            instrument: Instrument symbol
            date: Date in YYYY-MM-DD format
            
        Returns:
            Tick DataFrame
        """
        file_path = self.storage_path / "tick" / instrument / f"{date}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No tick data found: {instrument} {date}")
            
        df = pd.read_parquet(file_path)
        logger.debug(f"Loaded tick data: {instrument} {date} ({len(df)} rows)")
        return df
        
    def list_instruments(self, data_type: str = 'ohlcv') -> List[str]:
        """
        List available instruments.
        
        Args:
            data_type: Type of data ('ohlcv', 'tick')
            
        Returns:
            List of instrument symbols
        """
        data_path = self.storage_path / data_type
        
        if not data_path.exists():
            return []
            
        instruments = [d.name for d in data_path.iterdir() if d.is_dir()]
        return sorted(instruments)
        
    def list_timeframes(self, instrument: str) -> List[str]:
        """
        List available timeframes for instrument.
        
        Args:
            instrument: Instrument symbol
            
        Returns:
            List of timeframes
        """
        instrument_path = self.storage_path / "ohlcv" / instrument
        
        if not instrument_path.exists():
            return []
            
        timeframes = [f.stem for f in instrument_path.glob("*.parquet")]
        return sorted(timeframes)
        
    def get_data_info(self, instrument: str, data_type: str = 'ohlcv') -> Dict:
        """
        Get information about stored data.
        
        Args:
            instrument: Instrument symbol
            data_type: Type of data ('ohlcv', 'tick')
            
        Returns:
            Dictionary with data information
        """
        if data_type == 'ohlcv':
            timeframes = self.list_timeframes(instrument)
            info = {
                'instrument': instrument,
                'data_type': data_type,
                'timeframes': timeframes,
                'files': {}
            }
            
            for tf in timeframes:
                file_path = self.storage_path / "ohlcv" / instrument / f"{tf}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    info['files'][tf] = {
                        'rows': len(df),
                        'start_date': df.index.min().isoformat(),
                        'end_date': df.index.max().isoformat(),
                        'columns': list(df.columns)
                    }
                    
        elif data_type == 'tick':
            instrument_path = self.storage_path / "tick" / instrument
            if not instrument_path.exists():
                return {'instrument': instrument, 'data_type': data_type, 'files': {}}
                
            files = list(instrument_path.glob("*.parquet"))
            info = {
                'instrument': instrument,
                'data_type': data_type,
                'files': {}
            }
            
            for file_path in files:
                date = file_path.stem
                df = pd.read_parquet(file_path)
                info['files'][date] = {
                    'rows': len(df),
                    'start_time': df.index.min().isoformat(),
                    'end_time': df.index.max().isoformat(),
                    'columns': list(df.columns)
                }
        else:
            raise ValueError(f"Unknown data type: {data_type}")
            
        return info
        
    def _save_metadata(self, df: pd.DataFrame, file_path: Path,
                      data_type: str, instrument: str, identifier: str):
        """Save metadata about the stored data."""
        metadata = {
            'instrument': instrument,
            'data_type': data_type,
            'identifier': identifier,
            'rows': len(df),
            'columns': list(df.columns),
            'start_time': df.index.min().isoformat(),
            'end_time': df.index.max().isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'file_size': file_path.stat().st_size
        }
        
        metadata_path = self.storage_path / "metadata" / f"{instrument}_{data_type}_{identifier}.json"
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def cleanup_old_data(self, days_to_keep: int = 365):
        """
        Clean up old data files.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = datetime.now(timezone.utc) - pd.Timedelta(days=days_to_keep)
        
        # Clean up tick data
        tick_path = self.storage_path / "tick"
        for instrument_dir in tick_path.iterdir():
            if instrument_dir.is_dir():
                for file_path in instrument_dir.glob("*.parquet"):
                    file_date = pd.to_datetime(file_path.stem)
                    if file_date < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Removed old tick data: {file_path}")
                        
        logger.info(f"Cleanup complete: removed data older than {days_to_keep} days")