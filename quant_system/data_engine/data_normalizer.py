"""
Data Normalizer
===============

Standardizes data formats, timestamps, and column names across different data sources.
Ensures consistent data structure for downstream processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalizes market data to standard format.
    
    Standardizes:
    - Column names across different sources
    - Timestamp formats and timezones
    - Data types and missing value handling
    - Price and volume data validation
    """
    
    # Standard column mappings for different data sources
    COLUMN_MAPPINGS = {
        'mt5': {
            'time': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        },
        'oanda': {
            'time': 'timestamp',
            'openMid': 'open',
            'highMid': 'high',
            'lowMid': 'low', 
            'closeMid': 'close',
            'volume': 'volume'
        },
        'generic': {
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'bid': 'bid',
            'ask': 'ask',
            'spread': 'spread'
        }
    }
    
    def __init__(self, source_type: str = 'generic'):
        """
        Initialize DataNormalizer.
        
        Args:
            source_type: Type of data source ('mt5', 'oanda', 'generic')
        """
        self.source_type = source_type
        self.column_mapping = self.COLUMN_MAPPINGS.get(source_type, self.COLUMN_MAPPINGS['generic'])
        
    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to standard format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized column names
        """
        # Create reverse mapping for flexibility
        reverse_mapping = {v: k for k, v in self.column_mapping.items()}
        
        # Rename columns
        df_normalized = df.copy()
        df_normalized = df_normalized.rename(columns=reverse_mapping)
        
        logger.debug(f"Normalized columns: {list(df.columns)} -> {list(df_normalized.columns)}")
        return df_normalized
        
    def normalize_timestamps(self, df: pd.DataFrame, 
                           timezone_str: str = 'UTC') -> pd.DataFrame:
        """
        Normalize timestamps to UTC timezone.
        
        Args:
            df: Input DataFrame with datetime index
            timezone_str: Source timezone if not already timezone-aware
            
        Returns:
            DataFrame with UTC timestamps
        """
        df_normalized = df.copy()
        
        if not isinstance(df_normalized.index, pd.DatetimeIndex):
            df_normalized.index = pd.to_datetime(df_normalized.index)
            
        # Handle timezone conversion
        if df_normalized.index.tz is None:
            df_normalized.index = df_normalized.index.tz_localize(timezone_str)
        elif df_normalized.index.tz != timezone.utc:
            df_normalized.index = df_normalized.index.tz_convert('UTC')
            
        # Sort by timestamp
        df_normalized = df_normalized.sort_index()
        
        logger.debug(f"Normalized timestamps to UTC, {len(df_normalized)} rows")
        return df_normalized
        
    def normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize OHLCV data format and validate prices.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Normalized OHLCV DataFrame
        """
        df_normalized = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_normalized.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert to numeric, handling any non-numeric values
        for col in required_cols:
            df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
            
        # Validate OHLC relationships
        invalid_ohlc = (
            (df_normalized['high'] < df_normalized['low']) |
            (df_normalized['high'] < df_normalized['open']) |
            (df_normalized['high'] < df_normalized['close']) |
            (df_normalized['low'] > df_normalized['open']) |
            (df_normalized['low'] > df_normalized['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
            # Optionally fix or remove invalid rows
            df_normalized = df_normalized[~invalid_ohlc]
            
        # Ensure volume is non-negative
        df_normalized['volume'] = df_normalized['volume'].clip(lower=0)
        
        # Calculate spread if bid/ask available
        if 'bid' in df_normalized.columns and 'ask' in df_normalized.columns:
            df_normalized['spread'] = df_normalized['ask'] - df_normalized['bid']
            
        logger.debug(f"Normalized OHLCV data, {len(df_normalized)} valid rows")
        return df_normalized
        
    def normalize_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize tick data format.
        
        Args:
            df: Input DataFrame with tick data
            
        Returns:
            Normalized tick DataFrame
        """
        df_normalized = df.copy()
        
        # Ensure required columns exist
        required_cols = ['bid', 'ask']
        missing_cols = [col for col in required_cols if col not in df_normalized.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns for tick data: {missing_cols}")
            
        # Convert to numeric
        for col in ['bid', 'ask', 'bid_size', 'ask_size']:
            if col in df_normalized.columns:
                df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
                
        # Calculate mid price and spread
        df_normalized['mid'] = (df_normalized['bid'] + df_normalized['ask']) / 2
        df_normalized['spread'] = df_normalized['ask'] - df_normalized['bid']
        
        # Validate bid-ask relationship
        invalid_spread = df_normalized['spread'] < 0
        if invalid_spread.any():
            logger.warning(f"Found {invalid_spread.sum()} invalid bid-ask spreads")
            df_normalized = df_normalized[~invalid_spread]
            
        logger.debug(f"Normalized tick data, {len(df_normalized)} valid ticks")
        return df_normalized
        
    def normalize(self, df: pd.DataFrame, data_type: str = 'ohlcv') -> pd.DataFrame:
        """
        Complete normalization pipeline.
        
        Args:
            df: Input DataFrame
            data_type: Type of data ('ohlcv', 'tick')
            
        Returns:
            Fully normalized DataFrame
        """
        logger.info(f"Starting normalization for {data_type} data")
        
        # Step 1: Normalize columns
        df_normalized = self.normalize_columns(df)
        
        # Step 2: Normalize timestamps
        df_normalized = self.normalize_timestamps(df_normalized)
        
        # Step 3: Data-specific normalization
        if data_type == 'ohlcv':
            df_normalized = self.normalize_ohlcv(df_normalized)
        elif data_type == 'tick':
            df_normalized = self.normalize_tick_data(df_normalized)
        else:
            logger.warning(f"Unknown data type: {data_type}, skipping data-specific normalization")
            
        # Step 4: Remove duplicates and sort
        df_normalized = df_normalized[~df_normalized.index.duplicated(keep='last')]
        df_normalized = df_normalized.sort_index()
        
        logger.info(f"Normalization complete: {len(df_normalized)} rows, {len(df_normalized.columns)} columns")
        return df_normalized