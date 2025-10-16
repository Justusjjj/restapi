"""
Price Indicators
===============

Price-based technical indicators including moving averages, VWAP, and price statistics.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PriceIndicators:
    """
    Price-based technical indicators.
    
    All methods are static and deterministic.
    """
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            SMA series
        """
        return prices.rolling(window=window).mean()
        
    @staticmethod
    def ema(prices: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            prices: Price series
            window: Window size
            alpha: Smoothing factor (if None, calculated as 2/(window+1))
            
        Returns:
            EMA series
        """
        if alpha is None:
            alpha = 2.0 / (window + 1)
        return prices.ewm(alpha=alpha).mean()
        
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
             volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
        
    @staticmethod
    def intraday_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, session_start: str = '00:00') -> pd.Series:
        """
        Intraday VWAP (resets each day).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            session_start: Session start time (HH:MM format)
            
        Returns:
            Intraday VWAP series
        """
        typical_price = (high + low + close) / 3
        
        # Group by date and calculate VWAP for each day
        def daily_vwap(group):
            return (group['typical_price'] * group['volume']).cumsum() / group['volume'].cumsum()
            
        df = pd.DataFrame({
            'typical_price': typical_price,
            'volume': volume
        })
        
        # Add date column for grouping
        df['date'] = df.index.date
        
        # Calculate daily VWAP
        vwap = df.groupby('date').apply(daily_vwap)
        
        # Flatten the result and align with original index
        vwap = vwap.droplevel(0)
        vwap.index = typical_price.index
        
        return vwap
        
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, 
                       num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            prices: Price series
            window: Window size
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
        
    @staticmethod
    def price_zscore(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Price Z-score (standardized price).
        
        Args:
            prices: Price series
            window: Window size for mean and std calculation
            
        Returns:
            Z-score series
        """
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        return (prices - rolling_mean) / rolling_std
        
    @staticmethod
    def price_deviation_from_vwap(prices: pd.Series, vwap: pd.Series) -> pd.Series:
        """
        Price deviation from VWAP as percentage.
        
        Args:
            prices: Price series
            vwap: VWAP series
            
        Returns:
            Deviation percentage series
        """
        return ((prices - vwap) / vwap) * 100
        
    @staticmethod
    def price_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
        """
        Calculate price returns.
        
        Args:
            prices: Price series
            method: Return calculation method ('log' or 'pct')
            
        Returns:
            Returns series
        """
        if method == 'log':
            return np.log(prices / prices.shift(1))
        elif method == 'pct':
            return prices.pct_change()
        else:
            raise ValueError("Method must be 'log' or 'pct'")
            
    @staticmethod
    def rolling_percentile(prices: pd.Series, window: int, 
                          percentile: float = 50) -> pd.Series:
        """
        Rolling percentile of prices.
        
        Args:
            prices: Price series
            window: Window size
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Rolling percentile series
        """
        return prices.rolling(window=window).quantile(percentile / 100)
        
    @staticmethod
    def price_position_in_range(high: pd.Series, low: pd.Series, 
                               close: pd.Series, window: int = 20) -> pd.Series:
        """
        Position of close price within high-low range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size for rolling high/low
            
        Returns:
            Position series (0 = at low, 1 = at high)
        """
        rolling_high = high.rolling(window=window).max()
        rolling_low = low.rolling(window=window).min()
        
        return (close - rolling_low) / (rolling_high - rolling_low)
        
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, 
                         window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels.
        
        Args:
            high: High prices
            low: Low prices
            window: Window size
            
        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        upper_channel = high.rolling(window=window).max()
        lower_channel = low.rolling(window=window).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return upper_channel, middle_channel, lower_channel