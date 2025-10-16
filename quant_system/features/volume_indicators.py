"""
Volume Indicators
================

Volume-based indicators including OBV, volume-weighted metrics, and order flow analysis.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VolumeIndicators:
    """
    Volume-based indicators and metrics.
    
    All methods are static and deterministic.
    """
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.
        
        Args:
            close: Close prices
            volume: Volume series
            
        Returns:
            OBV series
        """
        price_change = close.diff()
        obv = volume.copy()
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        
        return obv.cumsum()
        
    @staticmethod
    def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Volume Simple Moving Average.
        
        Args:
            volume: Volume series
            window: Window size
            
        Returns:
            Volume SMA series
        """
        return volume.rolling(window=window).mean()
        
    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Volume ratio (current volume / average volume).
        
        Args:
            volume: Volume series
            window: Window size for average calculation
            
        Returns:
            Volume ratio series
        """
        avg_volume = volume.rolling(window=window).mean()
        return volume / avg_volume
        
    @staticmethod
    def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Price Trend (VPT).
        
        Args:
            close: Close prices
            volume: Volume series
            
        Returns:
            VPT series
        """
        price_change_pct = close.pct_change()
        vpt = (price_change_pct * volume).cumsum()
        
        return vpt
        
    @staticmethod
    def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series,
                        window: int = 14) -> pd.Series:
        """
        Ease of Movement.
        
        Args:
            high: High prices
            low: Low prices
            volume: Volume series
            window: Window size for smoothing
            
        Returns:
            Ease of Movement series
        """
        distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_height = volume / (high - low)
        
        eom = distance_moved / box_height
        eom_smoothed = eom.rolling(window=window).mean()
        
        return eom_smoothed
        
    @staticmethod
    def volume_weighted_average_price_deviation(close: pd.Series, vwap: pd.Series) -> pd.Series:
        """
        Deviation from VWAP.
        
        Args:
            close: Close prices
            vwap: VWAP series
            
        Returns:
            VWAP deviation series (as percentage)
        """
        return ((close - vwap) / vwap) * 100
        
    @staticmethod
    def volume_oscillator(volume: pd.Series, short_window: int = 5,
                         long_window: int = 10) -> pd.Series:
        """
        Volume Oscillator.
        
        Args:
            volume: Volume series
            short_window: Short-term window
            long_window: Long-term window
            
        Returns:
            Volume oscillator series
        """
        short_ma = volume.rolling(window=short_window).mean()
        long_ma = volume.rolling(window=long_window).mean()
        
        return ((short_ma - long_ma) / long_ma) * 100
        
    @staticmethod
    def accumulation_distribution(high: pd.Series, low: pd.Series, 
                                close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume series
            
        Returns:
            A/D line series
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        
        ad = (clv * volume).cumsum()
        
        return ad
        
    @staticmethod
    def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Chaikin Money Flow.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume series
            window: Window size
            
        Returns:
            CMF series
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        
        cmf = (clv * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
        
        return cmf
        
    @staticmethod
    def volume_rate_of_change(volume: pd.Series, window: int = 10) -> pd.Series:
        """
        Volume Rate of Change.
        
        Args:
            volume: Volume series
            window: Window size
            
        Returns:
            Volume ROC series (as percentage)
        """
        return ((volume - volume.shift(window)) / volume.shift(window)) * 100
        
    @staticmethod
    def volume_weighted_standard_deviation(close: pd.Series, volume: pd.Series,
                                         window: int = 20) -> pd.Series:
        """
        Volume-weighted standard deviation of prices.
        
        Args:
            close: Close prices
            volume: Volume series
            window: Window size
            
        Returns:
            Volume-weighted standard deviation series
        """
        def weighted_std(values, weights):
            if len(values) == 0 or weights.sum() == 0:
                return np.nan
            weighted_mean = np.average(values, weights=weights)
            return np.sqrt(np.average((values - weighted_mean)**2, weights=weights))
        
        return close.rolling(window=window).apply(
            lambda x: weighted_std(x.values, volume.iloc[x.index].values), raw=False
        )
        
    @staticmethod
    def volume_profile(high: pd.Series, low: pd.Series, volume: pd.Series,
                      price_bins: int = 20) -> pd.Series:
        """
        Volume profile (volume at each price level).
        
        Args:
            high: High prices
            low: Low prices
            volume: Volume series
            price_bins: Number of price bins
            
        Returns:
            Volume profile series
        """
        # Create price bins
        price_range = high.max() - low.min()
        bin_size = price_range / price_bins
        
        # Assign each bar to a price bin
        price_levels = np.arange(low.min(), high.max() + bin_size, bin_size)
        
        # Calculate volume at each price level
        volume_profile = pd.Series(0.0, index=price_levels[:-1])
        
        for i in range(len(high)):
            bar_volume = volume.iloc[i]
            bar_high = high.iloc[i]
            bar_low = low.iloc[i]
            
            # Distribute volume across price bins
            for j in range(len(price_levels) - 1):
                bin_low = price_levels[j]
                bin_high = price_levels[j + 1]
                
                # Calculate overlap between bar and bin
                overlap = max(0, min(bar_high, bin_high) - max(bar_low, bin_low))
                if overlap > 0:
                    volume_profile.iloc[j] += bar_volume * (overlap / (bar_high - bar_low))
        
        return volume_profile
        
    @staticmethod
    def bid_ask_imbalance(bid_volume: pd.Series, ask_volume: pd.Series,
                         window: int = 20) -> pd.Series:
        """
        Bid-ask volume imbalance.
        
        Args:
            bid_volume: Bid volume series
            ask_volume: Ask volume series
            window: Window size for smoothing
            
        Returns:
            Bid-ask imbalance series (-1 to 1)
        """
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume
        imbalance = imbalance.fillna(0)
        
        # Smooth the imbalance
        return imbalance.rolling(window=window).mean()
        
    @staticmethod
    def volume_weighted_momentum(close: pd.Series, volume: pd.Series,
                               window: int = 20) -> pd.Series:
        """
        Volume-weighted momentum.
        
        Args:
            close: Close prices
            volume: Volume series
            window: Window size
            
        Returns:
            Volume-weighted momentum series
        """
        price_change = close.diff()
        volume_weighted_change = (price_change * volume).rolling(window=window).sum()
        total_volume = volume.rolling(window=window).sum()
        
        return volume_weighted_change / total_volume