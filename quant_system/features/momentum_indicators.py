"""
Momentum Indicators
==================

Trend and momentum indicators including RSI, MACD, and trend strength measures.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """
    Momentum and trend indicators.
    
    All methods are static and deterministic.
    """
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    @staticmethod
    def macd(prices: pd.Series, fast_window: int = 12, 
             slow_window: int = 26, signal_window: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_window: Fast EMA window
            slow_window: Slow EMA window
            signal_window: Signal line window
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm(span=fast_window).mean()
        ema_slow = prices.ewm(span=slow_window).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_window).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                  k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_window: %K window
            d_window: %D window
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return k_percent, d_percent
        
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                  window: int = 14) -> pd.Series:
        """
        Williams %R.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size
            
        Returns:
            Williams %R series
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
        
    @staticmethod
    def momentum(prices: pd.Series, window: int = 10) -> pd.Series:
        """
        Price momentum.
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            Momentum series
        """
        return prices - prices.shift(window)
        
    @staticmethod
    def rate_of_change(prices: pd.Series, window: int = 10) -> pd.Series:
        """
        Rate of Change (ROC).
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            ROC series (as percentage)
        """
        return ((prices - prices.shift(window)) / prices.shift(window)) * 100
        
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series,
                               window: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size
            
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci
        
    @staticmethod
    def trend_strength(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Trend strength indicator.
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            Trend strength series (1 = strong uptrend, -1 = strong downtrend, 0 = sideways)
        """
        # Calculate linear regression slope
        def linear_regression_slope(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return 0
            return np.polyfit(x, y, 1)[0]
        
        slopes = prices.rolling(window=window).apply(linear_regression_slope, raw=False)
        
        # Normalize slopes to -1 to 1 range
        max_slope = slopes.rolling(window=window*2).std() * 2
        trend_strength = slopes / max_slope
        trend_strength = trend_strength.clip(-1, 1)
        
        return trend_strength
        
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (ADX).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size
            
        Returns:
            Tuple of (ADX, +DI, -DI, DX)
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        
        # Calculate DI
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = dx.rolling(window=window).mean()
        
        return adx, di_plus, di_minus, dx
        
    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Aroon Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            window: Window size
            
        Returns:
            Tuple of (Aroon Up, Aroon Down)
        """
        def aroon_up(series):
            return series.rolling(window=window).apply(
                lambda x: (window - x.argmax()) / window * 100, raw=False
            )
        
        def aroon_down(series):
            return series.rolling(window=window).apply(
                lambda x: (window - x.argmin()) / window * 100, raw=False
            )
        
        aroon_up_val = aroon_up(high)
        aroon_down_val = aroon_down(low)
        
        return aroon_up_val, aroon_down_val
        
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                        volume: pd.Series, window: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume series
            window: Window size
            
        Returns:
            MFI series (0-100)
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(window=window).sum()
        negative_flow_sum = negative_flow.rolling(window=window).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi