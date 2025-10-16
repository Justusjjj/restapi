"""
Volatility Indicators
====================

Volatility and risk measures including ATR, GARCH, and realized volatility.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VolatilityIndicators:
    """
    Volatility and risk indicators.
    
    All methods are static and deterministic.
    """
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            window: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size
            
        Returns:
            ATR series
        """
        # Calculate True Range components
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        # True Range is the maximum of the three
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR is the moving average of True Range
        return true_range.rolling(window=window).mean()
        
    @staticmethod
    def realized_volatility(returns: pd.Series, window: int = 20, 
                           annualize: bool = True) -> pd.Series:
        """
        Realized volatility (standard deviation of returns).
        
        Args:
            returns: Return series
            window: Window size
            annualize: Whether to annualize the volatility
            
        Returns:
            Realized volatility series
        """
        vol = returns.rolling(window=window).std()
        
        if annualize:
            # Assuming daily returns, multiply by sqrt(252)
            vol = vol * np.sqrt(252)
            
        return vol
        
    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, 
                           window: int = 20, annualize: bool = True) -> pd.Series:
        """
        Parkinson volatility estimator using high-low prices.
        
        Args:
            high: High prices
            low: Low prices
            window: Window size
            annualize: Whether to annualize the volatility
            
        Returns:
            Parkinson volatility series
        """
        # Parkinson estimator: sqrt(1/(4*ln(2)) * mean(ln(H/L)^2))
        log_hl = np.log(high / low)
        parkinson = np.sqrt(1 / (4 * np.log(2)) * log_hl.rolling(window=window).mean())
        
        if annualize:
            parkinson = parkinson * np.sqrt(252)
            
        return parkinson
        
    @staticmethod
    def garman_klass_volatility(open_price: pd.Series, high: pd.Series, 
                               low: pd.Series, close: pd.Series,
                               window: int = 20, annualize: bool = True) -> pd.Series:
        """
        Garman-Klass volatility estimator.
        
        Args:
            open_price: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size
            annualize: Whether to annualize the volatility
            
        Returns:
            Garman-Klass volatility series
        """
        # Garman-Klass estimator
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)
        
        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        gk_vol = np.sqrt(gk.rolling(window=window).mean())
        
        if annualize:
            gk_vol = gk_vol * np.sqrt(252)
            
        return gk_vol
        
    @staticmethod
    def rolling_volatility_regime(volatility: pd.Series, 
                                short_window: int = 20,
                                long_window: int = 60) -> pd.Series:
        """
        Identify volatility regimes (high/low volatility periods).
        
        Args:
            volatility: Volatility series
            short_window: Short-term window
            long_window: Long-term window
            
        Returns:
            Volatility regime series (1 = high vol, -1 = low vol, 0 = normal)
        """
        short_vol = volatility.rolling(window=short_window).mean()
        long_vol = volatility.rolling(window=long_window).mean()
        
        # High volatility if short-term > long-term + threshold
        high_vol_threshold = long_vol * 1.2
        low_vol_threshold = long_vol * 0.8
        
        regime = pd.Series(0, index=volatility.index)
        regime[short_vol > high_vol_threshold] = 1
        regime[short_vol < low_vol_threshold] = -1
        
        return regime
        
    @staticmethod
    def volatility_ratio(short_vol: pd.Series, long_vol: pd.Series) -> pd.Series:
        """
        Volatility ratio (short-term / long-term volatility).
        
        Args:
            short_vol: Short-term volatility
            long_vol: Long-term volatility
            
        Returns:
            Volatility ratio series
        """
        return short_vol / long_vol
        
    @staticmethod
    def volatility_percentile(volatility: pd.Series, window: int = 252) -> pd.Series:
        """
        Volatility percentile rank.
        
        Args:
            volatility: Volatility series
            window: Window size for percentile calculation
            
        Returns:
            Volatility percentile series (0-100)
        """
        return volatility.rolling(window=window).rank(pct=True) * 100
        
    @staticmethod
    def volatility_breakout(volatility: pd.Series, threshold_percentile: float = 80) -> pd.Series:
        """
        Detect volatility breakouts.
        
        Args:
            volatility: Volatility series
            threshold_percentile: Percentile threshold for breakout
            
        Returns:
            Breakout signal series (1 = breakout, 0 = normal)
        """
        threshold = volatility.rolling(window=252).quantile(threshold_percentile / 100)
        return (volatility > threshold).astype(int)
        
    @staticmethod
    def ewma_volatility(returns: pd.Series, lambda_param: float = 0.94,
                       annualize: bool = True) -> pd.Series:
        """
        Exponentially Weighted Moving Average volatility.
        
        Args:
            returns: Return series
            lambda_param: Decay factor
            annualize: Whether to annualize the volatility
            
        Returns:
            EWMA volatility series
        """
        # EWMA variance
        ewma_var = returns.ewm(alpha=1-lambda_param).var()
        ewma_vol = np.sqrt(ewma_var)
        
        if annualize:
            ewma_vol = ewma_vol * np.sqrt(252)
            
        return ewma_vol
        
    @staticmethod
    def volatility_of_volatility(volatility: pd.Series, window: int = 20) -> pd.Series:
        """
        Volatility of volatility (vol of vol).
        
        Args:
            volatility: Volatility series
            window: Window size
            
        Returns:
            Vol of vol series
        """
        return volatility.rolling(window=window).std()
        
    @staticmethod
    def intraday_volatility_pattern(volatility: pd.Series, 
                                  session_hours: list = None) -> pd.Series:
        """
        Identify intraday volatility patterns.
        
        Args:
            volatility: Volatility series
            session_hours: List of session hours to analyze
            
        Returns:
            Volatility pattern series
        """
        if session_hours is None:
            session_hours = [0, 6, 12, 18]  # Default 6-hour sessions
            
        # Extract hour from timestamp
        hour = volatility.index.hour
        
        # Create session labels
        session_labels = pd.cut(hour, bins=session_hours + [24], 
                               labels=[f'Session_{i}' for i in range(len(session_hours))],
                               include_lowest=True)
        
        # Calculate average volatility by session
        session_vol = volatility.groupby(session_labels).mean()
        
        # Map back to original index
        pattern = session_labels.map(session_vol)
        
        return pattern