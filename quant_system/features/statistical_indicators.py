"""
Statistical Indicators
=====================

Statistical measures including cointegration, correlation, and statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, List
import logging
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS

logger = logging.getLogger(__name__)


class StatisticalIndicators:
    """
    Statistical indicators and tests.
    
    All methods are static and deterministic.
    """
    
    @staticmethod
    def rolling_correlation(series1: pd.Series, series2: pd.Series, 
                           window: int = 20) -> pd.Series:
        """
        Rolling correlation between two series.
        
        Args:
            series1: First series
            series2: Second series
            window: Window size
            
        Returns:
            Rolling correlation series
        """
        return series1.rolling(window=window).corr(series2)
        
    @staticmethod
    def rolling_correlation_matrix(price_data: pd.DataFrame, 
                                 window: int = 20) -> pd.DataFrame:
        """
        Rolling correlation matrix for multiple instruments.
        
        Args:
            price_data: DataFrame with price data (columns = instruments)
            window: Window size
            
        Returns:
            Rolling correlation matrix
        """
        return price_data.rolling(window=window).corr()
        
    @staticmethod
    def cointegration_test(series1: pd.Series, series2: pd.Series,
                          max_lag: int = 4) -> Tuple[float, float, List[float]]:
        """
        Engle-Granger cointegration test.
        
        Args:
            series1: First price series
            series2: Second price series
            max_lag: Maximum lag for ADF test
            
        Returns:
            Tuple of (test_statistic, p_value, critical_values)
        """
        # Remove NaN values
        valid_data = pd.concat([series1, series2], axis=1).dropna()
        
        if len(valid_data) < 50:  # Need sufficient data
            return np.nan, np.nan, [np.nan, np.nan, np.nan]
            
        try:
            test_stat, p_value, critical_values = coint(
                valid_data.iloc[:, 0], 
                valid_data.iloc[:, 1], 
                maxlag=max_lag
            )
            return test_stat, p_value, critical_values
        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
            return np.nan, np.nan, [np.nan, np.nan, np.nan]
            
    @staticmethod
    def hedge_ratio(series1: pd.Series, series2: pd.Series,
                   window: int = 252) -> pd.Series:
        """
        Rolling hedge ratio (beta) between two series.
        
        Args:
            series1: First price series
            series2: Second price series
            window: Window size
            
        Returns:
            Rolling hedge ratio series
        """
        def calculate_beta(data):
            if len(data) < 10:  # Need minimum data points
                return np.nan
            try:
                y = data.iloc[:, 0].values
                x = data.iloc[:, 1].values
                # Add constant for intercept
                x_with_const = np.column_stack([np.ones(len(x)), x])
                model = OLS(y, x_with_const).fit()
                return model.params[1]  # Beta coefficient
            except:
                return np.nan
                
        # Rolling regression
        data = pd.concat([series1, series2], axis=1).dropna()
        hedge_ratios = data.rolling(window=window).apply(calculate_beta, raw=False)
        
        return hedge_ratios.iloc[:, 0]
        
    @staticmethod
    def spread_series(series1: pd.Series, series2: pd.Series,
                     hedge_ratio: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate spread series for pair trading.
        
        Args:
            series1: First price series
            series2: Second price series
            hedge_ratio: Hedge ratio series (if None, will be calculated)
            
        Returns:
            Spread series
        """
        if hedge_ratio is None:
            hedge_ratio = StatisticalIndicators.hedge_ratio(series1, series2)
            
        # Align series
        aligned_data = pd.concat([series1, series2, hedge_ratio], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return pd.Series(dtype=float)
            
        spread = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 2] * aligned_data.iloc[:, 1]
        
        return spread
        
    @staticmethod
    def spread_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Z-score of spread series.
        
        Args:
            spread: Spread series
            window: Window size for mean and std calculation
            
        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        return (spread - rolling_mean) / rolling_std
        
    @staticmethod
    def adf_test(series: pd.Series, max_lag: int = 4) -> Tuple[float, float, List[float]]:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        Args:
            series: Time series to test
            max_lag: Maximum lag for ADF test
            
        Returns:
            Tuple of (test_statistic, p_value, critical_values)
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 50:  # Need sufficient data
            return np.nan, np.nan, [np.nan, np.nan, np.nan]
            
        try:
            test_stat, p_value, critical_values, _ = adfuller(
                series_clean, maxlag=max_lag
            )
            return test_stat, p_value, critical_values
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return np.nan, np.nan, [np.nan, np.nan, np.nan]
            
    @staticmethod
    def half_life(series: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.
        
        Args:
            series: Price series
            
        Returns:
            Half-life in periods
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return np.nan
            
        try:
            # Calculate lagged series
            lagged = series_clean.shift(1).dropna()
            current = series_clean.iloc[1:]
            
            # Regression: current = alpha + beta * lagged
            x = lagged.values.reshape(-1, 1)
            y = current.values
            
            # Add constant
            x_with_const = np.column_stack([np.ones(len(x)), x])
            model = OLS(y, x_with_const).fit()
            
            beta = model.params[1]
            half_life = -np.log(2) / np.log(1 + beta)
            
            return half_life
        except Exception as e:
            logger.warning(f"Half-life calculation failed: {e}")
            return np.nan
            
    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent for long-term memory.
        
        Args:
            series: Time series
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent (0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting)
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 50:
            return np.nan
            
        try:
            lags = range(2, min(max_lag, len(series_clean) // 4))
            tau = []
            
            for lag in lags:
                # Calculate R/S statistic
                n = len(series_clean)
                m = series_clean.mean()
                y = series_clean - m
                z = y.cumsum()
                
                R = z.max() - z.min()
                S = series_clean.std()
                
                if S != 0:
                    tau.append(R / S)
                else:
                    tau.append(np.nan)
                    
            # Remove NaN values
            tau = np.array(tau)
            valid_idx = ~np.isnan(tau)
            
            if np.sum(valid_idx) < 3:
                return np.nan
                
            lags = np.array(lags)[valid_idx]
            tau = tau[valid_idx]
            
            # Linear regression: log(tau) = H * log(lag) + c
            log_lags = np.log(lags)
            log_tau = np.log(tau)
            
            H, _ = np.polyfit(log_lags, log_tau, 1)
            
            return H
        except Exception as e:
            logger.warning(f"Hurst exponent calculation failed: {e}")
            return np.nan
            
    @staticmethod
    def decay_weighted_correlation(series1: pd.Series, series2: pd.Series,
                                 window: int = 20, decay_factor: float = 0.9) -> pd.Series:
        """
        Decay-weighted correlation.
        
        Args:
            series1: First series
            series2: Second series
            window: Window size
            decay_factor: Decay factor for weighting
            
        Returns:
            Decay-weighted correlation series
        """
        def decay_weighted_corr(data):
            if len(data) < 5:
                return np.nan
                
            # Create decay weights
            weights = np.array([decay_factor ** i for i in range(len(data))])
            weights = weights[::-1]  # Most recent gets highest weight
            
            # Calculate weighted correlation
            x = data.iloc[:, 0].values
            y = data.iloc[:, 1].values
            
            # Remove NaN values
            valid_idx = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid_idx) < 3:
                return np.nan
                
            x = x[valid_idx]
            y = y[valid_idx]
            w = weights[valid_idx]
            
            # Weighted correlation
            w_mean_x = np.average(x, weights=w)
            w_mean_y = np.average(y, weights=w)
            
            numerator = np.sum(w * (x - w_mean_x) * (y - w_mean_y))
            denominator = np.sqrt(np.sum(w * (x - w_mean_x)**2) * np.sum(w * (y - w_mean_y)**2))
            
            if denominator == 0:
                return np.nan
                
            return numerator / denominator
            
        data = pd.concat([series1, series2], axis=1).dropna()
        return data.rolling(window=window).apply(decay_weighted_corr, raw=False)
        
    @staticmethod
    def rolling_regression(series1: pd.Series, series2: pd.Series,
                          window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Rolling regression between two series.
        
        Args:
            series1: Dependent variable (y)
            series2: Independent variable (x)
            window: Window size
            
        Returns:
            Tuple of (alpha, beta, r_squared)
        """
        def rolling_reg(data):
            if len(data) < 5:
                return pd.Series([np.nan, np.nan, np.nan])
                
            y = data.iloc[:, 0].values
            x = data.iloc[:, 1].values
            
            # Remove NaN values
            valid_idx = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid_idx) < 3:
                return pd.Series([np.nan, np.nan, np.nan])
                
            x = x[valid_idx]
            y = y[valid_idx]
            
            try:
                # Add constant for intercept
                x_with_const = np.column_stack([np.ones(len(x)), x])
                model = OLS(y, x_with_const).fit()
                
                alpha = model.params[0]
                beta = model.params[1]
                r_squared = model.rsquared
                
                return pd.Series([alpha, beta, r_squared])
            except:
                return pd.Series([np.nan, np.nan, np.nan])
                
        data = pd.concat([series1, series2], axis=1).dropna()
        results = data.rolling(window=window).apply(rolling_reg, raw=False)
        
        alpha = results.iloc[:, 0]
        beta = results.iloc[:, 1]
        r_squared = results.iloc[:, 2]
        
        return alpha, beta, r_squared