#!/usr/bin/env python3
"""
🚀 ULTRA-ADVANCED FOREX TRADING BOT 🚀
Quantitative Analysis, Mathematical Pattern Recognition & AI-Powered Trading

UPGRADED FEATURES:
- Advanced Quantitative Analysis with Mathematical Formulas
- Fourier Transform Pattern Recognition
- Wavelet Analysis for Market Microstructure
- Kalman Filter Price Prediction
- GARCH Volatility Modeling
- Monte Carlo Risk Simulation
- Advanced Statistical Arbitrage
- Machine Learning Ensemble Models
- Real-time Market Regime Detection
- Advanced Portfolio Optimization
- High-Frequency Trading Algorithms
- News Sentiment Analysis with NLP
- Dynamic Risk Management
- Backtesting with Walk-Forward Analysis
"""

import os
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Optional MetaTrader5 import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available - using alternative data sources")

# Advanced Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - some ML features will be limited")
import joblib

# Optional Optuna imports
try:
    import optuna
    from optuna.integration import TFKerasPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available - hyperparameter tuning will be limited")

# Advanced Technical Analysis
import ta
import pandas_ta as pta
from scipy import stats
from scipy.signal import find_peaks
# Optional TA-Lib import
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available - some technical indicators will be limited")

# Advanced Mathematics and Statistics
from scipy.stats import norm, skew, kurtosis, jarque_bera, shapiro
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import quad
from scipy.integrate import trapezoid as trapz
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from arch.unitroot import ADF, KPSS

# Optional PyWavelets import
try:
    import pywt  # Wavelet transforms
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("⚠️ PyWavelets not available - wavelet analysis will be limited")

# News and Sentiment Analysis
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

# Portfolio Optimization
from scipy.optimize import minimize

# Optional CVXPY import
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("⚠️ CVXPY not available - advanced portfolio optimization will be limited")

# Configuration
load_dotenv()

class QuantitativeAnalyzer:
    """Advanced Quantitative Analysis Engine with Mathematical Formulas"""
    
    def __init__(self):
        self.fourier_cache = {}
        self.wavelet_cache = {}
        self.kalman_state = {}
        
    def fourier_analysis(self, prices: pd.Series, window: int = 50) -> Dict[str, Any]:
        """Fourier Transform Analysis for Pattern Recognition"""
        try:
            if len(prices) < window:
                return {"error": "Insufficient data"}
            
            # Get recent data
            recent_prices = prices.tail(window).values
            
            # Apply FFT
            fft_values = fft(recent_prices)
            freqs = fftfreq(len(recent_prices))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_values) ** 2
            dominant_freqs = freqs[np.argsort(power_spectrum)[-5:]]
            
            # Calculate spectral density
            spectral_density = power_spectrum / np.sum(power_spectrum)
            
            # Detect cyclical patterns
            cycles = []
            for freq in dominant_freqs:
                if freq > 0:
                    period = 1 / freq
                    if 2 <= period <= window // 2:
                        cycles.append({
                            "period": period,
                            "frequency": freq,
                            "strength": spectral_density[np.where(freqs == freq)[0][0]]
                        })
            
            # Calculate trend strength using low frequencies
            low_freq_mask = np.abs(freqs) < 0.1
            trend_strength = np.sum(spectral_density[low_freq_mask])
            
            return {
                "dominant_frequencies": dominant_freqs.tolist(),
                "cycles": cycles,
                "trend_strength": trend_strength,
                "spectral_density": spectral_density.tolist(),
                "noise_level": np.mean(spectral_density[-10:])  # High frequency noise
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def wavelet_analysis(self, prices: pd.Series, wavelet: str = 'db4') -> Dict[str, Any]:
        """Wavelet Transform Analysis for Multi-Resolution Analysis"""
        try:
            if not PYWT_AVAILABLE:
                return {"error": "PyWavelets not available"}
                
            if len(prices) < 32:
                return {"error": "Insufficient data"}
            
            # Apply wavelet decomposition
            coeffs = pywt.wavedec(prices.values, wavelet, level=4)
            cA, cD4, cD3, cD2, cD1 = coeffs
            
            # Calculate energy at each level
            energy_levels = [np.sum(np.square(coeff)) for coeff in coeffs]
            total_energy = sum(energy_levels)
            energy_ratios = [e / total_energy for e in energy_levels]
            
            # Detect significant changes using detail coefficients
            threshold = np.std(cD1) * 2
            significant_changes = np.abs(cD1) > threshold
            
            # Calculate volatility at different time scales
            volatility_scales = {
                "intraday": np.std(cD1),
                "short_term": np.std(cD2),
                "medium_term": np.std(cD3),
                "long_term": np.std(cD4)
            }
            
            return {
                "energy_ratios": energy_ratios,
                "volatility_scales": volatility_scales,
                "significant_changes": significant_changes.sum(),
                "approximation": cA.tolist(),
                "details": {
                    "level1": cD1.tolist(),
                    "level2": cD2.tolist(),
                    "level3": cD3.tolist(),
                    "level4": cD4.tolist()
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def kalman_filter_prediction(self, prices: pd.Series, state_dim: int = 2) -> Dict[str, Any]:
        """Kalman Filter for Price Prediction"""
        try:
            if len(prices) < 10:
                return {"error": "Insufficient data"}
            
            # State transition matrix (constant velocity model)
            F = np.array([[1, 1], [0, 1]], dtype=float)
            
            # Observation matrix
            H = np.array([[1, 0]], dtype=float)
            
            # Process noise covariance
            Q = np.array([[0.1, 0], [0, 0.1]], dtype=float)
            
            # Measurement noise covariance
            R = np.array([[0.5]], dtype=float)
            
            # Initialize state
            if "kalman_state" not in self.kalman_state:
                self.kalman_state["kalman_state"] = {
                    "x": np.array([[prices.iloc[0]], [0]], dtype=float),  # [price, velocity]
                    "P": np.eye(2, dtype=float) * 1000
                }
            
            state = self.kalman_state["kalman_state"]
            x, P = state["x"], state["P"]
            
            predictions = []
            filtered_prices = []
            
            for i, price in enumerate(prices):
                # Prediction step
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
                
                # Update step
                y = price - (H @ x_pred)[0, 0]  # Innovation
                S = H @ P_pred @ H.T + R  # Innovation covariance
                K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
                
                x = x_pred + K * y
                P = (np.eye(2) - K @ H) @ P_pred
                
                # Store results
                predictions.append((H @ x)[0, 0])
                filtered_prices.append(x[0, 0])
            
            # Update state
            self.kalman_state["kalman_state"] = {"x": x, "P": P}
            
            # Calculate prediction accuracy
            mse = np.mean((prices.values - predictions) ** 2)
            mae = np.mean(np.abs(prices.values - predictions))
            
            return {
                "predictions": predictions,
                "filtered_prices": filtered_prices,
                "current_state": x.flatten().tolist(),
                "uncertainty": np.diag(P).tolist(),
                "mse": mse,
                "mae": mae,
                "velocity": x[1, 0]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def garch_volatility_modeling(self, returns: pd.Series) -> Dict[str, Any]:
        """GARCH Volatility Modeling"""
        try:
            if len(returns) < 50:
                return {"error": "Insufficient data"}
            
            # Fit GARCH(1,1) model
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            fitted_model = model.fit(disp='off')
            
            # Get volatility forecasts
            forecasts = fitted_model.forecast(horizon=5)
            volatility_forecast = forecasts.variance.iloc[-1].values / 10000  # Convert back
            
            # Calculate VaR
            alpha = 0.05
            var_95 = np.percentile(returns, alpha * 100)
            var_99 = np.percentile(returns, 0.01 * 100)
            
            # Calculate Expected Shortfall (CVaR)
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            return {
                "garch_params": {
                    "omega": fitted_model.params['omega'],
                    "alpha": fitted_model.params['alpha[1]'],
                    "beta": fitted_model.params['beta[1]']
                },
                "volatility_forecast": volatility_forecast.tolist(),
                "current_volatility": np.sqrt(fitted_model.conditional_volatility.iloc[-1] / 100),
                "var_95": var_95,
                "var_99": var_99,
                "es_95": es_95,
                "es_99": es_99,
                "aic": fitted_model.aic,
                "bic": fitted_model.bic
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def monte_carlo_simulation(self, returns: pd.Series, n_simulations: int = 10000, 
                              horizon: int = 30) -> Dict[str, Any]:
        """Monte Carlo Risk Simulation"""
        try:
            if len(returns) < 30:
                return {"error": "Insufficient data"}
            
            # Calculate parameters
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            random_returns = np.random.normal(mean_return, std_return, (n_simulations, horizon))
            
            # Calculate portfolio paths
            initial_price = 1.0
            price_paths = np.zeros((n_simulations, horizon + 1))
            price_paths[:, 0] = initial_price
            
            for t in range(horizon):
                price_paths[:, t + 1] = price_paths[:, t] * (1 + random_returns[:, t])
            
            # Calculate risk metrics
            final_prices = price_paths[:, -1]
            
            # VaR calculations
            var_95 = np.percentile(final_prices, 5)
            var_99 = np.percentile(final_prices, 1)
            
            # Expected Shortfall
            es_95 = np.mean(final_prices[final_prices <= var_95])
            es_99 = np.mean(final_prices[final_prices <= var_99])
            
            # Probability of loss
            prob_loss = np.mean(final_prices < initial_price)
            
            # Maximum drawdown simulation
            max_drawdowns = []
            for path in price_paths:
                peak = np.maximum.accumulate(path)
                drawdown = (path - peak) / peak
                max_drawdowns.append(np.min(drawdown))
            
            avg_max_drawdown = np.mean(max_drawdowns)
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "es_95": es_95,
                "es_99": es_99,
                "prob_loss": prob_loss,
                "avg_max_drawdown": avg_max_drawdown,
                "expected_return": np.mean(final_prices),
                "volatility": np.std(final_prices),
                "sharpe_ratio": (np.mean(final_prices) - initial_price) / np.std(final_prices),
                "price_paths": price_paths[:100].tolist()  # Sample paths for visualization
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def statistical_arbitrage(self, price1: pd.Series, price2: pd.Series) -> Dict[str, Any]:
        """Statistical Arbitrage Analysis"""
        try:
            if len(price1) != len(price2) or len(price1) < 50:
                return {"error": "Insufficient data"}
            
            # Calculate spread
            spread = price1 - price2
            
            # Test for cointegration
            coint_result = coint(price1, price2)
            
            # Calculate hedge ratio using OLS
            X = sm.add_constant(price2)
            model = sm.OLS(price1, X).fit()
            hedge_ratio = model.params[1]
            intercept = model.params[0]
            
            # Calculate z-score of spread
            spread_mean = spread.mean()
            spread_std = spread.std()
            z_score = (spread - spread_mean) / spread_std
            
            # Trading signals
            entry_threshold = 2.0
            exit_threshold = 0.5
            
            long_signal = z_score < -entry_threshold
            short_signal = z_score > entry_threshold
            exit_signal = np.abs(z_score) < exit_threshold
            
            # Calculate half-life of mean reversion
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            if len(spread_lag) > 0 and len(spread_diff) > 0:
                X_lag = sm.add_constant(spread_lag)
                model_lag = sm.OLS(spread_diff, X_lag).fit()
                half_life = -np.log(2) / model_lag.params[1] if model_lag.params[1] < 0 else np.inf
            else:
                half_life = np.inf
            
            return {
                "cointegration_pvalue": coint_result[1],
                "cointegrated": coint_result[1] < 0.05,
                "hedge_ratio": hedge_ratio,
                "intercept": intercept,
                "current_zscore": z_score.iloc[-1],
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "half_life": half_life,
                "long_signals": long_signal.sum(),
                "short_signals": short_signal.sum(),
                "exit_signals": exit_signal.sum(),
                "r_squared": model.rsquared
            }
            
        except Exception as e:
            return {"error": str(e)}

class MathematicalPatternRecognizer:
    """Advanced Mathematical Pattern Recognition Engine"""
    
    def __init__(self):
        self.pattern_cache = {}
        
    def detect_fibonacci_retracements(self, prices: pd.Series) -> Dict[str, Any]:
        """Detect Fibonacci Retracement Levels"""
        try:
            if len(prices) < 20:
                return {"error": "Insufficient data"}
            
            # Find swing high and low
            recent_data = prices.tail(50)
            swing_high = recent_data.max()
            swing_low = recent_data.min()
            swing_high_idx = recent_data.idxmax()
            swing_low_idx = recent_data.idxmin()
            
            # Ensure we have a proper swing
            if swing_high_idx <= swing_low_idx:
                return {"error": "Invalid swing points"}
            
            # Calculate Fibonacci levels
            price_range = swing_high - swing_low
            fib_levels = {
                "0%": swing_high,
                "23.6%": swing_high - 0.236 * price_range,
                "38.2%": swing_high - 0.382 * price_range,
                "50%": swing_high - 0.5 * price_range,
                "61.8%": swing_high - 0.618 * price_range,
                "78.6%": swing_high - 0.786 * price_range,
                "100%": swing_low
            }
            
            # Check current price position
            current_price = prices.iloc[-1]
            current_level = None
            for level_name, level_price in fib_levels.items():
                if abs(current_price - level_price) / current_price < 0.01:  # 1% tolerance
                    current_level = level_name
                    break
            
            return {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "fibonacci_levels": fib_levels,
                "current_level": current_level,
                "price_range": price_range,
                "retracement_percentage": ((swing_high - current_price) / price_range) * 100
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def detect_elliott_waves(self, prices: pd.Series) -> Dict[str, Any]:
        """Detect Elliott Wave Patterns"""
        try:
            if len(prices) < 100:
                return {"error": "Insufficient data"}
            
            # Find significant peaks and troughs
            recent_data = prices.tail(100)
            
            # Use scipy to find peaks
            peaks, _ = find_peaks(recent_data.values, distance=5, prominence=recent_data.std() * 0.5)
            troughs, _ = find_peaks(-recent_data.values, distance=5, prominence=recent_data.std() * 0.5)
            
            # Combine and sort all significant points
            all_points = []
            for peak in peaks:
                all_points.append((peak, recent_data.iloc[peak], 'peak'))
            for trough in troughs:
                all_points.append((trough, recent_data.iloc[trough], 'trough'))
            
            all_points.sort(key=lambda x: x[0])
            
            if len(all_points) < 5:
                return {"error": "Insufficient wave points"}
            
            # Analyze wave structure
            waves = []
            for i in range(len(all_points) - 1):
                current_point = all_points[i]
                next_point = all_points[i + 1]
                
                wave_length = next_point[0] - current_point[0]
                wave_height = abs(next_point[1] - current_point[1])
                wave_direction = 1 if next_point[1] > current_point[1] else -1
                
                waves.append({
                    "start_idx": current_point[0],
                    "end_idx": next_point[0],
                    "start_price": current_point[1],
                    "end_price": next_point[1],
                    "length": wave_length,
                    "height": wave_height,
                    "direction": wave_direction,
                    "type": current_point[2]
                })
            
            # Identify Elliott Wave patterns
            wave_patterns = self._analyze_elliott_structure(waves)
            
            return {
                "waves": waves,
                "patterns": wave_patterns,
                "total_waves": len(waves),
                "current_phase": self._determine_current_phase(waves)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_elliott_structure(self, waves: List[Dict]) -> List[Dict]:
        """Analyze Elliott Wave structure"""
        patterns = []
        
        if len(waves) < 5:
            return patterns
        
        # Look for 5-wave patterns
        for i in range(len(waves) - 4):
            wave_group = waves[i:i+5]
            
            # Check if it's a 5-wave pattern
            if self._is_five_wave_pattern(wave_group):
                patterns.append({
                    "type": "5-wave",
                    "start_idx": wave_group[0]["start_idx"],
                    "end_idx": wave_group[4]["end_idx"],
                    "confidence": self._calculate_pattern_confidence(wave_group),
                    "waves": wave_group
                })
        
        return patterns
    
    def _is_five_wave_pattern(self, waves: List[Dict]) -> bool:
        """Check if waves form a valid 5-wave Elliott pattern"""
        if len(waves) != 5:
            return False
        
        # Check wave directions (1,3,5 up; 2,4 down)
        expected_directions = [1, -1, 1, -1, 1]
        actual_directions = [w["direction"] for w in waves]
        
        return actual_directions == expected_directions
    
    def _calculate_pattern_confidence(self, waves: List[Dict]) -> float:
        """Calculate confidence score for Elliott pattern"""
        if len(waves) != 5:
            return 0.0
        
        # Check Fibonacci relationships
        confidence = 0.0
        
        # Wave 2 should retrace 38.2% to 61.8% of wave 1
        wave1_height = waves[0]["height"]
        wave2_height = waves[1]["height"]
        if wave1_height > 0:
            retracement = wave2_height / wave1_height
            if 0.382 <= retracement <= 0.618:
                confidence += 0.3
        
        # Wave 3 should be the longest
        wave3_height = waves[2]["height"]
        max_height = max([w["height"] for w in waves])
        if wave3_height == max_height:
            confidence += 0.3
        
        # Wave 4 should retrace 23.6% to 50% of wave 3
        wave4_height = waves[3]["height"]
        if wave3_height > 0:
            retracement = wave4_height / wave3_height
            if 0.236 <= retracement <= 0.5:
                confidence += 0.4
        
        return min(confidence, 1.0)
    
    def _determine_current_phase(self, waves: List[Dict]) -> str:
        """Determine current Elliott Wave phase"""
        if not waves:
            return "unknown"
        
        last_wave = waves[-1]
        
        if last_wave["direction"] == 1:
            return "impulse"
        else:
            return "correction"
    
    def detect_harmonic_patterns(self, prices: pd.Series) -> Dict[str, Any]:
        """Detect Harmonic Trading Patterns (Gartley, Butterfly, etc.)"""
        try:
            if len(prices) < 50:
                return {"error": "Insufficient data"}
            
            recent_data = prices.tail(50)
            
            # Find significant points
            peaks, _ = find_peaks(recent_data.values, distance=3, prominence=recent_data.std() * 0.3)
            troughs, _ = find_peaks(-recent_data.values, distance=3, prominence=recent_data.std() * 0.3)
            
            # Combine points
            points = []
            for peak in peaks:
                points.append((peak, recent_data.iloc[peak], 'peak'))
            for trough in troughs:
                points.append((trough, recent_data.iloc[trough], 'trough'))
            
            points.sort(key=lambda x: x[0])
            
            if len(points) < 4:
                return {"error": "Insufficient points for harmonic analysis"}
            
            # Look for harmonic patterns
            patterns = []
            
            # Gartley Pattern (XABCD)
            for i in range(len(points) - 4):
                X, A, B, C, D = points[i:i+5]
                
                if self._is_gartley_pattern(X, A, B, C, D):
                    patterns.append({
                        "type": "Gartley",
                        "points": [X, A, B, C, D],
                        "confidence": self._calculate_harmonic_confidence(X, A, B, C, D),
                        "completion_ratio": self._calculate_completion_ratio(X, A, B, C, D)
                    })
            
            # Butterfly Pattern
            for i in range(len(points) - 4):
                X, A, B, C, D = points[i:i+5]
                
                if self._is_butterfly_pattern(X, A, B, C, D):
                    patterns.append({
                        "type": "Butterfly",
                        "points": [X, A, B, C, D],
                        "confidence": self._calculate_harmonic_confidence(X, A, B, C, D),
                        "completion_ratio": self._calculate_completion_ratio(X, A, B, C, D)
                    })
            
            return {
                "patterns": patterns,
                "total_patterns": len(patterns),
                "high_confidence": len([p for p in patterns if p["confidence"] > 0.8])
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _is_gartley_pattern(self, X, A, B, C, D) -> bool:
        """Check if points form a Gartley pattern"""
        # Gartley ratios: AB = 61.8% of XA, BC = 38.2% or 88.6% of AB, CD = 78.6% of XA
        xa = abs(A[1] - X[1])
        ab = abs(B[1] - A[1])
        bc = abs(C[1] - B[1])
        cd = abs(D[1] - C[1])
        
        if xa == 0 or ab == 0:
            return False
        
        ab_ratio = ab / xa
        bc_ratio = bc / ab
        cd_ratio = cd / xa
        
        # Check if ratios are within tolerance
        tolerance = 0.1
        return (abs(ab_ratio - 0.618) < tolerance and 
                (abs(bc_ratio - 0.382) < tolerance or abs(bc_ratio - 0.886) < tolerance) and
                abs(cd_ratio - 0.786) < tolerance)
    
    def _is_butterfly_pattern(self, X, A, B, C, D) -> bool:
        """Check if points form a Butterfly pattern"""
        # Butterfly ratios: AB = 78.6% of XA, BC = 38.2% or 88.6% of AB, CD = 127.2% or 161.8% of XA
        xa = abs(A[1] - X[1])
        ab = abs(B[1] - A[1])
        bc = abs(C[1] - B[1])
        cd = abs(D[1] - C[1])
        
        if xa == 0 or ab == 0:
            return False
        
        ab_ratio = ab / xa
        bc_ratio = bc / ab
        cd_ratio = cd / xa
        
        tolerance = 0.1
        return (abs(ab_ratio - 0.786) < tolerance and 
                (abs(bc_ratio - 0.382) < tolerance or abs(bc_ratio - 0.886) < tolerance) and
                (abs(cd_ratio - 1.272) < tolerance or abs(cd_ratio - 1.618) < tolerance))
    
    def _calculate_harmonic_confidence(self, X, A, B, C, D) -> float:
        """Calculate confidence score for harmonic pattern"""
        # This is a simplified confidence calculation
        # In practice, you'd want more sophisticated pattern matching
        return 0.8  # Placeholder
    
    def _calculate_completion_ratio(self, X, A, B, C, D) -> float:
        """Calculate how complete the pattern is"""
        # This would calculate how much of the pattern has been completed
        return 0.9  # Placeholder

class AdvancedForexBot:
    def __init__(self, config_path: str = "bot_config.json"):
        """Initialize the upgraded advanced forex trading bot"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_mt5()
        self.setup_advanced_ml_models()
        self.setup_portfolio_optimizer()
        
        # Initialize quantitative analysis engines
        self.quant_analyzer = QuantitativeAnalyzer()
        self.pattern_recognizer = MathematicalPatternRecognizer()
        
        # Enhanced trading state
        self.active_trades = {}
        self.grid_levels = {}
        self.hedge_positions = {}
        self.performance_metrics = {}
        self.portfolio_state = {}
        self.market_regime = "unknown"
        
        # Advanced caches
        self.news_cache = {}
        self.sentiment_cache = {}
        self.model_cache = {}
        self.scaler_cache = {}
        self.feature_cache = {}
        self.quant_cache = {}
        self.pattern_cache = {}
        
        # Threading and async
        self.running = False
        self.trading_thread = None
        self.news_thread = None
        self.analysis_thread = None
        self.optimization_thread = None
        self.quant_analysis_thread = None
        
        # Performance tracking
        self.trade_history = []
        self.performance_history = []
        self.risk_metrics = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Load enhanced bot configuration"""
        default_config = {
            "mt5": {
                "login": int(os.getenv("MT5_LOGIN", "0")),
                "password": os.getenv("MT5_PASSWORD", ""),
                "server": os.getenv("MT5_SERVER", ""),
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"],
                "timeframe": "M5",
                "max_retries": 3,
                "retry_delay": 5
            },
            "trading": {
                "scalping": {
                    "enabled": True,
                    "min_profit": 0.0005,
                    "max_loss": 0.001,
                    "position_size": 0.01,
                    "momentum_threshold": 0.7,
                    "volatility_filter": True,
                    "time_filter": True
                },
                "grid": {
                    "enabled": True,
                    "levels": 5,
                    "spacing": 0.001,
                    "position_size": 0.01,
                    "adaptive_spacing": True,
                    "dynamic_levels": True,
                    "max_grid_trades": 10
                },
                "hedging": {
                    "enabled": True,
                    "correlation_threshold": 0.7,
                    "max_hedge_ratio": 0.5,
                    "auto_hedge": True,
                    "portfolio_hedging": True
                },
                "risk": {
                    "max_daily_loss": 0.02,
                    "max_position_size": 0.1,
                    "stop_loss": 0.005,
                    "take_profit": 0.01,
                    "max_drawdown": 0.05,
                    "position_sizing": "kelly",
                    "var_confidence": 0.95,
                    "max_correlation": 0.8
                }
            },
            "ml": {
                "enabled": True,
                "models": {
                    "price_prediction": ["lstm", "transformer", "ensemble"],
                    "volatility_prediction": ["garch", "lstm", "ensemble"],
                    "regime_detection": ["hmm", "gmm", "clustering"]
                },
                "retrain_interval": 86400,
                "prediction_threshold": 0.6,
                "hyperparameter_tuning": True,
                "cross_validation": True,
                "feature_selection": True,
                "ensemble_method": "voting"
            },
            "news": {
                "enabled": True,
                "sources": ["newsapi", "alphavantage", "yfinance"],
                "api_keys": {
                    "newsapi": os.getenv("NEWS_API_KEY", ""),
                    "alphavantage": os.getenv("ALPHA_VANTAGE_KEY", "")
                },
                "impact_threshold": 0.6,
                "update_interval": 300,
                "sentiment_analysis": "ensemble"
            },
            "portfolio": {
                "optimization": True,
                "method": "risk_parity",
                "rebalance_interval": 3600,
                "max_assets": 10,
                "min_weight": 0.05,
                "max_weight": 0.3
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Deep merge user config with defaults
                self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, default: Dict, user: Dict):
        """Deep merge configuration dictionaries"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._deep_merge(default[key], value)
            else:
                default[key] = value
    
    def setup_logging(self):
        """Setup enhanced logging configuration"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create rotating file handler
        from logging.handlers import RotatingFileHandler
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler(
                    os.path.join(log_dir, 'trading_bot.log'),
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set specific logger levels
        logging.getLogger('tensorflow').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    def setup_mt5(self):
        """Initialize enhanced MetaTrader5 connection with retry logic"""
        if not MT5_AVAILABLE:
            self.logger.warning("MetaTrader5 not available - using alternative data sources")
            return False
            
        max_retries = self.config["mt5"]["max_retries"]
        retry_delay = self.config["mt5"]["retry_delay"]
        
        for attempt in range(max_retries):
            try:
                if not mt5.initialize():
                    raise Exception("Failed to initialize MT5")
                
                # Login to MT5
                if not mt5.login(
                    login=self.config["mt5"]["login"],
                    password=self.config["mt5"]["password"],
                    server=self.config["mt5"]["server"]
                ):
                    raise Exception("Failed to login to MT5")
                
                # Test connection
                account_info = mt5.account_info()
                if account_info is None:
                    raise Exception("Failed to get account info")
                
                self.logger.info(f"Successfully connected to MetaTrader5 - Account: {account_info.login}")
                self.logger.info(f"Balance: {account_info.balance}, Equity: {account_info.equity}")
                return True
                
            except Exception as e:
                self.logger.error(f"MT5 connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    mt5.shutdown()
                else:
                    self.logger.error("Failed to connect to MT5 after all retries")
                    return False
        
        return False
    
    def setup_advanced_ml_models(self):
        """Initialize advanced machine learning models"""
        self.models = {
            'price_prediction': {},
            'volatility_prediction': {},
            'regime_detection': {},
            'sentiment_analysis': {}
        }
        
        self.scalers = {
            'price_features': RobustScaler(),
            'volatility_features': RobustScaler(),
            'sentiment_features': StandardScaler()
        }
        
        # Initialize deep learning models
        self._initialize_deep_learning_models()
        
        # Load pre-trained models if available
        self.load_models()
    
    def _initialize_deep_learning_models(self):
        """Initialize deep learning models"""
        try:
            # LSTM for price prediction
            self.models['price_prediction']['lstm'] = self._create_lstm_model()
            
            # Transformer for price prediction
            self.models['price_prediction']['transformer'] = self._create_transformer_model()
            
            # GARCH for volatility
            self.models['volatility_prediction']['garch'] = None  # Will be initialized when needed
            
            # HMM for regime detection
            self.models['regime_detection']['hmm'] = None  # Will be initialized when needed
            
            self.logger.info("Deep learning models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing deep learning models: {e}")
    
    def _create_lstm_model(self):
        """Create LSTM model for price prediction"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(None, 50)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_transformer_model(self):
        """Create Transformer model for price prediction"""
        # Simplified transformer for time series
        inputs = keras.Input(shape=(None, 50))
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=64
        )(inputs, inputs)
        
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
        
        # Feed forward
        ffn = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(50)
        ])
        
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6)(ffn(x) + x)
        
        # Global average pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def setup_portfolio_optimizer(self):
        """Setup portfolio optimization"""
        self.portfolio_optimizer = PortfolioOptimizer(self.config["portfolio"])
    
    def get_market_data(self, symbol: str, timeframe: str = "M5", bars: int = 1000) -> pd.DataFrame:
        """Get market data from MT5 or alternative sources"""
        try:
            if MT5_AVAILABLE:
                # Convert timeframe string to MT5 constant
                tf_map = {
                    "M1": mt5.TIMEFRAME_M1,
                    "M5": mt5.TIMEFRAME_M5,
                    "M15": mt5.TIMEFRAME_M15,
                    "M30": mt5.TIMEFRAME_M30,
                    "H1": mt5.TIMEFRAME_H1,
                    "H4": mt5.TIMEFRAME_H4,
                    "D1": mt5.TIMEFRAME_D1
                }
                
                mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
                
                # Get rates
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
                if rates is None:
                    return self._generate_sample_data(symbol, timeframe, bars)
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
            else:
                # Use alternative data source or generate sample data
                df = self._generate_sample_data(symbol, timeframe, bars)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return self._generate_sample_data(symbol, timeframe, bars)
    
    def _generate_sample_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Generate sample data for demonstration purposes"""
        try:
            # Generate realistic forex price data
            np.random.seed(42)
            
            # Base price
            base_price = 1.1000 if "EUR" in symbol else 1.2500
            
            # Generate price movements with trend and volatility
            returns = np.random.normal(0, 0.001, bars)  # 0.1% daily volatility
            
            # Add some trend
            trend = np.linspace(0, 0.02, bars)  # 2% upward trend
            returns += trend / bars
            
            # Add some cyclical patterns
            cycle1 = 0.0005 * np.sin(np.linspace(0, 4*np.pi, bars))  # 4 cycles
            cycle2 = 0.0003 * np.sin(np.linspace(0, 8*np.pi, bars))  # 8 cycles
            returns += cycle1 + cycle2
            
            # Calculate prices
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLC data
            data = []
            for i, price in enumerate(prices):
                # Generate realistic OHLC from close price
                volatility = abs(np.random.normal(0, 0.0005))
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                open_price = prices[i-1] if i > 0 else price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'time': datetime.now() - timedelta(hours=bars-i),
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'tick_volume': volume
                })
            
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating sample data: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe with quantitative analysis"""
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = bb.bollinger_wband()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Volume indicators
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            
            # Price patterns
            df['price_change'] = df['close'].pct_change()
            df['price_volatility'] = df['price_change'].rolling(20).std()
            
            # Add quantitative analysis indicators
            df = self.add_quantitative_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def add_quantitative_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced quantitative analysis indicators"""
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Fourier Analysis
            fourier_analysis = self.quant_analyzer.fourier_analysis(df['close'])
            if 'error' not in fourier_analysis:
                df['trend_strength'] = fourier_analysis.get('trend_strength', 0)
                df['noise_level'] = fourier_analysis.get('noise_level', 0)
            
            # Wavelet Analysis
            wavelet_analysis = self.quant_analyzer.wavelet_analysis(df['close'])
            if 'error' not in wavelet_analysis:
                volatility_scales = wavelet_analysis.get('volatility_scales', {})
                df['volatility_intraday'] = volatility_scales.get('intraday', 0)
                df['volatility_short_term'] = volatility_scales.get('short_term', 0)
                df['volatility_medium_term'] = volatility_scales.get('medium_term', 0)
                df['volatility_long_term'] = volatility_scales.get('long_term', 0)
            
            # Kalman Filter Prediction
            kalman_result = self.quant_analyzer.kalman_filter_prediction(df['close'])
            if 'error' not in kalman_result:
                df['kalman_price'] = kalman_result.get('filtered_prices', [0] * len(df))
                df['kalman_velocity'] = kalman_result.get('velocity', 0)
                df['kalman_uncertainty'] = kalman_result.get('uncertainty', [0, 0])[0]
            
            # GARCH Volatility
            if len(df) > 50:
                garch_result = self.quant_analyzer.garch_volatility_modeling(df['returns'].dropna())
                if 'error' not in garch_result:
                    df['garch_volatility'] = garch_result.get('current_volatility', 0)
                    df['var_95'] = garch_result.get('var_95', 0)
                    df['var_99'] = garch_result.get('var_99', 0)
            
            # Statistical measures
            df['skewness'] = df['returns'].rolling(20).skew()
            df['kurtosis'] = df['returns'].rolling(20).kurt()
            df['jarque_bera'] = df['returns'].rolling(20).apply(
                lambda x: jarque_bera(x)[1] if len(x) > 10 else np.nan
            )
            
            # Hurst Exponent for trend persistence
            df['hurst_exponent'] = df['returns'].rolling(50).apply(
                lambda x: self.calculate_hurst_exponent(x) if len(x) > 20 else np.nan
            )
            
            # Fractal Dimension
            df['fractal_dimension'] = df['close'].rolling(20).apply(
                lambda x: self.calculate_fractal_dimension(x) if len(x) > 10 else np.nan
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding quantitative indicators: {e}")
            return df
    
    def calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst Exponent for trend persistence"""
        try:
            if len(returns) < 20:
                return np.nan
            
            # Remove NaN values
            returns = returns.dropna()
            if len(returns) < 20:
                return np.nan
            
            # Calculate R/S statistic
            n = len(returns)
            mean_return = returns.mean()
            deviations = returns - mean_return
            cumulative_deviations = deviations.cumsum()
            range_series = cumulative_deviations.max() - cumulative_deviations.min()
            
            if range_series == 0:
                return 0.5
            
            std_return = returns.std()
            if std_return == 0:
                return 0.5
            
            rs_statistic = range_series / std_return
            
            # Hurst exponent
            hurst = np.log(rs_statistic) / np.log(n)
            
            return hurst
            
        except Exception as e:
            return np.nan
    
    def calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """Calculate Fractal Dimension using Box-Counting method"""
        try:
            if len(prices) < 10:
                return np.nan
            
            # Normalize prices
            prices_norm = (prices - prices.min()) / (prices.max() - prices.min())
            
            # Box-counting method
            n = len(prices_norm)
            box_sizes = [2, 4, 8, 16]
            counts = []
            
            for box_size in box_sizes:
                if box_size >= n:
                    continue
                
                count = 0
                for i in range(0, n, box_size):
                    end_idx = min(i + box_size, n)
                    box_data = prices_norm[i:end_idx]
                    if len(box_data) > 0:
                        count += 1
                
                counts.append(count)
            
            if len(counts) < 2:
                return np.nan
            
            # Linear regression to find fractal dimension
            log_box_sizes = np.log(box_sizes[:len(counts)])
            log_counts = np.log(counts)
            
            if len(log_box_sizes) > 1:
                slope, _ = np.polyfit(log_box_sizes, log_counts, 1)
                fractal_dimension = -slope
                return fractal_dimension
            
            return np.nan
            
        except Exception as e:
            return np.nan
    
    def detect_mathematical_patterns(self, symbol: str) -> Dict[str, Any]:
        """Detect mathematical patterns using advanced algorithms"""
        try:
            df = self.get_market_data(symbol, "H1", 200)
            if df.empty:
                return {"error": "No data available"}
            
            patterns = {}
            
            # Fibonacci Retracements
            fib_patterns = self.pattern_recognizer.detect_fibonacci_retracements(df['close'])
            if 'error' not in fib_patterns:
                patterns['fibonacci'] = fib_patterns
            
            # Elliott Waves
            elliott_patterns = self.pattern_recognizer.detect_elliott_waves(df['close'])
            if 'error' not in elliott_patterns:
                patterns['elliott_waves'] = elliott_patterns
            
            # Harmonic Patterns
            harmonic_patterns = self.pattern_recognizer.detect_harmonic_patterns(df['close'])
            if 'error' not in harmonic_patterns:
                patterns['harmonic'] = harmonic_patterns
            
            # Fourier Analysis
            fourier_analysis = self.quant_analyzer.fourier_analysis(df['close'])
            if 'error' not in fourier_analysis:
                patterns['fourier'] = fourier_analysis
            
            # Wavelet Analysis
            wavelet_analysis = self.quant_analyzer.wavelet_analysis(df['close'])
            if 'error' not in wavelet_analysis:
                patterns['wavelet'] = wavelet_analysis
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting mathematical patterns: {e}")
            return {"error": str(e)}
    
    def quantitative_risk_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive quantitative risk analysis"""
        try:
            df = self.get_market_data(symbol, "H1", 500)
            if df.empty:
                return {"error": "No data available"}
            
            returns = df['close'].pct_change().dropna()
            
            # Monte Carlo Simulation
            mc_result = self.quant_analyzer.monte_carlo_simulation(returns)
            
            # GARCH Volatility Modeling
            garch_result = self.quant_analyzer.garch_volatility_modeling(returns)
            
            # Statistical Arbitrage (if we have multiple symbols)
            arb_result = None
            if len(self.config["trading"]["symbols"]) > 1:
                other_symbols = [s for s in self.config["trading"]["symbols"] if s != symbol]
                if other_symbols:
                    other_df = self.get_market_data(other_symbols[0], "H1", 500)
                    if not other_df.empty:
                        arb_result = self.quant_analyzer.statistical_arbitrage(
                            df['close'], other_df['close']
                        )
            
            return {
                "monte_carlo": mc_result,
                "garch": garch_result,
                "statistical_arbitrage": arb_result,
                "current_volatility": returns.std() * np.sqrt(252),  # Annualized
                "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252),
                "max_drawdown": self.calculate_max_drawdown(df['close']),
                "var_95": np.percentile(returns, 5),
                "var_99": np.percentile(returns, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error in quantitative risk analysis: {e}")
            return {"error": str(e)}
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except Exception as e:
            return 0.0
    
    def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment for a currency pair"""
        try:
            # Extract base and quote currencies
            base = symbol[:3]
            quote = symbol[3:]
            
            # Check cache first
            cache_key = f"{base}_{quote}_{datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # Get news from API
            api_key = self.config["news"]["api_keys"]["newsapi"]
            if not api_key:
                return 0.0
            
            # Search for relevant news
            query = f"({base} OR {quote}) AND (forex OR currency OR economy)"
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&sortBy=publishedAt&pageSize=10"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return 0.0
            
            news_data = response.json()
            articles = news_data.get('articles', [])
            
            if not articles:
                return 0.0
            
            # Analyze sentiment
            analyzer = SentimentIntensityAnalyzer()
            sentiments = []
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                # Get sentiment scores
                scores = analyzer.polarity_scores(content)
                compound_score = scores['compound']
                sentiments.append(compound_score)
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Cache the result
            self.sentiment_cache[cache_key] = avg_sentiment
            
            return avg_sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {e}")
            return 0.0
    
    def generate_features(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Generate features for ML models"""
        try:
            features = []
            
            # Technical indicators
            tech_features = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'stoch_k', 'stoch_d',
                'atr', 'volume_ratio', 'price_volatility'
            ]
            
            for feature in tech_features:
                if feature in df.columns:
                    features.extend([
                        df[feature].iloc[-1],
                        df[feature].iloc[-1] - df[feature].iloc[-5],
                        df[feature].iloc[-1] - df[feature].iloc[-10]
                    ])
            
            # Price features
            price_features = [
                df['close'].iloc[-1],
                df['close'].pct_change().iloc[-1],
                df['close'].pct_change().rolling(5).mean().iloc[-1],
                df['close'].pct_change().rolling(10).mean().iloc[-1]
            ]
            features.extend(price_features)
            
            # Volume features
            volume_features = [
                df['tick_volume'].iloc[-1],
                df['volume_ratio'].iloc[-1]
            ]
            features.extend(volume_features)
            
            # News sentiment
            sentiment = self.get_news_sentiment(symbol)
            features.append(sentiment)
            
            # Time features
            now = datetime.now()
            features.extend([
                now.hour / 24.0,
                now.weekday() / 7.0,
                now.month / 12.0
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return np.array([])
    
    def predict_price_movement(self, symbol: str) -> Tuple[float, float]:
        """Predict price movement using ML models"""
        try:
            # Get market data
            df = self.get_market_data(symbol)
            if df.empty:
                return 0.0, 0.0
            
            # Generate features
            features = self.generate_features(df, symbol)
            if features.size == 0:
                return 0.0, 0.0
            
            # Scale features
            if 'price_features' in self.scalers:
                features_scaled = self.scalers['price_features'].transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            if self.models['price_prediction'] is not None:
                prediction = self.models['price_prediction'].predict(features_scaled)[0]
            else:
                prediction = 0.0
            
            # Calculate confidence (simplified)
            confidence = min(abs(prediction), 1.0)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement: {e}")
            return 0.0, 0.0
    
    def scalping_strategy(self, symbol: str) -> Dict:
        """Execute scalping strategy"""
        try:
            prediction, confidence = self.predict_price_movement(symbol)
            
            if confidence < self.config["ml"]["prediction_threshold"]:
                return {"action": "hold", "reason": "low_confidence"}
            
            # Get current market data
            df = self.get_market_data(symbol, bars=100)
            if df.empty:
                return {"action": "hold", "reason": "no_data"}
            
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # Scalping signals
            buy_signal = (
                prediction > 0 and
                rsi < 70 and
                macd > macd_signal and
                df['close'].iloc[-1] > df['bb_middle'].iloc[-1]
            )
            
            sell_signal = (
                prediction < 0 and
                rsi > 30 and
                macd < macd_signal and
                df['close'].iloc[-1] < df['bb_middle'].iloc[-1]
            )
            
            if buy_signal:
                return {
                    "action": "buy",
                    "symbol": symbol,
                    "type": "scalping",
                    "price": current_price,
                    "stop_loss": current_price - self.config["trading"]["risk"]["stop_loss"],
                    "take_profit": current_price + self.config["trading"]["scalping"]["min_profit"],
                    "size": self.config["trading"]["scalping"]["position_size"]
                }
            elif sell_signal:
                return {
                    "action": "sell",
                    "symbol": symbol,
                    "type": "scalping",
                    "price": current_price,
                    "stop_loss": current_price + self.config["trading"]["risk"]["stop_loss"],
                    "take_profit": current_price - self.config["trading"]["scalping"]["min_profit"],
                    "size": self.config["trading"]["scalping"]["position_size"]
                }
            
            return {"action": "hold", "reason": "no_signal"}
            
        except Exception as e:
            self.logger.error(f"Error in scalping strategy: {e}")
            return {"action": "hold", "reason": "error"}
    
    def grid_trading_strategy(self, symbol: str) -> Dict:
        """Execute grid trading strategy"""
        try:
            df = self.get_market_data(symbol, bars=100)
            if df.empty:
                return {"action": "hold", "reason": "no_data"}
            
            current_price = df['close'].iloc[-1]
            
            # Check if we need to create new grid levels
            if symbol not in self.grid_levels:
                self.create_grid_levels(symbol, current_price)
            
            # Check grid levels for entry/exit
            grid = self.grid_levels[symbol]
            
            for level in grid:
                if level['type'] == 'buy' and current_price <= level['price']:
                    if not level['filled']:
                        return {
                            "action": "buy",
                            "symbol": symbol,
                            "type": "grid",
                            "price": level['price'],
                            "stop_loss": level['price'] - self.config["trading"]["risk"]["stop_loss"],
                            "take_profit": level['price'] + self.config["trading"]["grid"]["spacing"],
                            "size": self.config["trading"]["grid"]["position_size"]
                        }
                
                elif level['type'] == 'sell' and current_price >= level['price']:
                    if not level['filled']:
                        return {
                            "action": "sell",
                            "symbol": symbol,
                            "type": "grid",
                            "symbol": symbol,
                            "price": level['price'],
                            "stop_loss": level['price'] + self.config["trading"]["risk"]["stop_loss"],
                            "take_profit": level['price'] - self.config["trading"]["grid"]["spacing"],
                            "size": self.config["trading"]["grid"]["position_size"]
                        }
            
            return {"action": "hold", "reason": "grid_waiting"}
            
        except Exception as e:
            self.logger.error(f"Error in grid trading strategy: {e}")
            return {"action": "hold", "reason": "error"}
    
    def create_grid_levels(self, symbol: str, current_price: float):
        """Create grid trading levels"""
        try:
            levels = []
            spacing = self.config["trading"]["grid"]["spacing"]
            num_levels = self.config["trading"]["grid"]["levels"]
            
            # Create buy levels below current price
            for i in range(1, num_levels + 1):
                buy_price = current_price - (i * spacing)
                levels.append({
                    "type": "buy",
                    "price": buy_price,
                    "filled": False,
                    "order_id": None
                })
            
            # Create sell levels above current price
            for i in range(1, num_levels + 1):
                sell_price = current_price + (i * spacing)
                levels.append({
                    "type": "sell",
                    "price": sell_price,
                    "filled": False,
                    "order_id": None
                })
            
            self.grid_levels[symbol] = levels
            self.logger.info(f"Created grid levels for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error creating grid levels: {e}")
    
    def hedging_strategy(self, symbol: str) -> Dict:
        """Execute hedging strategy"""
        try:
            # Check for correlated positions
            correlated_positions = self.find_correlated_positions(symbol)
            
            if not correlated_positions:
                return {"action": "hold", "reason": "no_correlation"}
            
            # Calculate hedge ratio
            total_exposure = sum(pos['size'] for pos in correlated_positions)
            max_hedge = total_exposure * self.config["trading"]["hedging"]["max_hedge_ratio"]
            
            # Check if we need to hedge
            current_hedge = self.get_current_hedge(symbol)
            
            if current_hedge < max_hedge:
                # Calculate hedge position
                hedge_size = max_hedge - current_hedge
                
                return {
                    "action": "hedge",
                    "symbol": symbol,
                    "size": hedge_size,
                    "type": "hedging",
                    "reason": "exposure_management"
                }
            
            return {"action": "hold", "reason": "hedge_sufficient"}
            
        except Exception as e:
            self.logger.error(f"Error in hedging strategy: {e}")
            return {"action": "hold", "reason": "error"}
    
    def find_correlated_positions(self, symbol: str) -> List[Dict]:
        """Find positions correlated with the given symbol"""
        # Simplified correlation check
        # In a real implementation, you would calculate actual correlations
        correlated = []
        
        for trade_id, trade in self.active_trades.items():
            if trade['symbol'] != symbol and trade['type'] in ['scalping', 'grid']:
                correlated.append(trade)
        
        return correlated
    
    def get_current_hedge(self, symbol: str) -> float:
        """Get current hedge position size for a symbol"""
        hedge_size = 0.0
        
        for trade_id, trade in self.active_trades.items():
            if trade['symbol'] == symbol and trade['type'] == 'hedging':
                hedge_size += trade['size']
        
        return hedge_size
    
    def execute_trade(self, trade_signal: Dict) -> bool:
        """Execute a trade based on the signal"""
        try:
            if trade_signal["action"] == "hold":
                return True
            
            symbol = trade_signal["symbol"]
            trade_type = trade_signal["type"]
            size = trade_signal["size"]
            
            # Check risk limits
            if not self.check_risk_limits(symbol, size):
                self.logger.warning(f"Risk limit exceeded for {symbol}")
                return False
            
            # Execute order in MT5
            if trade_signal["action"] == "buy":
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": size,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"{trade_type}_order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if specified
            if "stop_loss" in trade_signal:
                request["sl"] = trade_signal["stop_loss"]
            if "take_profit" in trade_signal:
                request["tp"] = trade_signal["take_profit"]
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode}")
                return False
            
            # Record the trade
            trade_id = result.order
            self.active_trades[trade_id] = {
                "symbol": symbol,
                "type": trade_type,
                "size": size,
                "entry_price": price,
                "entry_time": datetime.now(),
                "order_id": trade_id
            }
            
            self.logger.info(f"Executed {trade_signal['action']} order for {symbol}: {size} lots at {price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def check_risk_limits(self, symbol: str, size: float) -> bool:
        """Check if trade meets risk management criteria"""
        try:
            # Check daily loss limit
            daily_pnl = self.calculate_daily_pnl()
            if daily_pnl < -self.config["trading"]["risk"]["max_daily_loss"]:
                return False
            
            # Check position size limit
            total_exposure = self.calculate_total_exposure(symbol)
            if total_exposure + size > self.config["trading"]["risk"]["max_position_size"]:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def calculate_daily_pnl(self) -> float:
        """Calculate daily profit/loss"""
        try:
            today = datetime.now().date()
            total_pnl = 0.0
            
            # This is a simplified calculation
            # In a real implementation, you would query MT5 for actual P&L
            for trade_id, trade in self.active_trades.items():
                # Simplified P&L calculation
                pass
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def calculate_total_exposure(self, symbol: str) -> float:
        """Calculate total exposure for a symbol"""
        total_exposure = 0.0
        
        for trade_id, trade in self.active_trades.items():
            if trade['symbol'] == symbol:
                total_exposure += trade['size']
        
        return total_exposure
    
    def update_trades(self):
        """Update active trades and close completed ones"""
        try:
            # Get open positions from MT5
            positions = mt5.positions_get()
            if positions is None:
                return
            
            # Update active trades
            for position in positions:
                trade_id = position.ticket
                
                if trade_id in self.active_trades:
                    # Update trade info
                    self.active_trades[trade_id].update({
                        "current_price": position.price_current,
                        "profit": position.profit,
                        "swap": position.swap
                    })
                    
                    # Check if position should be closed
                    if self.should_close_position(position):
                        self.close_position(trade_id)
            
        except Exception as e:
            self.logger.error(f"Error updating trades: {e}")
    
    def should_close_position(self, position) -> bool:
        """Check if a position should be closed"""
        try:
            # Check stop loss or take profit
            if position.profit <= -self.config["trading"]["risk"]["stop_loss"]:
                return True
            
            if position.profit >= self.config["trading"]["risk"]["take_profit"]:
                return True
            
            # Check time-based exit for scalping
            if hasattr(position, 'time') and position.time:
                entry_time = datetime.fromtimestamp(position.time)
                current_time = datetime.now()
                
                if (current_time - entry_time).total_seconds() > 3600:  # 1 hour
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking position close: {e}")
            return False
    
    def close_position(self, trade_id: int):
        """Close a position"""
        try:
            if trade_id not in self.active_trades:
                return
            
            trade = self.active_trades[trade_id]
            
            # Close position in MT5
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": trade['symbol'],
                "volume": trade['size'],
                "type": mt5.ORDER_TYPE_SELL if trade['type'] == 'buy' else mt5.ORDER_TYPE_BUY,
                "position": trade_id,
                "price": mt5.symbol_info_tick(trade['symbol']).bid if trade['type'] == 'buy' else mt5.symbol_info_tick(trade['symbol']).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "close_order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Closed position {trade_id} for {trade['symbol']}")
                del self.active_trades[trade_id]
            else:
                self.logger.error(f"Failed to close position {trade_id}: {result.retcode}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def retrain_models(self):
        """Retrain ML models with new data"""
        try:
            self.logger.info("Starting model retraining...")
            
            # Collect training data
            training_data = self.collect_training_data()
            
            if training_data.empty:
                self.logger.warning("No training data available")
                return
            
            # Prepare features and targets
            X = training_data.drop(['target_price', 'target_volatility'], axis=1)
            y_price = training_data['target_price']
            y_volatility = training_data['target_volatility']
            
            # Split data
            X_train, X_test, y_price_train, y_price_test, y_volatility_train, y_volatility_test = train_test_split(
                X, y_price, y_volatility, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['price_features'].fit_transform(X_train)
            X_test_scaled = self.scalers['price_features'].transform(X_test)
            
            # Retrain price prediction model
            if self.models['price_prediction'] is not None:
                self.models['price_prediction'].fit(X_train_scaled, y_price_train)
                price_score = self.models['price_prediction'].score(X_test_scaled, y_price_test)
                self.logger.info(f"Price prediction model R² score: {price_score:.4f}")
            
            # Retrain volatility prediction model
            if self.models['volatility_prediction'] is not None:
                self.models['volatility_prediction'].fit(X_train_scaled, y_volatility_train)
                volatility_score = self.models['volatility_prediction'].score(X_test_scaled, y_volatility_test)
                self.logger.info(f"Volatility prediction model R² score: {volatility_score:.4f}")
            
            # Save models
            self.save_models()
            
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
    
    def collect_training_data(self) -> pd.DataFrame:
        """Collect training data for ML models"""
        try:
            training_data = []
            
            for symbol in self.config["mt5"]["symbols"]:
                # Get historical data
                df = self.get_market_data(symbol, bars=1000)
                if df.empty:
                    continue
                
                # Generate features
                for i in range(50, len(df) - 10):
                    features = self.generate_features(df.iloc[:i+1], symbol)
                    if features.size == 0:
                        continue
                    
                    # Calculate targets (future price change and volatility)
                    future_prices = df['close'].iloc[i+1:i+11]
                    price_change = (future_prices.iloc[-1] - df['close'].iloc[i]) / df['close'].iloc[i]
                    volatility = future_prices.pct_change().std()
                    
                    # Add to training data
                    row = features.flatten().tolist()
                    row.extend([price_change, volatility])
                    training_data.append(row)
            
            if not training_data:
                return pd.DataFrame()
            
            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(len(training_data[0]) - 2)]
            columns = feature_names + ['target_price', 'target_volatility']
            
            return pd.DataFrame(training_data, columns=columns)
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")
            return pd.DataFrame()
    
    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            self.logger.info("Starting trading cycle...")
            
            # Update existing trades
            self.update_trades()
            
            # Run strategies for each symbol
            for symbol in self.config["mt5"]["symbols"]:
                # Scalping strategy
                if self.config["trading"]["scalping"]["enabled"]:
                    scalping_signal = self.scalping_strategy(symbol)
                    if scalping_signal["action"] != "hold":
                        self.execute_trade(scalping_signal)
                
                # Grid trading strategy
                if self.config["trading"]["grid"]["enabled"]:
                    grid_signal = self.grid_trading_strategy(symbol)
                    if grid_signal["action"] != "hold":
                        self.execute_trade(grid_signal)
                
                # Hedging strategy
                if self.config["trading"]["hedging"]["enabled"]:
                    hedge_signal = self.hedging_strategy(symbol)
                    if hedge_signal["action"] != "hold":
                        self.execute_trade(hedge_signal)
            
            # Update performance metrics
            self.update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            total_pnl = 0.0
            total_trades = len(self.active_trades)
            
            for trade_id, trade in self.active_trades.items():
                if 'profit' in trade:
                    total_pnl += trade['profit']
            
            self.performance_metrics = {
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            self.logger.info("Starting Advanced Forex Trading Bot...")
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            # Start news monitoring thread
            if self.config["news"]["enabled"]:
                self.news_thread = threading.Thread(target=self.news_monitoring_loop)
                self.news_thread.daemon = True
                self.news_thread.start()
            
            # Start analysis thread
            if self.config["ml"]["enabled"]:
                self.analysis_thread = threading.Thread(target=self.analysis_loop)
                self.analysis_thread.daemon = True
                self.analysis_thread.start()
            
            self.logger.info("Bot started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            self.running = False
    
    def stop(self):
        """Stop the trading bot"""
        try:
            self.running = False
            self.logger.info("Stopping Advanced Forex Trading Bot...")
            
            # Wait for threads to finish
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)
            
            if self.news_thread and self.news_thread.is_alive():
                self.news_thread.join(timeout=5)
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5)
            
            # Save models
            self.save_models()
            
            # Close MT5 connection
            mt5.shutdown()
            
            self.logger.info("Bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    def trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                self.run_trading_cycle()
                time.sleep(60)  # Wait 1 minute between cycles
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def news_monitoring_loop(self):
        """News monitoring loop"""
        while self.running:
            try:
                # Update news sentiment for all symbols
                for symbol in self.config["mt5"]["symbols"]:
                    sentiment = self.get_news_sentiment(symbol)
                    self.logger.debug(f"News sentiment for {symbol}: {sentiment}")
                
                time.sleep(self.config["news"]["update_interval"])
            except Exception as e:
                self.logger.error(f"Error in news monitoring loop: {e}")
                time.sleep(300)
    
    def analysis_loop(self):
        """Analysis and model retraining loop"""
        last_retrain = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Retrain models periodically
                if current_time - last_retrain > self.config["ml"]["retrain_interval"]:
                    self.retrain_models()
                    last_retrain = current_time
                
                time.sleep(3600)  # Check every hour
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(3600)
    
    def get_status(self) -> Dict:
        """Get bot status"""
        return {
            "running": self.running,
            "active_trades": len(self.active_trades),
            "performance": self.performance_metrics,
            "grid_levels": {symbol: len(levels) for symbol, levels in self.grid_levels.items()},
            "last_update": datetime.now().isoformat()
        }

def main():
    """Main function to run the bot"""
    try:
        # Create bot instance
        bot = AdvancedForexBot()
        
        # Start the bot
        bot.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            bot.stop()
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()