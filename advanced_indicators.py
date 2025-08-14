#!/usr/bin/env python3
"""
Advanced Mathematical Indicators and Analysis Tools
Advanced technical indicators and mathematical analysis for forex trading
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedMathematicalIndicators:
    """
    Advanced Mathematical Indicators and Analysis Tools
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        
    def _default_config(self):
        """Default configuration for indicators"""
        return {
            'smoothing_factor': 0.1,
            'confidence_level': 0.95,
            'outlier_threshold': 3.0,
            'trend_strength_threshold': 0.7
        }
    
    def calculate_adaptive_moving_average(self, prices, period=20, sensitivity=0.1):
        """
        Calculate Adaptive Moving Average (AMA)
        
        Args:
            prices: Price series
            period: Lookback period
            sensitivity: Smoothing sensitivity
            
        Returns:
            Series with AMA values
        """
        try:
            ama = pd.Series(index=prices.index, dtype=float)
            
            # Calculate efficiency ratio
            price_change = prices.diff(period)
            volatility = prices.diff().abs().rolling(period).sum()
            
            # Avoid division by zero
            efficiency_ratio = np.where(volatility != 0, price_change / volatility, 0)
            
            # Calculate fast and slow smoothing constants
            fast_constant = 2 / (2 + 1)
            slow_constant = 2 / (period + 1)
            
            # Calculate adaptive constant
            adaptive_constant = efficiency_ratio * (fast_constant - slow_constant) + slow_constant
            adaptive_constant = adaptive_constant ** 2  # Square for stronger effect
            
            # Calculate AMA
            ama.iloc[0] = prices.iloc[0]
            for i in range(1, len(prices)):
                if not np.isnan(adaptive_constant.iloc[i-1]):
                    ama.iloc[i] = ama.iloc[i-1] + adaptive_constant.iloc[i-1] * (prices.iloc[i] - ama.iloc[i-1])
                else:
                    ama.iloc[i] = ama.iloc[i-1]
            
            return ama
            
        except Exception as e:
            print(f"Error calculating AMA: {e}")
            return pd.Series(prices.values, index=prices.index)
    
    def calculate_kaufman_adaptive_ma(self, prices, period=10, fast_ma=2, slow_ma=30):
        """
        Calculate Kaufman Adaptive Moving Average (KAMA)
        
        Args:
            prices: Price series
            period: Efficiency ratio period
            fast_ma: Fast EMA period
            slow_ma: Slow EMA period
            
        Returns:
            Series with KAMA values
        """
        try:
            kama = pd.Series(index=prices.index, dtype=float)
            
            # Calculate efficiency ratio
            price_change = prices.diff(period).abs()
            volatility = prices.diff().abs().rolling(period).sum()
            
            # Avoid division by zero
            efficiency_ratio = np.where(volatility != 0, price_change / volatility, 0)
            
            # Calculate smoothing constant
            fast_constant = 2 / (fast_ma + 1)
            slow_constant = 2 / (slow_ma + 1)
            smoothing_constant = (efficiency_ratio * (fast_constant - slow_constant) + slow_constant) ** 2
            
            # Calculate KAMA
            kama.iloc[0] = prices.iloc[0]
            for i in range(1, len(prices)):
                if not np.isnan(smoothing_constant.iloc[i-1]):
                    kama.iloc[i] = kama.iloc[i-1] + smoothing_constant.iloc[i-1] * (prices.iloc[i] - kama.iloc[i-1])
                else:
                    kama.iloc[i] = kama.iloc[i-1]
            
            return kama
            
        except Exception as e:
            print(f"Error calculating KAMA: {e}")
            return pd.Series(prices.values, index=prices.index)
    
    def calculate_vidya_ma(self, prices, period=20, alpha=0.2):
        """
        Calculate Variable Index Dynamic Average (VIDYA)
        
        Args:
            prices: Price series
            period: CMO period
            alpha: Smoothing factor
            
        Returns:
            Series with VIDYA values
        """
        try:
            vidya = pd.Series(index=prices.index, dtype=float)
            
            # Calculate Chande Momentum Oscillator (CMO)
            up_sum = prices.diff().where(prices.diff() > 0, 0).rolling(period).sum()
            down_sum = prices.diff().where(prices.diff() < 0, 0).abs().rolling(period).sum()
            
            # Calculate CMO
            cmo = np.where((up_sum + down_sum) != 0, (up_sum - down_sum) / (up_sum + down_sum), 0)
            
            # Calculate adaptive alpha
            adaptive_alpha = alpha * (1 + abs(cmo))
            
            # Calculate VIDYA
            vidya.iloc[0] = prices.iloc[0]
            for i in range(1, len(prices)):
                if not np.isnan(adaptive_alpha.iloc[i-1]):
                    vidya.iloc[i] = adaptive_alpha.iloc[i-1] * prices.iloc[i] + (1 - adaptive_alpha.iloc[i-1]) * vidya.iloc[i-1]
                else:
                    vidya.iloc[i] = vidya.iloc[i-1]
            
            return vidya
            
        except Exception as e:
            print(f"Error calculating VIDYA: {e}")
            return pd.Series(prices.values, index=prices.index)
    
    def calculate_adaptive_rsi(self, prices, period=14, sensitivity=0.1):
        """
        Calculate Adaptive RSI with dynamic period
        
        Args:
            prices: Price series
            period: Base period
            sensitivity: Volatility sensitivity
            
        Returns:
            Series with adaptive RSI values
        """
        try:
            # Calculate volatility
            returns = prices.pct_change()
            volatility = returns.rolling(period).std()
            
            # Calculate adaptive period
            adaptive_period = period * (1 + sensitivity * volatility * 100)
            adaptive_period = adaptive_period.round().astype(int)
            adaptive_period = adaptive_period.clip(lower=5, upper=50)  # Limit period range
            
            # Calculate adaptive RSI
            adaptive_rsi = pd.Series(index=prices.index, dtype=float)
            
            for i in range(len(prices)):
                if i < adaptive_period.iloc[i]:
                    adaptive_rsi.iloc[i] = 50  # Neutral value for insufficient data
                else:
                    # Calculate RSI for adaptive period
                    period_data = prices.iloc[i-adaptive_period.iloc[i]+1:i+1]
                    gains = period_data.diff().where(period_data.diff() > 0, 0)
                    losses = period_data.diff().where(period_data.diff() < 0, 0).abs()
                    
                    avg_gain = gains.mean()
                    avg_loss = losses.mean()
                    
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                        adaptive_rsi.iloc[i] = 100 - (100 / (1 + rs))
                    else:
                        adaptive_rsi.iloc[i] = 100
            
            return adaptive_rsi
            
        except Exception as e:
            print(f"Error calculating adaptive RSI: {e}")
            return pd.Series(50, index=prices.index)
    
    def calculate_adaptive_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Adaptive MACD with dynamic periods
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with adaptive MACD components
        """
        try:
            # Calculate volatility
            returns = prices.pct_change()
            volatility = returns.rolling(20).std()
            
            # Calculate adaptive periods based on volatility
            volatility_factor = volatility * 100
            
            adaptive_fast = (fast_period * (1 + volatility_factor)).round().astype(int)
            adaptive_slow = (slow_period * (1 + volatility_factor)).round().astype(int)
            adaptive_signal = (signal_period * (1 + volatility_factor)).round().astype(int)
            
            # Limit period ranges
            adaptive_fast = adaptive_fast.clip(lower=5, upper=20)
            adaptive_slow = adaptive_slow.clip(lower=20, upper=50)
            adaptive_signal = adaptive_signal.clip(lower=5, upper=20)
            
            # Calculate adaptive EMAs
            adaptive_fast_ema = prices.ewm(span=adaptive_fast.iloc[0]).mean()
            adaptive_slow_ema = prices.ewm(span=adaptive_slow.iloc[0]).mean()
            
            # Calculate MACD line
            macd_line = adaptive_fast_ema - adaptive_slow_ema
            
            # Calculate adaptive signal line
            signal_line = macd_line.ewm(span=adaptive_signal.iloc[0]).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'adaptive_fast_period': adaptive_fast,
                'adaptive_slow_period': adaptive_slow,
                'adaptive_signal_period': adaptive_signal
            }
            
        except Exception as e:
            print(f"Error calculating adaptive MACD: {e}")
            return {}
    
    def calculate_adaptive_bollinger_bands(self, prices, period=20, std_dev=2, sensitivity=0.1):
        """
        Calculate Adaptive Bollinger Bands with dynamic parameters
        
        Args:
            prices: Price series
            period: Base period
            std_dev: Base standard deviation multiplier
            sensitivity: Volatility sensitivity
            
        Returns:
            Dictionary with adaptive Bollinger Bands
        """
        try:
            # Calculate volatility
            returns = prices.pct_change()
            volatility = returns.rolling(period).std()
            
            # Calculate adaptive parameters
            volatility_factor = volatility * 100
            
            adaptive_period = (period * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period = adaptive_period.clip(lower=10, upper=50)
            
            adaptive_std = std_dev * (1 + sensitivity * volatility_factor)
            adaptive_std = adaptive_std.clip(lower=1.0, upper=4.0)
            
            # Calculate adaptive moving average
            adaptive_ma = prices.rolling(window=adaptive_period.iloc[0]).mean()
            
            # Calculate adaptive standard deviation
            adaptive_std_dev = prices.rolling(window=adaptive_period.iloc[0]).std()
            
            # Calculate bands
            upper_band = adaptive_ma + (adaptive_std * adaptive_std_dev)
            lower_band = adaptive_ma - (adaptive_std * adaptive_std_dev)
            
            # Calculate bandwidth and %B
            bandwidth = (upper_band - lower_band) / adaptive_ma
            percent_b = (prices - lower_band) / (upper_band - lower_band)
            
            return {
                'upper_band': upper_band,
                'middle_band': adaptive_ma,
                'lower_band': lower_band,
                'bandwidth': bandwidth,
                'percent_b': percent_b,
                'adaptive_period': adaptive_period,
                'adaptive_std': adaptive_std
            }
            
        except Exception as e:
            print(f"Error calculating adaptive Bollinger Bands: {e}")
            return {}
    
    def calculate_adaptive_stochastic(self, high, low, close, k_period=14, d_period=3, sensitivity=0.1):
        """
        Calculate Adaptive Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
            sensitivity: Volatility sensitivity
            
        Returns:
            Dictionary with adaptive stochastic components
        """
        try:
            # Calculate volatility
            returns = close.pct_change()
            volatility = returns.rolling(k_period).std()
            
            # Calculate adaptive period
            volatility_factor = volatility * 100
            adaptive_k_period = (k_period * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_k_period = adaptive_k_period.clip(lower=5, upper=30)
            
            # Calculate adaptive %K
            adaptive_k = pd.Series(index=close.index, dtype=float)
            
            for i in range(len(close)):
                if i < adaptive_k_period.iloc[i]:
                    adaptive_k.iloc[i] = 50  # Neutral value
                else:
                    period_data_high = high.iloc[i-adaptive_k_period.iloc[i]+1:i+1]
                    period_data_low = low.iloc[i-adaptive_k_period.iloc[i]+1:i+1]
                    period_data_close = close.iloc[i-adaptive_k_period.iloc[i]+1:i+1]
                    
                    highest_high = period_data_high.max()
                    lowest_low = period_data_low.min()
                    
                    if highest_high != lowest_low:
                        adaptive_k.iloc[i] = 100 * (period_data_close.iloc[-1] - lowest_low) / (highest_high - lowest_low)
                    else:
                        adaptive_k.iloc[i] = 50
            
            # Calculate adaptive %D
            adaptive_d = adaptive_k.rolling(window=d_period).mean()
            
            return {
                'k_percent': adaptive_k,
                'd_percent': adaptive_d,
                'adaptive_k_period': adaptive_k_period
            }
            
        except Exception as e:
            print(f"Error calculating adaptive stochastic: {e}")
            return {}
    
    def calculate_adaptive_williams_r(self, high, low, close, period=14, sensitivity=0.1):
        """
        Calculate Adaptive Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Base period
            sensitivity: Volatility sensitivity
            
        Returns:
            Series with adaptive Williams %R values
        """
        try:
            # Calculate volatility
            returns = close.pct_change()
            volatility = returns.rolling(period).std()
            
            # Calculate adaptive period
            volatility_factor = volatility * 100
            adaptive_period = (period * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period = adaptive_period.clip(lower=5, upper=30)
            
            # Calculate adaptive Williams %R
            adaptive_williams_r = pd.Series(index=close.index, dtype=float)
            
            for i in range(len(close)):
                if i < adaptive_period.iloc[i]:
                    adaptive_williams_r.iloc[i] = -50  # Neutral value
                else:
                    period_data_high = high.iloc[i-adaptive_period.iloc[i]+1:i+1]
                    period_data_low = low.iloc[i-adaptive_period.iloc[i]+1:i+1]
                    period_data_close = close.iloc[i-adaptive_period.iloc[i]+1:i+1]
                    
                    highest_high = period_data_high.max()
                    lowest_low = period_data_low.min()
                    
                    if highest_high != lowest_low:
                        adaptive_williams_r.iloc[i] = -100 * (highest_high - period_data_close.iloc[-1]) / (highest_high - lowest_low)
                    else:
                        adaptive_williams_r.iloc[i] = -50
            
            return adaptive_williams_r
            
        except Exception as e:
            print(f"Error calculating adaptive Williams %R: {e}")
            return pd.Series(-50, index=close.index)
    
    def calculate_adaptive_cci(self, high, low, close, period=20, sensitivity=0.1):
        """
        Calculate Adaptive Commodity Channel Index (CCI)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Base period
            sensitivity: Volatility sensitivity
            
        Returns:
            Series with adaptive CCI values
        """
        try:
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate volatility
            returns = typical_price.pct_change()
            volatility = returns.rolling(period).std()
            
            # Calculate adaptive period
            volatility_factor = volatility * 100
            adaptive_period = (period * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period = adaptive_period.clip(lower=10, upper=50)
            
            # Calculate adaptive CCI
            adaptive_cci = pd.Series(index=close.index, dtype=float)
            
            for i in range(len(close)):
                if i < adaptive_period.iloc[i]:
                    adaptive_cci.iloc[i] = 0  # Neutral value
                else:
                    period_data = typical_price.iloc[i-adaptive_period.iloc[i]+1:i+1]
                    sma = period_data.mean()
                    mean_deviation = period_data.apply(lambda x: abs(x - sma)).mean()
                    
                    if mean_deviation != 0:
                        adaptive_cci.iloc[i] = (period_data.iloc[-1] - sma) / (0.015 * mean_deviation)
                    else:
                        adaptive_cci.iloc[i] = 0
            
            return adaptive_cci
            
        except Exception as e:
            print(f"Error calculating adaptive CCI: {e}")
            return pd.Series(0, index=close.index)
    
    def calculate_adaptive_momentum(self, prices, period=10, sensitivity=0.1):
        """
        Calculate Adaptive Momentum Oscillator
        
        Args:
            prices: Price series
            period: Base period
            sensitivity: Volatility sensitivity
            
        Returns:
            Series with adaptive momentum values
        """
        try:
            # Calculate volatility
            returns = prices.pct_change()
            volatility = returns.rolling(period).std()
            
            # Calculate adaptive period
            volatility_factor = volatility * 100
            adaptive_period = (period * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period = adaptive_period.clip(lower=5, upper=30)
            
            # Calculate adaptive momentum
            adaptive_momentum = pd.Series(index=prices.index, dtype=float)
            
            for i in range(len(prices)):
                if i < adaptive_period.iloc[i]:
                    adaptive_momentum.iloc[i] = 0  # Neutral value
                else:
                    current_price = prices.iloc[i]
                    past_price = prices.iloc[i-adaptive_period.iloc[i]]
                    adaptive_momentum.iloc[i] = current_price - past_price
            
            return adaptive_momentum
            
        except Exception as e:
            print(f"Error calculating adaptive momentum: {e}")
            return pd.Series(0, index=prices.index)
    
    def calculate_adaptive_rate_of_change(self, prices, period=10, sensitivity=0.1):
        """
        Calculate Adaptive Rate of Change (ROC)
        
        Args:
            prices: Price series
            period: Base period
            sensitivity: Volatility sensitivity
            
        Returns:
            Series with adaptive ROC values
        """
        try:
            # Calculate volatility
            returns = prices.pct_change()
            volatility = returns.rolling(period).std()
            
            # Calculate adaptive period
            volatility_factor = volatility * 100
            adaptive_period = (period * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period = adaptive_period.clip(lower=5, upper=30)
            
            # Calculate adaptive ROC
            adaptive_roc = pd.Series(index=prices.index, dtype=float)
            
            for i in range(len(prices)):
                if i < adaptive_period.iloc[i]:
                    adaptive_roc.iloc[i] = 0  # Neutral value
                else:
                    current_price = prices.iloc[i]
                    past_price = prices.iloc[i-adaptive_period.iloc[i]]
                    
                    if past_price != 0:
                        adaptive_roc.iloc[i] = ((current_price - past_price) / past_price) * 100
                    else:
                        adaptive_roc.iloc[i] = 0
            
            return adaptive_roc
            
        except Exception as e:
            print(f"Error calculating adaptive ROC: {e}")
            return pd.Series(0, index=prices.index)
    
    def calculate_adaptive_ultimate_oscillator(self, high, low, close, period1=7, period2=14, period3=28, sensitivity=0.1):
        """
        Calculate Adaptive Ultimate Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period1: Short period
            period2: Medium period
            period3: Long period
            sensitivity: Volatility sensitivity
            
        Returns:
            Series with adaptive ultimate oscillator values
        """
        try:
            # Calculate volatility
            returns = close.pct_change()
            volatility = returns.rolling(period3).std()
            
            # Calculate adaptive periods
            volatility_factor = volatility * 100
            
            adaptive_period1 = (period1 * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period2 = (period2 * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period3 = (period3 * (1 + sensitivity * volatility_factor)).round().astype(int)
            
            # Limit period ranges
            adaptive_period1 = adaptive_period1.clip(lower=3, upper=15)
            adaptive_period2 = adaptive_period2.clip(lower=10, upper=25)
            adaptive_period3 = adaptive_period3.clip(lower=20, upper=40)
            
            # Calculate buying and selling pressure
            buying_pressure = close - np.minimum(low.shift(1), close)
            selling_pressure = np.maximum(high.shift(1), close) - close
            
            # Calculate adaptive averages
            bp1 = buying_pressure.rolling(window=adaptive_period1.iloc[0]).sum()
            sp1 = selling_pressure.rolling(window=adaptive_period1.iloc[0]).sum()
            
            bp2 = buying_pressure.rolling(window=adaptive_period2.iloc[0]).sum()
            sp2 = selling_pressure.rolling(window=adaptive_period2.iloc[0]).sum()
            
            bp3 = buying_pressure.rolling(window=adaptive_period3.iloc[0]).sum()
            sp3 = selling_pressure.rolling(window=adaptive_period3.iloc[0]).sum()
            
            # Calculate ultimate oscillator
            ultimate_oscillator = 100 * ((4 * bp1 / sp1) + (2 * bp2 / sp2) + (bp3 / sp3)) / 7
            
            # Handle division by zero
            ultimate_oscillator = ultimate_oscillator.replace([np.inf, -np.inf], 50)
            ultimate_oscillator = ultimate_oscillator.fillna(50)
            
            return ultimate_oscillator
            
        except Exception as e:
            print(f"Error calculating adaptive ultimate oscillator: {e}")
            return pd.Series(50, index=close.index)
    
    def calculate_adaptive_aroon(self, high, low, period=25, sensitivity=0.1):
        """
        Calculate Adaptive Aroon Oscillator
        
        Args:
            high: High price series
            low: Low price series
            period: Base period
            sensitivity: Volatility sensitivity
            
        Returns:
            Dictionary with adaptive Aroon components
        """
        try:
            # Calculate volatility
            returns = (high + low) / 2
            returns = returns.pct_change()
            volatility = returns.rolling(period).std()
            
            # Calculate adaptive period
            volatility_factor = volatility * 100
            adaptive_period = (period * (1 + sensitivity * volatility_factor)).round().astype(int)
            adaptive_period = adaptive_period.clip(lower=10, upper=50)
            
            # Calculate adaptive Aroon Up and Down
            aroon_up = pd.Series(index=high.index, dtype=float)
            aroon_down = pd.Series(index=low.index, dtype=float)
            
            for i in range(len(high)):
                if i < adaptive_period.iloc[i]:
                    aroon_up.iloc[i] = 50
                    aroon_down.iloc[i] = 50
                else:
                    # Aroon Up
                    period_high = high.iloc[i-adaptive_period.iloc[i]+1:i+1]
                    days_since_high = adaptive_period.iloc[i] - period_high.values.argmax() - 1
                    aroon_up.iloc[i] = ((adaptive_period.iloc[i] - days_since_high) / adaptive_period.iloc[i]) * 100
                    
                    # Aroon Down
                    period_low = low.iloc[i-adaptive_period.iloc[i]+1:i+1]
                    days_since_low = adaptive_period.iloc[i] - period_low.values.argmin() - 1
                    aroon_down.iloc[i] = ((adaptive_period.iloc[i] - days_since_low) / adaptive_period.iloc[i]) * 100
            
            # Calculate Aroon Oscillator
            aroon_oscillator = aroon_up - aroon_down
            
            return {
                'aroon_up': aroon_up,
                'aroon_down': aroon_down,
                'aroon_oscillator': aroon_oscillator,
                'adaptive_period': adaptive_period
            }
            
        except Exception as e:
            print(f"Error calculating adaptive Aroon: {e}")
            return {}
    
    def get_all_adaptive_indicators(self, df):
        """
        Calculate all adaptive indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all adaptive indicators
        """
        try:
            result_df = df.copy()
            
            # Price-based indicators
            if 'close' in df.columns:
                result_df['adaptive_ama'] = self.calculate_adaptive_moving_average(df['close'])
                result_df['adaptive_kama'] = self.calculate_kaufman_adaptive_ma(df['close'])
                result_df['adaptive_vidya'] = self.calculate_vidya_ma(df['close'])
                result_df['adaptive_rsi'] = self.calculate_adaptive_rsi(df['close'])
                result_df['adaptive_momentum'] = self.calculate_adaptive_momentum(df['close'])
                result_df['adaptive_roc'] = self.calculate_adaptive_rate_of_change(df['close'])
            
            # MACD
            if all(col in df.columns for col in ['close']):
                macd_result = self.calculate_adaptive_macd(df['close'])
                if macd_result:
                    result_df['adaptive_macd'] = macd_result['macd_line']
                    result_df['adaptive_macd_signal'] = macd_result['signal_line']
                    result_df['adaptive_macd_histogram'] = macd_result['histogram']
            
            # Bollinger Bands
            if 'close' in df.columns:
                bb_result = self.calculate_adaptive_bollinger_bands(df['close'])
                if bb_result:
                    result_df['adaptive_bb_upper'] = bb_result['upper_band']
                    result_df['adaptive_bb_middle'] = bb_result['middle_band']
                    result_df['adaptive_bb_lower'] = bb_result['lower_band']
                    result_df['adaptive_bb_bandwidth'] = bb_result['bandwidth']
                    result_df['adaptive_bb_percent_b'] = bb_result['percent_b']
            
            # Stochastic
            if all(col in df.columns for col in ['high', 'low', 'close']):
                stoch_result = self.calculate_adaptive_stochastic(df['high'], df['low'], df['close'])
                if stoch_result:
                    result_df['adaptive_stoch_k'] = stoch_result['k_percent']
                    result_df['adaptive_stoch_d'] = stoch_result['d_percent']
            
            # Williams %R
            if all(col in df.columns for col in ['high', 'low', 'close']):
                result_df['adaptive_williams_r'] = self.calculate_adaptive_williams_r(df['high'], df['low'], df['close'])
            
            # CCI
            if all(col in df.columns for col in ['high', 'low', 'close']):
                result_df['adaptive_cci'] = self.calculate_adaptive_cci(df['high'], df['low'], df['close'])
            
            # Ultimate Oscillator
            if all(col in df.columns for col in ['high', 'low', 'close']):
                result_df['adaptive_ultimate_osc'] = self.calculate_adaptive_ultimate_oscillator(df['high'], df['low'], df['close'])
            
            # Aroon
            if all(col in df.columns for col in ['high', 'low']):
                aroon_result = self.calculate_adaptive_aroon(df['high'], df['low'])
                if aroon_result:
                    result_df['adaptive_aroon_up'] = aroon_result['aroon_up']
                    result_df['adaptive_aroon_down'] = aroon_result['aroon_down']
                    result_df['adaptive_aroon_oscillator'] = aroon_result['aroon_oscillator']
            
            return result_df
            
        except Exception as e:
            print(f"Error calculating all adaptive indicators: {e}")
            return df

if __name__ == "__main__":
    # Example usage
    indicators = AdvancedMathematicalIndicators()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.normal(1.1000, 0.005, len(dates)),
        'high': np.random.normal(1.1050, 0.005, len(dates)),
        'low': np.random.normal(1.0950, 0.005, len(dates)),
        'close': np.random.normal(1.1000, 0.005, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure high > low and close is between them
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.002, len(dates)))
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.002, len(dates)))
    
    print("Sample data created successfully!")
    print(f"Data shape: {sample_data.shape}")
    
    # Calculate all adaptive indicators
    result_df = indicators.get_all_adaptive_indicators(sample_data)
    
    print("\nAdaptive indicators calculated successfully!")
    print(f"Result shape: {result_df.shape}")
    print(f"New columns: {[col for col in result_df.columns if 'adaptive' in col]}")
    
    # Display sample results
    print("\nSample results (last 5 rows):")
    adaptive_columns = [col for col in result_df.columns if 'adaptive' in col]
    print(result_df[adaptive_columns].tail())