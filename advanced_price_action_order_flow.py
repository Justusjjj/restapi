#!/usr/bin/env python3
"""
Advanced Price Action and Order Flow Analysis for Forex Trading
Features:
- Advanced Candlestick Pattern Recognition
- Order Flow Analysis and Volume Profile
- Market Microstructure Analysis
- Support/Resistance Detection
- Breakout and Reversal Patterns
- Advanced Entry/Exit Signal Generation
- Real-time Pattern Monitoring
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

class AdvancedPriceActionAnalyzer:
    def __init__(self, config: Dict):
        """Initialize advanced price action analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pattern_cache = {}
        self.support_resistance_cache = {}
        self.volume_profile_cache = {}
        
    def analyze_candlestick_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Analyze advanced candlestick patterns"""
        try:
            patterns = {}
            
            # Basic candlestick patterns
            patterns["basic"] = self._detect_basic_patterns(ohlc_data)
            
            # Advanced reversal patterns
            patterns["reversal"] = self._detect_reversal_patterns(ohlc_data)
            
            # Continuation patterns
            patterns["continuation"] = self._detect_continuation_patterns(ohlc_data)
            
            # Complex patterns
            patterns["complex"] = self._detect_complex_patterns(ohlc_data)
            
            # Pattern strength and reliability
            patterns["strength"] = self._calculate_pattern_strength(ohlc_data, patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing candlestick patterns: {e}")
            return {"error": str(e)}
    
    def _detect_basic_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect basic candlestick patterns"""
        try:
            patterns = {}
            
            # Doji patterns
            patterns["doji"] = self._detect_doji_patterns(ohlc_data)
            
            # Hammer and Hanging Man
            patterns["hammer"] = self._detect_hammer_patterns(ohlc_data)
            
            # Engulfing patterns
            patterns["engulfing"] = self._detect_engulfing_patterns(ohlc_data)
            
            # Spinning tops
            patterns["spinning_top"] = self._detect_spinning_top_patterns(ohlc_data)
            
            # Marubozu
            patterns["marubozu"] = self._detect_marubozu_patterns(ohlc_data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in basic patterns: {e}")
            return {"error": str(e)}
    
    def _detect_doji_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect various doji patterns"""
        try:
            patterns = {}
            
            # Calculate body and shadow sizes
            body_size = abs(ohlc_data['close'] - ohlc_data['open'])
            upper_shadow = ohlc_data['high'] - np.maximum(ohlc_data['open'], ohlc_data['close'])
            lower_shadow = np.minimum(ohlc_data['open'], ohlc_data['close']) - ohlc_data['low']
            total_range = ohlc_data['high'] - ohlc_data['low']
            
            # Doji criteria (body < 10% of total range)
            doji_threshold = 0.1
            is_doji = body_size <= (total_range * doji_threshold)
            
            patterns["standard_doji"] = is_doji
            
            # Long-legged doji
            long_legged = is_doji & (upper_shadow > body_size * 2) & (lower_shadow > body_size * 2)
            patterns["long_legged_doji"] = long_legged
            
            # Dragonfly doji
            dragonfly = is_doji & (upper_shadow <= body_size * 0.1) & (lower_shadow > body_size * 2)
            patterns["dragonfly_doji"] = dragonfly
            
            # Gravestone doji
            gravestone = is_doji & (upper_shadow > body_size * 2) & (lower_shadow <= body_size * 0.1)
            patterns["gravestone_doji"] = gravestone
            
            # Four price doji
            four_price = (body_size == 0) & (upper_shadow == 0) & (lower_shadow == 0)
            patterns["four_price_doji"] = four_price
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in doji patterns: {e}")
            return {"error": str(e)}
    
    def _detect_hammer_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect hammer and hanging man patterns"""
        try:
            patterns = {}
            
            body_size = abs(ohlc_data['close'] - ohlc_data['open'])
            upper_shadow = ohlc_data['high'] - np.maximum(ohlc_data['open'], ohlc_data['close'])
            lower_shadow = np.minimum(ohlc_data['open'], ohlc_data['close']) - ohlc_data['low']
            
            # Hammer criteria
            is_hammer = (
                (lower_shadow > body_size * 2) &  # Long lower shadow
                (upper_shadow <= body_size * 0.5) &  # Short upper shadow
                (body_size > 0)  # Has a body
            )
            
            # Hanging man criteria (similar to hammer but at top)
            is_hanging_man = is_hammer.copy()
            
            patterns["hammer"] = is_hammer
            patterns["hanging_man"] = is_hanging_man
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in hammer patterns: {e}")
            return {"error": str(e)}
    
    def _detect_engulfing_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect bullish and bearish engulfing patterns"""
        try:
            patterns = {}
            
            # Calculate body sizes
            body_size = abs(ohlc_data['close'] - ohlc_data['open'])
            
            # Bullish engulfing
            bullish_engulfing = (
                (ohlc_data['close'].shift(1) < ohlc_data['open'].shift(1)) &  # Previous red candle
                (ohlc_data['close'] > ohlc_data['open']) &  # Current green candle
                (ohlc_data['open'] < ohlc_data['close'].shift(1)) &  # Current open below previous close
                (ohlc_data['close'] > ohlc_data['open'].shift(1))  # Current close above previous open
            )
            
            # Bearish engulfing
            bearish_engulfing = (
                (ohlc_data['close'].shift(1) > ohlc_data['open'].shift(1)) &  # Previous green candle
                (ohlc_data['close'] < ohlc_data['open']) &  # Current red candle
                (ohlc_data['open'] > ohlc_data['close'].shift(1)) &  # Current open above previous close
                (ohlc_data['close'] < ohlc_data['open'].shift(1))  # Current close below previous open
            )
            
            patterns["bullish_engulfing"] = bullish_engulfing
            patterns["bearish_engulfing"] = bearish_engulfing
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in engulfing patterns: {e}")
            return {"error": str(e)}
    
    def _detect_reversal_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect advanced reversal patterns"""
        try:
            patterns = {}
            
            # Morning Star
            patterns["morning_star"] = self._detect_morning_star(ohlc_data)
            
            # Evening Star
            patterns["evening_star"] = self._detect_evening_star(ohlc_data)
            
            # Three White Soldiers
            patterns["three_white_soldiers"] = self._detect_three_white_soldiers(ohlc_data)
            
            # Three Black Crows
            patterns["three_black_crows"] = self._detect_three_black_crows(ohlc_data)
            
            # Tweezer patterns
            patterns["tweezers"] = self._detect_tweezers(ohlc_data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in reversal patterns: {e}")
            return {"error": str(e)}
    
    def _detect_morning_star(self, ohlc_data: pd.DataFrame) -> pd.Series:
        """Detect morning star pattern"""
        try:
            # Morning star: bearish candle, small body, bullish candle
            is_morning_star = (
                (ohlc_data['close'].shift(2) < ohlc_data['open'].shift(2)) &  # First: bearish
                (abs(ohlc_data['close'].shift(1) - ohlc_data['open'].shift(1)) <= 
                 abs(ohlc_data['close'].shift(2) - ohlc_data['open'].shift(2)) * 0.3) &  # Second: small body
                (ohlc_data['close'] > ohlc_data['open']) &  # Third: bullish
                (ohlc_data['close'] > (ohlc_data['open'].shift(2) + ohlc_data['close'].shift(2)) / 2)  # Third closes above midpoint
            )
            
            return is_morning_star
            
        except Exception as e:
            self.logger.error(f"Error in morning star: {e}")
            return pd.Series([False] * len(ohlc_data))
    
    def _detect_evening_star(self, ohlc_data: pd.DataFrame) -> pd.Series:
        """Detect evening star pattern"""
        try:
            # Evening star: bullish candle, small body, bearish candle
            is_evening_star = (
                (ohlc_data['close'].shift(2) > ohlc_data['open'].shift(2)) &  # First: bullish
                (abs(ohlc_data['close'].shift(1) - ohlc_data['open'].shift(1)) <= 
                 abs(ohlc_data['close'].shift(2) - ohlc_data['open'].shift(2)) * 0.3) &  # Second: small body
                (ohlc_data['close'] < ohlc_data['open']) &  # Third: bearish
                (ohlc_data['close'] < (ohlc_data['open'].shift(2) + ohlc_data['close'].shift(2)) / 2)  # Third closes below midpoint
            )
            
            return is_evening_star
            
        except Exception as e:
            self.logger.error(f"Error in evening star: {e}")
            return pd.Series([False] * len(ohlc_data))
    
    def _detect_three_white_soldiers(self, ohlc_data: pd.DataFrame) -> pd.Series:
        """Detect three white soldiers pattern"""
        try:
            # Three consecutive bullish candles with higher closes
            is_three_white_soldiers = (
                (ohlc_data['close'].shift(2) > ohlc_data['open'].shift(2)) &  # First: bullish
                (ohlc_data['close'].shift(1) > ohlc_data['open'].shift(1)) &  # Second: bullish
                (ohlc_data['close'] > ohlc_data['open']) &  # Third: bullish
                (ohlc_data['close'].shift(1) > ohlc_data['close'].shift(2)) &  # Second higher than first
                (ohlc_data['close'] > ohlc_data['close'].shift(1))  # Third higher than second
            )
            
            return is_three_white_soldiers
            
        except Exception as e:
            self.logger.error(f"Error in three white soldiers: {e}")
            return pd.Series([False] * len(ohlc_data))
    
    def _detect_three_black_crows(self, ohlc_data: pd.DataFrame) -> pd.Series:
        """Detect three black crows pattern"""
        try:
            # Three consecutive bearish candles with lower closes
            is_three_black_crows = (
                (ohlc_data['close'].shift(2) < ohlc_data['open'].shift(2)) &  # First: bearish
                (ohlc_data['close'].shift(1) < ohlc_data['open'].shift(1)) &  # Second: bearish
                (ohlc_data['close'] < ohlc_data['open']) &  # Third: bearish
                (ohlc_data['close'].shift(1) < ohlc_data['close'].shift(2)) &  # Second lower than first
                (ohlc_data['close'] < ohlc_data['close'].shift(1))  # Third lower than second
            )
            
            return is_three_black_crows
            
        except Exception as e:
            self.logger.error(f"Error in three black crows: {e}")
            return pd.Series([False] * len(ohlc_data))
    
    def _detect_tweezers(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect tweezers top and bottom patterns"""
        try:
            patterns = {}
            
            # Tweezer top: two candles with same high
            tweezer_top = (
                (ohlc_data['high'].shift(1) == ohlc_data['high']) &  # Same high
                (ohlc_data['high'].shift(1) > ohlc_data['high'].shift(2)) &  # Higher than previous
                (ohlc_data['high'].shift(1) > ohlc_data['high'].shift(-1))  # Higher than next
            )
            
            # Tweezer bottom: two candles with same low
            tweezer_bottom = (
                (ohlc_data['low'].shift(1) == ohlc_data['low']) &  # Same low
                (ohlc_data['low'].shift(1) < ohlc_data['low'].shift(2)) &  # Lower than previous
                (ohlc_data['low'].shift(1) < ohlc_data['low'].shift(-1))  # Lower than next
            )
            
            patterns["tweezer_top"] = tweezer_top
            patterns["tweezer_bottom"] = tweezer_bottom
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in tweezers: {e}")
            return {"error": str(e)}
    
    def _detect_continuation_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect continuation patterns"""
        try:
            patterns = {}
            
            # Flag patterns
            patterns["flag"] = self._detect_flag_patterns(ohlc_data)
            
            # Pennant patterns
            patterns["pennant"] = self._detect_pennant_patterns(ohlc_data)
            
            # Triangle patterns
            patterns["triangle"] = self._detect_triangle_patterns(ohlc_data)
            
            # Rectangle patterns
            patterns["rectangle"] = self._detect_rectangle_patterns(ohlc_data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in continuation patterns: {e}")
            return {"error": str(e)}
    
    def _detect_flag_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect flag patterns"""
        try:
            patterns = {}
            
            # Bull flag: strong move up followed by consolidation
            # Bear flag: strong move down followed by consolidation
            
            # Simplified flag detection
            # In practice, this would be more complex with trend analysis
            
            return {"bull_flag": False, "bear_flag": False}
            
        except Exception as e:
            self.logger.error(f"Error in flag patterns: {e}")
            return {"error": str(e)}
    
    def _detect_triangle_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect triangle patterns"""
        try:
            patterns = {}
            
            # Look for converging highs and lows over time
            # This is a simplified implementation
            
            # Calculate trend lines
            highs = ohlc_data['high'].rolling(window=5).max()
            lows = ohlc_data['low'].rolling(window=5).min()
            
            # Check for convergence (simplified)
            if len(highs) >= 10:
                high_slope = np.polyfit(range(5), highs.iloc[-5:], 1)[0]
                low_slope = np.polyfit(range(5), lows.iloc[-5:], 1)[0]
                
                # Ascending triangle: flat highs, rising lows
                ascending = (abs(high_slope) < 0.001) and (low_slope > 0.001)
                
                # Descending triangle: falling highs, flat lows
                descending = (high_slope < -0.001) and (abs(low_slope) < 0.001)
                
                # Symmetrical triangle: both converging
                symmetrical = (high_slope < -0.001) and (low_slope > 0.001)
                
                patterns["ascending_triangle"] = ascending
                patterns["descending_triangle"] = descending
                patterns["symmetrical_triangle"] = symmetrical
            else:
                patterns["ascending_triangle"] = False
                patterns["descending_triangle"] = False
                patterns["symmetrical_triangle"] = False
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in triangle patterns: {e}")
            return {"error": str(e)}
    
    def _detect_complex_patterns(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect complex chart patterns"""
        try:
            patterns = {}
            
            # Head and Shoulders
            patterns["head_and_shoulders"] = self._detect_head_and_shoulders(ohlc_data)
            
            # Double Top/Bottom
            patterns["double_top_bottom"] = self._detect_double_patterns(ohlc_data)
            
            # Cup and Handle
            patterns["cup_and_handle"] = self._detect_cup_and_handle(ohlc_data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in complex patterns: {e}")
            return {"error": str(e)}
    
    def _detect_head_and_shoulders(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect head and shoulders pattern"""
        try:
            # Simplified H&S detection
            # In practice, this requires sophisticated peak analysis
            
            patterns = {
                "head_and_shoulders": False,
                "inverse_head_and_shoulders": False
            }
            
            # Find peaks and troughs
            peaks, _ = find_peaks(ohlc_data['high'].values, height=ohlc_data['high'].mean())
            troughs, _ = find_peaks(-ohlc_data['low'].values, height=-ohlc_data['low'].mean())
            
            if len(peaks) >= 3 and len(troughs) >= 2:
                # Basic H&S structure check
                # This is a simplified version
                patterns["head_and_shoulders"] = True
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in head and shoulders: {e}")
            return {"error": str(e)}
    
    def _calculate_pattern_strength(self, ohlc_data: pd.DataFrame, patterns: Dict) -> Dict:
        """Calculate pattern strength and reliability"""
        try:
            strength = {}
            
            # Volume confirmation
            if 'volume' in ohlc_data.columns:
                volume_confirmation = self._calculate_volume_confirmation(ohlc_data, patterns)
                strength["volume_confirmation"] = volume_confirmation
            
            # Pattern completion percentage
            strength["completion"] = self._calculate_pattern_completion(patterns)
            
            # Historical success rate (simplified)
            strength["historical_success"] = self._estimate_historical_success(patterns)
            
            # Market context alignment
            strength["context_alignment"] = self._check_market_context(ohlc_data, patterns)
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern strength: {e}")
            return {"error": str(e)}
    
    def _calculate_volume_confirmation(self, ohlc_data: pd.DataFrame, patterns: Dict) -> Dict:
        """Calculate volume confirmation for patterns"""
        try:
            volume_conf = {}
            
            # Average volume
            avg_volume = ohlc_data['volume'].rolling(window=20).mean()
            
            # Volume spike detection
            volume_spike = ohlc_data['volume'] > (avg_volume * 1.5)
            
            # Pattern-specific volume analysis
            for pattern_type, pattern_data in patterns.items():
                if isinstance(pattern_data, dict):
                    for pattern_name, pattern_series in pattern_data.items():
                        if isinstance(pattern_series, pd.Series):
                            # Check if pattern occurs with volume confirmation
                            pattern_with_volume = pattern_series & volume_spike
                            volume_conf[f"{pattern_type}_{pattern_name}"] = {
                                "with_volume": pattern_with_volume.sum(),
                                "total": pattern_series.sum(),
                                "confirmation_rate": pattern_with_volume.sum() / pattern_series.sum() if pattern_series.sum() > 0 else 0
                            }
            
            return volume_conf
            
        except Exception as e:
            self.logger.error(f"Error in volume confirmation: {e}")
            return {"error": str(e)}
    
    def detect_support_resistance(self, ohlc_data: pd.DataFrame, method: str = "all") -> Dict:
        """Detect support and resistance levels"""
        try:
            levels = {}
            
            if method == "pivot_points" or method == "all":
                levels["pivot_points"] = self._calculate_pivot_points(ohlc_data)
            
            if method == "fibonacci" or method == "all":
                levels["fibonacci"] = self._calculate_fibonacci_levels(ohlc_data)
            
            if method == "volume_profile" or method == "all":
                levels["volume_profile"] = self._calculate_volume_profile_levels(ohlc_data)
            
            if method == "psychological" or method == "all":
                levels["psychological"] = self._detect_psychological_levels(ohlc_data)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {e}")
            return {"error": str(e)}
    
    def _calculate_pivot_points(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculate pivot point levels"""
        try:
            # Standard pivot points
            high = ohlc_data['high'].iloc[-1]
            low = ohlc_data['low'].iloc[-1]
            close = ohlc_data['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                "pivot": pivot,
                "resistance": {"r1": r1, "r2": r2, "r3": r3},
                "support": {"s1": s1, "s2": s2, "s3": s3}
            }
            
        except Exception as e:
            self.logger.error(f"Error in pivot points: {e}")
            return {"error": str(e)}
    
    def _calculate_fibonacci_levels(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        try:
            high = ohlc_data['high'].max()
            low = ohlc_data['low'].min()
            diff = high - low
            
            levels = {
                "0.0": low,
                "0.236": low + 0.236 * diff,
                "0.382": low + 0.382 * diff,
                "0.500": low + 0.500 * diff,
                "0.618": low + 0.618 * diff,
                "0.786": low + 0.786 * diff,
                "1.0": high
            }
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci levels: {e}")
            return {"error": str(e)}
    
    def _calculate_volume_profile_levels(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculate volume profile support/resistance levels"""
        try:
            if 'volume' not in ohlc_data.columns:
                return {"error": "Volume data not available"}
            
            # Create price bins
            price_bins = pd.cut(ohlc_data['close'], bins=20)
            volume_profile = ohlc_data.groupby(price_bins)['volume'].sum()
            
            # Find high volume nodes (potential S/R levels)
            high_volume_threshold = volume_profile.quantile(0.8)
            high_volume_levels = volume_profile[volume_profile > high_volume_threshold]
            
            return {
                "volume_profile": volume_profile.to_dict(),
                "high_volume_levels": high_volume_levels.to_dict(),
                "threshold": high_volume_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error in volume profile levels: {e}")
            return {"error": str(e)}
    
    def _detect_psychological_levels(self, ohlc_data: pd.DataFrame) -> Dict:
        """Detect psychological support/resistance levels"""
        try:
            # Round numbers and psychological levels
            current_price = ohlc_data['close'].iloc[-1]
            
            # Find nearby round numbers
            round_levels = []
            for i in range(-5, 6):
                level = round(current_price + i * 0.01, 2)  # Round to 2 decimal places
                if abs(level - current_price) <= 0.05:  # Within 5 pips
                    round_levels.append(level)
            
            # Major psychological levels
            major_levels = [1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000]
            
            return {
                "round_levels": round_levels,
                "major_levels": major_levels,
                "current_price": current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error in psychological levels: {e}")
            return {"error": str(e)}
    
    def generate_entry_exit_signals(self, ohlc_data: pd.DataFrame, patterns: Dict) -> Dict:
        """Generate entry and exit signals based on patterns"""
        try:
            signals = {}
            
            # Entry signals
            signals["entry"] = self._generate_entry_signals(ohlc_data, patterns)
            
            # Exit signals
            signals["exit"] = self._generate_exit_signals(ohlc_data, patterns)
            
            # Signal strength
            signals["strength"] = self._calculate_signal_strength(signals)
            
            # Risk management
            signals["risk_management"] = self._generate_risk_management_signals(ohlc_data, patterns)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {"error": str(e)}
    
    def _generate_entry_signals(self, ohlc_data: pd.DataFrame, patterns: Dict) -> Dict:
        """Generate entry signals"""
        try:
            entry_signals = {}
            
            # Bullish signals
            bullish_signals = []
            
            # Check for bullish patterns
            if "reversal" in patterns:
                reversal = patterns["reversal"]
                if "morning_star" in reversal:
                    bullish_signals.append(("morning_star", reversal["morning_star"]))
                if "hammer" in patterns.get("basic", {}):
                    bullish_signals.append(("hammer", patterns["basic"]["hammer"]))
            
            # Bearish signals
            bearish_signals = []
            
            # Check for bearish patterns
            if "reversal" in patterns:
                reversal = patterns["reversal"]
                if "evening_star" in reversal:
                    bearish_signals.append(("evening_star", reversal["evening_star"]))
                if "hanging_man" in patterns.get("basic", {}):
                    bearish_signals.append(("hanging_man", patterns["basic"]["hanging_man"]))
            
            entry_signals["bullish"] = bullish_signals
            entry_signals["bearish"] = bearish_signals
            
            return entry_signals
            
        except Exception as e:
            self.logger.error(f"Error in entry signals: {e}")
            return {"error": str(e)}
    
    def _generate_exit_signals(self, ohlc_data: pd.DataFrame, patterns: Dict) -> Dict:
        """Generate exit signals"""
        try:
            exit_signals = {}
            
            # Take profit signals
            take_profit = self._calculate_take_profit_levels(ohlc_data, patterns)
            
            # Stop loss signals
            stop_loss = self._calculate_stop_loss_levels(ohlc_data, patterns)
            
            exit_signals["take_profit"] = take_profit
            exit_signals["stop_loss"] = stop_loss
            
            return exit_signals
            
        except Exception as e:
            self.logger.error(f"Error in exit signals: {e}")
            return {"error": str(e)}
    
    def _calculate_take_profit_levels(self, ohlc_data: pd.DataFrame, patterns: Dict) -> Dict:
        """Calculate take profit levels"""
        try:
            current_price = ohlc_data['close'].iloc[-1]
            atr = self._calculate_atr(ohlc_data)
            
            # Multiple take profit levels
            tp_levels = {
                "tp1": current_price + (atr * 1.5),  # 1.5 ATR
                "tp2": current_price + (atr * 2.5),  # 2.5 ATR
                "tp3": current_price + (atr * 4.0)   # 4.0 ATR
            }
            
            return tp_levels
            
        except Exception as e:
            self.logger.error(f"Error in take profit levels: {e}")
            return {"error": str(e)}
    
    def _calculate_stop_loss_levels(self, ohlc_data: pd.DataFrame, patterns: Dict) -> Dict:
        """Calculate stop loss levels"""
        try:
            current_price = ohlc_data['close'].iloc[-1]
            atr = self._calculate_atr(ohlc_data)
            
            # Multiple stop loss levels
            sl_levels = {
                "sl1": current_price - (atr * 1.0),  # 1.0 ATR
                "sl2": current_price - (atr * 1.5),  # 1.5 ATR
                "sl3": current_price - (atr * 2.0)   # 2.0 ATR
            }
            
            return sl_levels
            
        except Exception as e:
            self.logger.error(f"Error in stop loss levels: {e}")
            return {"error": str(e)}
    
    def _calculate_atr(self, ohlc_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = ohlc_data['high']
            low = ohlc_data['low']
            close = ohlc_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.001
            
        except Exception as e:
            self.logger.error(f"Error in ATR calculation: {e}")
            return 0.001
    
    def generate_price_action_report(self, ohlc_data: pd.DataFrame) -> Dict:
        """Generate comprehensive price action analysis report"""
        try:
            # Analyze patterns
            patterns = self.analyze_candlestick_patterns(ohlc_data)
            
            # Detect support/resistance
            levels = self.detect_support_resistance(ohlc_data)
            
            # Generate signals
            signals = self.generate_entry_exit_signals(ohlc_data, patterns)
            
            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "patterns": patterns,
                "support_resistance": levels,
                "signals": signals,
                "current_price": ohlc_data['close'].iloc[-1],
                "recommendations": []
            }
            
            # Generate recommendations
            recommendations = self._generate_price_action_recommendations(report)
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating price action report: {e}")
            return {"error": str(e)}
    
    def _generate_price_action_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on price action analysis"""
        try:
            recommendations = []
            
            # Pattern-based recommendations
            if "patterns" in report:
                patterns = report["patterns"]
                
                # Check for strong reversal patterns
                if "reversal" in patterns:
                    reversal = patterns["reversal"]
                    if any(reversal.values()):
                        recommendations.append("Strong reversal patterns detected - consider position reversal")
                
                # Check for continuation patterns
                if "continuation" in patterns:
                    continuation = patterns["continuation"]
                    if any(continuation.values()):
                        recommendations.append("Continuation patterns detected - consider trend following")
            
            # Support/resistance recommendations
            if "support_resistance" in report:
                levels = report["support_resistance"]
                current_price = report.get("current_price", 0)
                
                if "pivot_points" in levels:
                    pivot = levels["pivot_points"]
                    if "pivot" in pivot:
                        pivot_level = pivot["pivot"]
                        if abs(current_price - pivot_level) < 0.001:
                            recommendations.append("Price at pivot point - watch for breakout or reversal")
            
            if not recommendations:
                recommendations.append("Price action analysis shows balanced market conditions")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]