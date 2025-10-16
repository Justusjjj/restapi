"""
Momentum Strategy
================

Trend-following momentum strategy using EMA crossovers and breakout signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, Position
from ..features import PriceIndicators, MomentumIndicators, VolatilityIndicators

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy for trend-following.
    
    Strategy Logic:
    1. Use EMA crossovers for trend identification
    2. Use RSI for momentum confirmation
    3. Use ATR for stop loss and position sizing
    4. Enter on trend confirmation with volume filter
    5. Exit on trend reversal or stop loss
    """
    
    def _initialize_parameters(self):
        """Initialize strategy parameters."""
        self.fast_ema = self.config.get('fast_ema', 12)
        self.slow_ema = self.config.get('slow_ema', 26)
        self.signal_ema = self.config.get('signal_ema', 9)
        self.rsi_window = self.config.get('rsi_window', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.atr_window = self.config.get('atr_window', 14)
        self.stop_loss_atr_multiple = self.config.get('stop_loss_atr_multiple', 2.0)
        self.take_profit_atr_multiple = self.config.get('take_profit_atr_multiple', 3.0)
        self.target_volatility = self.config.get('target_volatility', 0.02)
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.2)
        self.trend_strength_threshold = self.config.get('trend_strength_threshold', 0.3)
        
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate momentum signals.
        
        Args:
            market_data: Dictionary of market data by instrument
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        for instrument, data in market_data.items():
            if len(data) < max(self.fast_ema, self.slow_ema, self.rsi_window, self.atr_window):
                continue
                
            try:
                # Calculate indicators
                ema_fast = PriceIndicators.ema(data['close'], self.fast_ema)
                ema_slow = PriceIndicators.ema(data['close'], self.slow_ema)
                ema_signal = PriceIndicators.ema(data['close'], self.signal_ema)
                
                rsi = MomentumIndicators.rsi(data['close'], self.rsi_window)
                
                atr = VolatilityIndicators.atr(
                    data['high'], data['low'], data['close'], self.atr_window
                )
                
                # Trend strength
                trend_strength = PriceIndicators.trend_strength(data['close'], 20)
                
                # Volume filter
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
                
                # MACD for additional confirmation
                macd_line, signal_line, histogram = MomentumIndicators.macd(
                    data['close'], self.fast_ema, self.slow_ema, self.signal_ema
                )
                
                # Get latest values
                latest_idx = data.index[-1]
                latest_close = data['close'].iloc[-1]
                latest_ema_fast = ema_fast.iloc[-1]
                latest_ema_slow = ema_slow.iloc[-1]
                latest_ema_signal = ema_signal.iloc[-1]
                latest_rsi = rsi.iloc[-1]
                latest_atr = atr.iloc[-1]
                latest_trend_strength = trend_strength.iloc[-1]
                latest_volume_ratio = volume_ratio.iloc[-1]
                latest_macd = macd_line.iloc[-1]
                latest_signal_line = signal_line.iloc[-1]
                latest_histogram = histogram.iloc[-1]
                
                # Skip if insufficient data
                if (pd.isna(latest_ema_fast) or pd.isna(latest_ema_slow) or 
                    pd.isna(latest_rsi) or pd.isna(latest_atr)):
                    continue
                    
                # Volume filter
                if latest_volume_ratio < self.min_volume_ratio:
                    continue
                    
                # Generate signals based on current position
                current_position = self.positions.get(instrument)
                
                if current_position is None:
                    # No position - look for entry signals
                    
                    # Long signal conditions
                    long_conditions = [
                        latest_ema_fast > latest_ema_slow,  # Fast EMA above slow EMA
                        latest_ema_signal > latest_ema_slow,  # Signal EMA above slow EMA
                        latest_rsi > 50,  # RSI above 50 (momentum)
                        latest_rsi < self.rsi_overbought,  # Not overbought
                        latest_trend_strength > self.trend_strength_threshold,  # Strong trend
                        latest_macd > latest_signal_line,  # MACD bullish
                        latest_histogram > 0  # MACD histogram positive
                    ]
                    
                    if all(long_conditions):
                        signal_strength = self._calculate_signal_strength(
                            latest_rsi, latest_trend_strength, latest_histogram
                        )
                        
                        signal = Signal(
                            timestamp=latest_idx,
                            instrument=instrument,
                            signal_type='long',
                            strength=signal_strength,
                            price=latest_close,
                            metadata={
                                'ema_fast': latest_ema_fast,
                                'ema_slow': latest_ema_slow,
                                'rsi': latest_rsi,
                                'atr': latest_atr,
                                'trend_strength': latest_trend_strength,
                                'volume_ratio': latest_volume_ratio,
                                'macd': latest_macd,
                                'signal_line': latest_signal_line,
                                'histogram': latest_histogram
                            }
                        )
                        signals.append(signal)
                        
                    # Short signal conditions
                    short_conditions = [
                        latest_ema_fast < latest_ema_slow,  # Fast EMA below slow EMA
                        latest_ema_signal < latest_ema_slow,  # Signal EMA below slow EMA
                        latest_rsi < 50,  # RSI below 50 (momentum)
                        latest_rsi > self.rsi_oversold,  # Not oversold
                        latest_trend_strength < -self.trend_strength_threshold,  # Strong downtrend
                        latest_macd < latest_signal_line,  # MACD bearish
                        latest_histogram < 0  # MACD histogram negative
                    ]
                    
                    if all(short_conditions):
                        signal_strength = self._calculate_signal_strength(
                            latest_rsi, latest_trend_strength, latest_histogram, is_short=True
                        )
                        
                        signal = Signal(
                            timestamp=latest_idx,
                            instrument=instrument,
                            signal_type='short',
                            strength=signal_strength,
                            price=latest_close,
                            metadata={
                                'ema_fast': latest_ema_fast,
                                'ema_slow': latest_ema_slow,
                                'rsi': latest_rsi,
                                'atr': latest_atr,
                                'trend_strength': latest_trend_strength,
                                'volume_ratio': latest_volume_ratio,
                                'macd': latest_macd,
                                'signal_line': latest_signal_line,
                                'histogram': latest_histogram
                            }
                        )
                        signals.append(signal)
                        
                else:
                    # Have position - look for exit signals
                    
                    if current_position.side == 'long':
                        # Exit long conditions
                        exit_conditions = [
                            latest_ema_fast < latest_ema_slow,  # EMA crossover
                            latest_rsi < 50,  # RSI below 50
                            latest_macd < latest_signal_line,  # MACD bearish
                            latest_trend_strength < 0  # Trend reversal
                        ]
                        
                        if any(exit_conditions):
                            signal = Signal(
                                timestamp=latest_idx,
                                instrument=instrument,
                                signal_type='close_long',
                                strength=1.0,
                                price=latest_close,
                                metadata={
                                    'exit_reason': 'trend_reversal',
                                    'ema_fast': latest_ema_fast,
                                    'ema_slow': latest_ema_slow,
                                    'rsi': latest_rsi,
                                    'macd': latest_macd,
                                    'signal_line': latest_signal_line
                                }
                            )
                            signals.append(signal)
                            
                    elif current_position.side == 'short':
                        # Exit short conditions
                        exit_conditions = [
                            latest_ema_fast > latest_ema_slow,  # EMA crossover
                            latest_rsi > 50,  # RSI above 50
                            latest_macd > latest_signal_line,  # MACD bullish
                            latest_trend_strength > 0  # Trend reversal
                        ]
                        
                        if any(exit_conditions):
                            signal = Signal(
                                timestamp=latest_idx,
                                instrument=instrument,
                                signal_type='close_short',
                                strength=1.0,
                                price=latest_close,
                                metadata={
                                    'exit_reason': 'trend_reversal',
                                    'ema_fast': latest_ema_fast,
                                    'ema_slow': latest_ema_slow,
                                    'rsi': latest_rsi,
                                    'macd': latest_macd,
                                    'signal_line': latest_signal_line
                                }
                            )
                            signals.append(signal)
                            
            except Exception as e:
                logger.error(f"Error generating signals for {instrument}: {e}")
                continue
                
        return signals
        
    def _calculate_signal_strength(self, rsi: float, trend_strength: float, 
                                 histogram: float, is_short: bool = False) -> float:
        """
        Calculate signal strength based on multiple factors.
        
        Args:
            rsi: RSI value
            trend_strength: Trend strength value
            histogram: MACD histogram value
            is_short: Whether this is a short signal
            
        Returns:
            Signal strength (0-1)
        """
        # RSI strength (0-1)
        if is_short:
            rsi_strength = (100 - rsi) / 100
        else:
            rsi_strength = rsi / 100
            
        # Trend strength (0-1)
        trend_strength_norm = abs(trend_strength)
        
        # MACD histogram strength (0-1)
        histogram_strength = abs(histogram) / (abs(histogram) + 1)  # Normalize to 0-1
        
        # Combined strength
        combined_strength = (rsi_strength + trend_strength_norm + histogram_strength) / 3
        
        return min(1.0, max(0.1, combined_strength))
        
    def calculate_position_size(self, signal: Signal, 
                              current_position: Optional[Position],
                              account_value: float) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            signal: Trading signal
            current_position: Current position (if any)
            account_value: Current account value
            
        Returns:
            Position size
        """
        if signal.signal_type in ['close_long', 'close_short']:
            return current_position.size if current_position else 0.0
            
        # Get ATR from signal metadata
        atr = signal.metadata.get('atr', 0.0)
        if atr <= 0:
            return 0.0
            
        # Calculate position size based on target volatility
        target_vol_value = self.target_volatility * account_value
        position_size = target_vol_value / (atr * signal.price)
        
        # Apply maximum position size limit
        max_size = self.max_position_size * account_value / signal.price
        position_size = min(position_size, max_size)
        
        # Scale by signal strength
        position_size *= signal.strength
        
        # Ensure minimum position size
        min_size = 0.001 * account_value / signal.price
        position_size = max(position_size, min_size)
        
        return position_size
        
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate momentum signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        # Call parent validation
        if not super().validate_signal(signal):
            return False
            
        # Check for required metadata
        required_metadata = ['ema_fast', 'ema_slow', 'rsi', 'atr', 'trend_strength']
        for key in required_metadata:
            if key not in signal.metadata or pd.isna(signal.metadata[key]):
                logger.warning(f"Missing required metadata: {key}")
                return False
                
        # Check RSI bounds
        rsi = signal.metadata['rsi']
        if rsi < 0 or rsi > 100:
            logger.warning(f"Invalid RSI: {rsi}")
            return False
            
        # Check trend strength
        trend_strength = signal.metadata['trend_strength']
        if abs(trend_strength) > 1:
            logger.warning(f"Invalid trend strength: {trend_strength}")
            return False
            
        # Check ATR
        atr = signal.metadata['atr']
        if atr <= 0:
            logger.warning(f"Invalid ATR: {atr}")
            return False
            
        return True
        
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """
        Get strategy-specific metrics.
        
        Returns:
            Dictionary of strategy metrics
        """
        base_metrics = super().get_strategy_metrics()
        
        # Add strategy-specific metrics
        strategy_metrics = {
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'rsi_window': self.rsi_window,
            'target_volatility': self.target_volatility,
            'max_position_size': self.max_position_size,
            'trend_strength_threshold': self.trend_strength_threshold
        }
        
        base_metrics.update(strategy_metrics)
        return base_metrics