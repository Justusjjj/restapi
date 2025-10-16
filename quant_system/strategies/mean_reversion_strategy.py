"""
Mean Reversion Strategy
======================

Intraday mean reversion strategy using VWAP and z-score signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, Position
from ..features import PriceIndicators, VolatilityIndicators

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy for intraday trading.
    
    Strategy Logic:
    1. Calculate VWAP and rolling z-score of price vs VWAP
    2. Enter long when z-score < -threshold (oversold)
    3. Enter short when z-score > +threshold (overbought)
    4. Exit when z-score reverts to mean or stop loss hit
    5. Position sizing based on volatility (ATR)
    """
    
    def _initialize_parameters(self):
        """Initialize strategy parameters."""
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)
        self.zscore_exit_threshold = self.config.get('zscore_exit_threshold', 0.5)
        self.vwap_window = self.config.get('vwap_window', 20)
        self.zscore_window = self.config.get('zscore_window', 20)
        self.atr_window = self.config.get('atr_window', 14)
        self.stop_loss_atr_multiple = self.config.get('stop_loss_atr_multiple', 2.0)
        self.target_volatility = self.config.get('target_volatility', 0.02)  # 2% daily vol
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% of account
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.0)  # Min volume vs average
        
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate mean reversion signals.
        
        Args:
            market_data: Dictionary of market data by instrument
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        for instrument, data in market_data.items():
            if len(data) < max(self.vwap_window, self.zscore_window, self.atr_window):
                continue
                
            try:
                # Calculate indicators
                vwap = PriceIndicators.vwap(
                    data['high'], data['low'], data['close'], data['volume']
                )
                
                zscore = PriceIndicators.price_zscore(data['close'], self.zscore_window)
                vwap_zscore = (data['close'] - vwap) / vwap.rolling(self.zscore_window).std()
                
                atr = VolatilityIndicators.atr(
                    data['high'], data['low'], data['close'], self.atr_window
                )
                
                # Volume filter
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
                
                # Get latest values
                latest_idx = data.index[-1]
                latest_close = data['close'].iloc[-1]
                latest_vwap = vwap.iloc[-1]
                latest_zscore = zscore.iloc[-1]
                latest_vwap_zscore = vwap_zscore.iloc[-1]
                latest_atr = atr.iloc[-1]
                latest_volume_ratio = volume_ratio.iloc[-1]
                
                # Skip if insufficient data
                if pd.isna(latest_zscore) or pd.isna(latest_vwap_zscore) or pd.isna(latest_atr):
                    continue
                    
                # Volume filter
                if latest_volume_ratio < self.min_volume_ratio:
                    continue
                    
                # Generate signals based on current position
                current_position = self.positions.get(instrument)
                
                if current_position is None:
                    # No position - look for entry signals
                    
                    # Long signal (oversold)
                    if latest_zscore < -self.zscore_threshold and latest_vwap_zscore < -self.zscore_threshold:
                        signal = Signal(
                            timestamp=latest_idx,
                            instrument=instrument,
                            signal_type='long',
                            strength=min(1.0, abs(latest_zscore) / self.zscore_threshold),
                            price=latest_close,
                            metadata={
                                'zscore': latest_zscore,
                                'vwap_zscore': latest_vwap_zscore,
                                'vwap': latest_vwap,
                                'atr': latest_atr,
                                'volume_ratio': latest_volume_ratio
                            }
                        )
                        signals.append(signal)
                        
                    # Short signal (overbought)
                    elif latest_zscore > self.zscore_threshold and latest_vwap_zscore > self.zscore_threshold:
                        signal = Signal(
                            timestamp=latest_idx,
                            instrument=instrument,
                            signal_type='short',
                            strength=min(1.0, abs(latest_zscore) / self.zscore_threshold),
                            price=latest_close,
                            metadata={
                                'zscore': latest_zscore,
                                'vwap_zscore': latest_vwap_zscore,
                                'vwap': latest_vwap,
                                'atr': latest_atr,
                                'volume_ratio': latest_volume_ratio
                            }
                        )
                        signals.append(signal)
                        
                else:
                    # Have position - look for exit signals
                    
                    if current_position.side == 'long':
                        # Exit long if z-score reverts to mean or becomes positive
                        if (latest_zscore > -self.zscore_exit_threshold or 
                            latest_vwap_zscore > -self.zscore_exit_threshold):
                            signal = Signal(
                                timestamp=latest_idx,
                                instrument=instrument,
                                signal_type='close_long',
                                strength=1.0,
                                price=latest_close,
                                metadata={
                                    'zscore': latest_zscore,
                                    'vwap_zscore': latest_vwap_zscore,
                                    'exit_reason': 'mean_reversion'
                                }
                            )
                            signals.append(signal)
                            
                    elif current_position.side == 'short':
                        # Exit short if z-score reverts to mean or becomes negative
                        if (latest_zscore < self.zscore_exit_threshold or 
                            latest_vwap_zscore < self.zscore_exit_threshold):
                            signal = Signal(
                                timestamp=latest_idx,
                                instrument=instrument,
                                signal_type='close_short',
                                strength=1.0,
                                price=latest_close,
                                metadata={
                                    'zscore': latest_zscore,
                                    'vwap_zscore': latest_vwap_zscore,
                                    'exit_reason': 'mean_reversion'
                                }
                            )
                            signals.append(signal)
                            
            except Exception as e:
                logger.error(f"Error generating signals for {instrument}: {e}")
                continue
                
        return signals
        
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
        # Position size = (target_volatility * account_value) / (atr * price)
        target_vol_value = self.target_volatility * account_value
        position_size = target_vol_value / (atr * signal.price)
        
        # Apply maximum position size limit
        max_size = self.max_position_size * account_value / signal.price
        position_size = min(position_size, max_size)
        
        # Ensure minimum position size
        min_size = 0.001 * account_value / signal.price
        position_size = max(position_size, min_size)
        
        return position_size
        
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate mean reversion signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        # Call parent validation
        if not super().validate_signal(signal):
            return False
            
        # Check for required metadata
        required_metadata = ['zscore', 'vwap_zscore', 'atr']
        for key in required_metadata:
            if key not in signal.metadata or pd.isna(signal.metadata[key]):
                logger.warning(f"Missing required metadata: {key}")
                return False
                
        # Check z-score thresholds
        zscore = signal.metadata['zscore']
        vwap_zscore = signal.metadata['vwap_zscore']
        
        if signal.signal_type == 'long':
            if zscore >= -self.zscore_threshold or vwap_zscore >= -self.zscore_threshold:
                logger.warning(f"Long signal z-score too high: {zscore}, {vwap_zscore}")
                return False
        elif signal.signal_type == 'short':
            if zscore <= self.zscore_threshold or vwap_zscore <= self.zscore_threshold:
                logger.warning(f"Short signal z-score too low: {zscore}, {vwap_zscore}")
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
            'zscore_threshold': self.zscore_threshold,
            'target_volatility': self.target_volatility,
            'max_position_size': self.max_position_size,
            'atr_window': self.atr_window,
            'vwap_window': self.vwap_window
        }
        
        base_metrics.update(strategy_metrics)
        return base_metrics