"""
Pair Trading Strategy
====================

Statistical arbitrage strategy using cointegration and spread trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, Position
from ..features import StatisticalIndicators, PriceIndicators

logger = logging.getLogger(__name__)


class PairTradingStrategy(BaseStrategy):
    """
    Pair trading strategy using statistical arbitrage.
    
    Strategy Logic:
    1. Find cointegrated pairs using Engle-Granger test
    2. Calculate rolling hedge ratio (beta)
    3. Construct spread series
    4. Enter when spread z-score exceeds threshold
    5. Exit when spread reverts to mean
    6. Rebalance hedge ratio periodically
    """
    
    def _initialize_parameters(self):
        """Initialize strategy parameters."""
        self.zscore_entry_threshold = self.config.get('zscore_entry_threshold', 2.0)
        self.zscore_exit_threshold = self.config.get('zscore_exit_threshold', 0.5)
        self.zscore_window = self.config.get('zscore_window', 20)
        self.hedge_ratio_window = self.config.get('hedge_ratio_window', 60)
        self.rebalance_frequency = self.config.get('rebalance_frequency', 5)  # Days
        self.min_cointegration_pvalue = self.config.get('min_cointegration_pvalue', 0.05)
        self.target_volatility = self.config.get('target_volatility', 0.015)
        self.max_position_size = self.config.get('max_position_size', 0.05)
        self.min_half_life = self.config.get('min_half_life', 1)  # Days
        self.max_half_life = self.config.get('max_half_life', 30)  # Days
        
        # Pair tracking
        self.pairs = {}
        self.last_rebalance = {}
        self.spread_data = {}
        
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate pair trading signals.
        
        Args:
            market_data: Dictionary of market data by instrument
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        # Get all possible pairs
        instruments = list(market_data.keys())
        
        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                pair_name = f"{instruments[i]}_{instruments[j]}"
                
                try:
                    # Get pair data
                    data1 = market_data[instruments[i]]
                    data2 = market_data[instruments[j]]
                    
                    if len(data1) < self.hedge_ratio_window or len(data2) < self.hedge_ratio_window:
                        continue
                        
                    # Check if we need to rebalance hedge ratio
                    if self._should_rebalance(pair_name):
                        self._update_hedge_ratio(pair_name, data1, data2)
                        
                    # Generate signals for this pair
                    pair_signals = self._generate_pair_signals(
                        pair_name, instruments[i], instruments[j], data1, data2
                    )
                    signals.extend(pair_signals)
                    
                except Exception as e:
                    logger.error(f"Error processing pair {pair_name}: {e}")
                    continue
                    
        return signals
        
    def _should_rebalance(self, pair_name: str) -> bool:
        """Check if hedge ratio should be rebalanced."""
        if pair_name not in self.last_rebalance:
            return True
            
        days_since_rebalance = (datetime.now() - self.last_rebalance[pair_name]).days
        return days_since_rebalance >= self.rebalance_frequency
        
    def _update_hedge_ratio(self, pair_name: str, data1: pd.DataFrame, data2: pd.DataFrame):
        """Update hedge ratio for a pair."""
        try:
            # Align data
            common_index = data1.index.intersection(data2.index)
            if len(common_index) < self.hedge_ratio_window:
                return
                
            series1 = data1.loc[common_index, 'close']
            series2 = data2.loc[common_index, 'close']
            
            # Test cointegration
            test_stat, p_value, critical_values = StatisticalIndicators.cointegration_test(
                series1, series2
            )
            
            if p_value > self.min_cointegration_pvalue:
                logger.debug(f"Pair {pair_name} not cointegrated (p-value: {p_value:.4f})")
                return
                
            # Calculate hedge ratio
            hedge_ratio = StatisticalIndicators.hedge_ratio(series1, series2, self.hedge_ratio_window)
            latest_hedge_ratio = hedge_ratio.iloc[-1]
            
            if pd.isna(latest_hedge_ratio):
                return
                
            # Calculate spread
            spread = StatisticalIndicators.spread_series(series1, series2, hedge_ratio)
            
            # Calculate half-life
            half_life = StatisticalIndicators.half_life(spread)
            
            if half_life < self.min_half_life or half_life > self.max_half_life:
                logger.debug(f"Pair {pair_name} half-life out of range: {half_life:.2f}")
                return
                
            # Store pair information
            self.pairs[pair_name] = {
                'instrument1': series1.name,
                'instrument2': series2.name,
                'hedge_ratio': latest_hedge_ratio,
                'cointegration_pvalue': p_value,
                'half_life': half_life,
                'spread_mean': spread.mean(),
                'spread_std': spread.std()
            }
            
            self.last_rebalance[pair_name] = datetime.now()
            
            logger.info(f"Updated pair {pair_name}: hedge_ratio={latest_hedge_ratio:.4f}, "
                       f"p_value={p_value:.4f}, half_life={half_life:.2f}")
                       
        except Exception as e:
            logger.error(f"Error updating hedge ratio for {pair_name}: {e}")
            
    def _generate_pair_signals(self, pair_name: str, instrument1: str, instrument2: str,
                              data1: pd.DataFrame, data2: pd.DataFrame) -> List[Signal]:
        """Generate signals for a specific pair."""
        signals = []
        
        if pair_name not in self.pairs:
            return signals
            
        try:
            # Get pair information
            pair_info = self.pairs[pair_name]
            hedge_ratio = pair_info['hedge_ratio']
            
            # Align data
            common_index = data1.index.intersection(data2.index)
            if len(common_index) < self.zscore_window:
                return signals
                
            series1 = data1.loc[common_index, 'close']
            series2 = data2.loc[common_index, 'close']
            
            # Calculate spread
            spread = series1 - hedge_ratio * series2
            
            # Calculate spread z-score
            spread_zscore = StatisticalIndicators.spread_zscore(spread, self.zscore_window)
            
            # Get latest values
            latest_idx = common_index[-1]
            latest_spread = spread.iloc[-1]
            latest_zscore = spread_zscore.iloc[-1]
            latest_price1 = series1.iloc[-1]
            latest_price2 = series2.iloc[-1]
            
            if pd.isna(latest_zscore):
                return signals
                
            # Check current positions
            pos1 = self.positions.get(instrument1)
            pos2 = self.positions.get(instrument2)
            
            if pos1 is None and pos2 is None:
                # No positions - look for entry signals
                
                if latest_zscore > self.zscore_entry_threshold:
                    # Spread is high - short spread (short instrument1, long instrument2)
                    signal1 = Signal(
                        timestamp=latest_idx,
                        instrument=instrument1,
                        signal_type='short',
                        strength=min(1.0, latest_zscore / self.zscore_entry_threshold),
                        price=latest_price1,
                        metadata={
                            'pair_name': pair_name,
                            'spread': latest_spread,
                            'zscore': latest_zscore,
                            'hedge_ratio': hedge_ratio,
                            'side': 'spread_short'
                        }
                    )
                    signals.append(signal1)
                    
                    signal2 = Signal(
                        timestamp=latest_idx,
                        instrument=instrument2,
                        signal_type='long',
                        strength=min(1.0, latest_zscore / self.zscore_entry_threshold),
                        price=latest_price2,
                        metadata={
                            'pair_name': pair_name,
                            'spread': latest_spread,
                            'zscore': latest_zscore,
                            'hedge_ratio': hedge_ratio,
                            'side': 'spread_short'
                        }
                    )
                    signals.append(signal2)
                    
                elif latest_zscore < -self.zscore_entry_threshold:
                    # Spread is low - long spread (long instrument1, short instrument2)
                    signal1 = Signal(
                        timestamp=latest_idx,
                        instrument=instrument1,
                        signal_type='long',
                        strength=min(1.0, abs(latest_zscore) / self.zscore_entry_threshold),
                        price=latest_price1,
                        metadata={
                            'pair_name': pair_name,
                            'spread': latest_spread,
                            'zscore': latest_zscore,
                            'hedge_ratio': hedge_ratio,
                            'side': 'spread_long'
                        }
                    )
                    signals.append(signal1)
                    
                    signal2 = Signal(
                        timestamp=latest_idx,
                        instrument=instrument2,
                        signal_type='short',
                        strength=min(1.0, abs(latest_zscore) / self.zscore_entry_threshold),
                        price=latest_price2,
                        metadata={
                            'pair_name': pair_name,
                            'spread': latest_spread,
                            'zscore': latest_zscore,
                            'hedge_ratio': hedge_ratio,
                            'side': 'spread_long'
                        }
                    )
                    signals.append(signal2)
                    
            else:
                # Have positions - look for exit signals
                
                if (pos1 and pos1.metadata.get('pair_name') == pair_name and
                    pos2 and pos2.metadata.get('pair_name') == pair_name):
                    
                    # Check if spread has reverted
                    if abs(latest_zscore) < self.zscore_exit_threshold:
                        # Close both positions
                        signal1 = Signal(
                            timestamp=latest_idx,
                            instrument=instrument1,
                            signal_type='close_long' if pos1.side == 'long' else 'close_short',
                            strength=1.0,
                            price=latest_price1,
                            metadata={
                                'pair_name': pair_name,
                                'exit_reason': 'spread_reversion',
                                'zscore': latest_zscore
                            }
                        )
                        signals.append(signal1)
                        
                        signal2 = Signal(
                            timestamp=latest_idx,
                            instrument=instrument2,
                            signal_type='close_long' if pos2.side == 'long' else 'close_short',
                            strength=1.0,
                            price=latest_price2,
                            metadata={
                                'pair_name': pair_name,
                                'exit_reason': 'spread_reversion',
                                'zscore': latest_zscore
                            }
                        )
                        signals.append(signal2)
                        
        except Exception as e:
            logger.error(f"Error generating signals for pair {pair_name}: {e}")
            
        return signals
        
    def calculate_position_size(self, signal: Signal, 
                              current_position: Optional[Position],
                              account_value: float) -> float:
        """
        Calculate position size for pair trading.
        
        Args:
            signal: Trading signal
            current_position: Current position (if any)
            account_value: Current account value
            
        Returns:
            Position size
        """
        if signal.signal_type in ['close_long', 'close_short']:
            return current_position.size if current_position else 0.0
            
        # Get pair information
        pair_name = signal.metadata.get('pair_name')
        if not pair_name or pair_name not in self.pairs:
            return 0.0
            
        pair_info = self.pairs[pair_name]
        
        # Calculate position size based on target volatility
        # Use spread volatility for sizing
        spread_std = pair_info.get('spread_std', 0.0)
        if spread_std <= 0:
            return 0.0
            
        # Position size = (target_volatility * account_value) / (spread_std * price)
        target_vol_value = self.target_volatility * account_value
        position_size = target_vol_value / (spread_std * signal.price)
        
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
        Validate pair trading signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        # Call parent validation
        if not super().validate_signal(signal):
            return False
            
        # Check for required metadata
        required_metadata = ['pair_name', 'spread', 'zscore', 'hedge_ratio']
        for key in required_metadata:
            if key not in signal.metadata or pd.isna(signal.metadata[key]):
                logger.warning(f"Missing required metadata: {key}")
                return False
                
        # Check pair exists
        pair_name = signal.metadata['pair_name']
        if pair_name not in self.pairs:
            logger.warning(f"Pair {pair_name} not found")
            return False
            
        # Check z-score threshold
        zscore = signal.metadata['zscore']
        if abs(zscore) < self.zscore_entry_threshold:
            logger.warning(f"Z-score {zscore} below entry threshold")
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
            'zscore_entry_threshold': self.zscore_entry_threshold,
            'zscore_exit_threshold': self.zscore_exit_threshold,
            'target_volatility': self.target_volatility,
            'max_position_size': self.max_position_size,
            'hedge_ratio_window': self.hedge_ratio_window,
            'rebalance_frequency': self.rebalance_frequency,
            'active_pairs': len(self.pairs),
            'pairs': list(self.pairs.keys())
        }
        
        base_metrics.update(strategy_metrics)
        return base_metrics