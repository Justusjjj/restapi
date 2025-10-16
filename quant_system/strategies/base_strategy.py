"""
Base Strategy
============

Abstract base class for all trading strategies.
Defines the interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Represents a trading signal."""
    timestamp: datetime
    instrument: str
    signal_type: str  # 'long', 'short', 'close_long', 'close_short'
    strength: float  # Signal strength (-1 to 1)
    price: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Represents a trading position."""
    instrument: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - generate_signals: Generate trading signals
    - calculate_position_size: Calculate position size
    - validate_signal: Validate signal before execution
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = {}
        
        # Initialize strategy-specific parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize strategy-specific parameters. Override in subclasses."""
        pass
        
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data: Dictionary of market data by instrument
            
        Returns:
            List of Signal objects
        """
        pass
        
    @abstractmethod
    def calculate_position_size(self, signal: Signal, 
                              current_position: Optional[Position],
                              account_value: float) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            current_position: Current position (if any)
            account_value: Current account value
            
        Returns:
            Position size
        """
        pass
        
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate signal before execution.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        # Basic validation
        if signal.strength < -1 or signal.strength > 1:
            logger.warning(f"Invalid signal strength: {signal.strength}")
            return False
            
        if signal.price <= 0:
            logger.warning(f"Invalid signal price: {signal.price}")
            return False
            
        return True
        
    def update_position(self, instrument: str, side: str, size: float,
                       price: float, timestamp: datetime) -> None:
        """
        Update position for an instrument.
        
        Args:
            instrument: Instrument symbol
            side: Position side ('long' or 'short')
            size: Position size
            price: Execution price
            timestamp: Execution timestamp
        """
        if instrument in self.positions:
            # Update existing position
            pos = self.positions[instrument]
            if pos.side == side:
                # Same side - add to position
                total_size = pos.size + size
                avg_price = ((pos.entry_price * pos.size) + (price * size)) / total_size
                pos.size = total_size
                pos.entry_price = avg_price
            else:
                # Opposite side - reduce or reverse position
                if size >= pos.size:
                    # Complete reversal
                    pos.side = side
                    pos.size = size - pos.size
                    pos.entry_price = price
                    pos.entry_time = timestamp
                else:
                    # Partial close
                    pos.size -= size
        else:
            # New position
            self.positions[instrument] = Position(
                instrument=instrument,
                side=side,
                size=size,
                entry_price=price,
                entry_time=timestamp,
                current_price=price,
                unrealized_pnl=0.0
            )
            
    def close_position(self, instrument: str) -> Optional[Position]:
        """
        Close position for an instrument.
        
        Args:
            instrument: Instrument symbol
            
        Returns:
            Closed position or None
        """
        if instrument in self.positions:
            closed_position = self.positions.pop(instrument)
            return closed_position
        return None
        
    def update_pnl(self, instrument: str, current_price: float) -> None:
        """
        Update unrealized PnL for a position.
        
        Args:
            instrument: Instrument symbol
            current_price: Current market price
        """
        if instrument in self.positions:
            pos = self.positions[instrument]
            pos.current_price = current_price
            
            if pos.side == 'long':
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            else:  # short
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
                
    def get_total_exposure(self) -> float:
        """
        Get total exposure across all positions.
        
        Returns:
            Total exposure value
        """
        total_exposure = 0.0
        for pos in self.positions.values():
            total_exposure += abs(pos.size * pos.current_price)
        return total_exposure
        
    def get_position_count(self) -> int:
        """
        Get number of open positions.
        
        Returns:
            Number of open positions
        """
        return len(self.positions)
        
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.positions:
            return {
                'total_positions': 0,
                'total_exposure': 0.0,
                'unrealized_pnl': 0.0
            }
            
        total_exposure = self.get_total_exposure()
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_positions': len(self.positions),
            'total_exposure': total_exposure,
            'unrealized_pnl': total_unrealized_pnl,
            'positions': {
                inst: {
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for inst, pos in self.positions.items()
            }
        }
        
    def reset(self) -> None:
        """Reset strategy state."""
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = {}
        
    def log_signal(self, signal: Signal) -> None:
        """
        Log signal for analysis.
        
        Args:
            signal: Signal to log
        """
        self.signals_history.append(signal)
        logger.debug(f"Strategy {self.name}: {signal.signal_type} signal for {signal.instrument} "
                    f"at {signal.price} with strength {signal.strength}")
        
    def get_signals_summary(self) -> Dict[str, Any]:
        """
        Get summary of generated signals.
        
        Returns:
            Dictionary with signal statistics
        """
        if not self.signals_history:
            return {'total_signals': 0}
            
        signal_types = [s.signal_type for s in self.signals_history]
        signal_counts = pd.Series(signal_types).value_counts().to_dict()
        
        return {
            'total_signals': len(self.signals_history),
            'signal_counts': signal_counts,
            'latest_signal': self.signals_history[-1] if self.signals_history else None
        }