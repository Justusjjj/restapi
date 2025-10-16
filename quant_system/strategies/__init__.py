"""
Strategies Module
================

Rule-based trading strategies for institutional quantitative trading.

Key Components:
- BaseStrategy: Abstract base class for all strategies
- MeanReversionStrategy: Intraday mean reversion strategy
- MomentumStrategy: Trend-following momentum strategy
- PairTradingStrategy: Statistical arbitrage pair trading
- CarryStrategy: Interest rate differential strategy
- ExecutionStrategy: VWAP/TWAP execution strategies
"""

from .base_strategy import BaseStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .momentum_strategy import MomentumStrategy
from .pair_trading_strategy import PairTradingStrategy
from .carry_strategy import CarryStrategy
from .execution_strategy import ExecutionStrategy

__all__ = [
    'BaseStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'PairTradingStrategy',
    'CarryStrategy',
    'ExecutionStrategy'
]