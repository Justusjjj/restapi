"""
Backtest Engine Module
=====================

Event-driven backtesting engine with realistic transaction costs and risk controls.

Key Components:
- BacktestEngine: Main backtesting engine
- Portfolio: Portfolio management and accounting
- OrderManager: Order execution and fill simulation
- TransactionCostModel: Realistic cost modeling
- PerformanceAnalyzer: Performance metrics and analysis
"""

from .backtest_engine import BacktestEngine
from .portfolio import Portfolio
from .order_manager import OrderManager
from .transaction_cost_model import TransactionCostModel
from .performance_analyzer import PerformanceAnalyzer

__all__ = [
    'BacktestEngine',
    'Portfolio',
    'OrderManager',
    'TransactionCostModel',
    'PerformanceAnalyzer'
]