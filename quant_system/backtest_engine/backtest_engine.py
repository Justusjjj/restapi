"""
Backtest Engine
===============

Main event-driven backtesting engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
import time

from .portfolio import Portfolio, Trade
from .transaction_cost_model import TransactionCostModel
from .performance_analyzer import PerformanceAnalyzer
from ..data_engine import ReplayEngine
from ..strategies import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Features:
    - Event-driven simulation
    - Realistic transaction costs
    - Portfolio management
    - Performance analysis
    - Risk controls
    """
    
    def __init__(self, initial_capital: float = 100000.0,
                 transaction_cost_config: Optional[Dict] = None):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost_config: Transaction cost configuration
        """
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        
        # Transaction cost model
        cost_config = transaction_cost_config or self._default_cost_config()
        self.cost_model = TransactionCostModel(cost_config)
        
        # Performance analyzer
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Backtest state
        self.current_time = None
        self.strategies = []
        self.market_data = {}
        self.is_running = False
        
        # Statistics
        self.start_time = None
        self.end_time = None
        self.total_events = 0
        self.executed_trades = 0
        
    def _default_cost_config(self) -> Dict[str, Any]:
        """Default transaction cost configuration."""
        return {
            'commission_per_trade': 1.0,  # $1 per trade
            'commission_per_value': 0.001,  # 0.1% of trade value
            'spread_bps': 1.0,  # 1 basis point spread
            'spread_model': 'fixed',
            'slippage_bps': 0.5,  # 0.5 basis points slippage
            'slippage_model': 'sqrt',
            'max_slippage_bps': 5.0,
            'impact_factor': 0.1,
            'funding_rate': 0.0
        }
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the backtest."""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")
        
    def run_backtest(self, replay_engine: ReplayEngine, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            replay_engine: Data replay engine
            start_time: Start time for backtest
            end_time: End time for backtest
            progress_callback: Optional progress callback function
            
        Returns:
            Backtest results dictionary
        """
        logger.info("Starting backtest...")
        self.start_time = time.time()
        self.is_running = True
        
        try:
            # Initialize replay engine
            if start_time and end_time:
                replay_engine.start_replay(
                    replay_engine.instruments,
                    start_time,
                    end_time,
                    replay_engine.config['time_step']
                )
                
            # Main backtest loop
            while not replay_engine.is_replay_complete() and self.is_running:
                # Get next batch of market events
                events = replay_engine.get_next_events(max_events=100)
                
                if not events:
                    break
                    
                # Process events
                for event in events:
                    self._process_market_event(event)
                    
                # Update progress
                if progress_callback:
                    progress = replay_engine.get_replay_progress()
                    progress_callback(progress)
                    
                self.total_events += len(events)
                
            # Finalize backtest
            self.end_time = time.time()
            self.is_running = False
            
            # Generate results
            results = self._generate_results()
            
            logger.info(f"Backtest completed: {self.executed_trades} trades executed")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            self.is_running = False
            raise
            
    def _process_market_event(self, event):
        """Process a single market event."""
        self.current_time = event.timestamp
        
        # Update market data
        if event.instrument not in self.market_data:
            self.market_data[event.instrument] = {}
            
        self.market_data[event.instrument].update(event.data)
        
        # Update portfolio with current prices
        current_prices = {inst: data.get('close', data.get('mid', 0)) 
                         for inst, data in self.market_data.items()}
        self.portfolio.update_market_prices(current_prices, event.timestamp)
        
        # Generate signals from all strategies
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(self.market_data)
                
                for signal in signals:
                    self._process_signal(signal, strategy)
                    
            except Exception as e:
                logger.error(f"Error processing signals for strategy {strategy.name}: {e}")
                
    def _process_signal(self, signal, strategy: BaseStrategy):
        """Process a trading signal."""
        try:
            # Validate signal
            if not strategy.validate_signal(signal):
                return
                
            # Get current position
            current_position = self.portfolio.positions.get(signal.instrument)
            
            # Calculate position size
            position_size = strategy.calculate_position_size(
                signal, current_position, self.portfolio.get_portfolio_value()
            )
            
            if position_size <= 0:
                return
                
            # Determine trade side
            if signal.signal_type in ['long', 'close_short']:
                trade_side = 'buy'
            elif signal.signal_type in ['short', 'close_long']:
                trade_side = 'sell'
            else:
                return
                
            # Get current price
            current_price = self.market_data[signal.instrument].get('close', 
                                                                   self.market_data[signal.instrument].get('mid', signal.price))
            
            # Calculate transaction costs
            costs = self.cost_model.calculate_costs(
                signal.instrument, trade_side, position_size, current_price,
                self.market_data[signal.instrument], signal.timestamp
            )
            
            # Create trade
            trade = Trade(
                timestamp=signal.timestamp,
                instrument=signal.instrument,
                side=trade_side,
                size=position_size,
                price=current_price,
                commission=costs['commission'],
                slippage=costs['slippage'],
                total_cost=costs['total'],
                trade_id=f"{signal.timestamp.strftime('%Y%m%d_%H%M%S')}_{signal.instrument}_{self.executed_trades}",
                strategy=strategy.name
            )
            
            # Execute trade
            if self.portfolio.execute_trade(trade):
                self.executed_trades += 1
                strategy.log_signal(signal)
                
                # Update strategy position
                if signal.signal_type in ['long', 'short']:
                    strategy.update_position(
                        signal.instrument, 
                        'long' if signal.signal_type == 'long' else 'short',
                        position_size, 
                        current_price, 
                        signal.timestamp
                    )
                elif signal.signal_type in ['close_long', 'close_short']:
                    strategy.close_position(signal.instrument)
                    
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            
    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results."""
        # Portfolio performance
        portfolio_metrics = self.portfolio.get_performance_metrics()
        
        # Trade summary
        trade_summary = self.portfolio.get_trade_summary()
        
        # Strategy performance
        strategy_metrics = {}
        for strategy in self.strategies:
            strategy_metrics[strategy.name] = strategy.get_strategy_metrics()
            
        # Cost analysis
        trades = [trade.__dict__ for trade in self.portfolio.trades]
        cost_summary = self.cost_model.get_cost_summary(trades)
        
        # Performance analysis
        performance_analysis = self.performance_analyzer.analyze(
            self.portfolio.daily_returns,
            self.portfolio.daily_dates
        )
        
        # Backtest statistics
        backtest_stats = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.end_time - self.start_time if self.end_time else 0,
            'total_events': self.total_events,
            'executed_trades': self.executed_trades,
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio.get_portfolio_value()
        }
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'trade_summary': trade_summary,
            'strategy_metrics': strategy_metrics,
            'cost_summary': cost_summary,
            'performance_analysis': performance_analysis,
            'backtest_stats': backtest_stats,
            'daily_returns': self.portfolio.daily_returns,
            'daily_dates': [d.isoformat() for d in self.portfolio.daily_dates]
        }
        
    def stop_backtest(self):
        """Stop the backtest."""
        self.is_running = False
        logger.info("Backtest stopped")
        
    def reset(self):
        """Reset the backtest engine."""
        self.portfolio.reset()
        self.current_time = None
        self.market_data = {}
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.total_events = 0
        self.executed_trades = 0
        
        for strategy in self.strategies:
            strategy.reset()
            
    def get_current_status(self) -> Dict[str, Any]:
        """Get current backtest status."""
        return {
            'is_running': self.is_running,
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'total_events': self.total_events,
            'executed_trades': self.executed_trades,
            'portfolio_value': self.portfolio.get_portfolio_value(),
            'num_positions': len([p for p in self.portfolio.positions.values() if p['size'] != 0])
        }