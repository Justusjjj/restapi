"""
Portfolio Management
===================

Portfolio accounting, position tracking, and PnL calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    timestamp: datetime
    instrument: str
    side: str  # 'buy' or 'sell'
    size: float
    price: float
    commission: float
    slippage: float
    total_cost: float
    trade_id: str
    strategy: str = None
    
    @property
    def net_price(self) -> float:
        """Net price after costs."""
        if self.side == 'buy':
            return self.price + (self.total_cost / self.size)
        else:
            return self.price - (self.total_cost / self.size)


class Portfolio:
    """
    Portfolio management and accounting.
    
    Features:
    - Position tracking
    - PnL calculation
    - Cash management
    - Trade logging
    - Risk metrics
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 base_currency: str = 'USD'):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            base_currency: Base currency for accounting
        """
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.trade_counter = 0
        
        # Performance tracking
        self.daily_returns = []
        self.daily_pnl = []
        self.daily_dates = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        self.current_drawdown = 0.0
        
    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade and update portfolio.
        
        Args:
            trade: Trade to execute
            
        Returns:
            True if trade was executed successfully
        """
        try:
            # Check if we have sufficient capital
            if not self._check_capital_requirement(trade):
                logger.warning(f"Insufficient capital for trade {trade.trade_id}")
                return False
                
            # Update cash
            if trade.side == 'buy':
                self.cash -= (trade.price * trade.size + trade.total_cost)
            else:  # sell
                self.cash += (trade.price * trade.size - trade.total_cost)
                
            # Update position
            self._update_position(trade)
            
            # Record trade
            self.trades.append(trade)
            self.trade_counter += 1
            
            logger.debug(f"Executed trade {trade.trade_id}: {trade.side} {trade.size} "
                        f"{trade.instrument} at {trade.price}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade {trade.trade_id}: {e}")
            return False
            
    def _check_capital_requirement(self, trade: Trade) -> bool:
        """Check if we have sufficient capital for the trade."""
        if trade.side == 'buy':
            required_capital = trade.price * trade.size + trade.total_cost
            return self.cash >= required_capital
        else:  # sell
            # For selling, check if we have the position
            if trade.instrument in self.positions:
                return self.positions[trade.instrument]['size'] >= trade.size
            return False
            
    def _update_position(self, trade: Trade):
        """Update position after trade execution."""
        if trade.instrument not in self.positions:
            self.positions[trade.instrument] = {
                'size': 0.0,
                'avg_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
            
        pos = self.positions[trade.instrument]
        
        if trade.side == 'buy':
            # Add to position
            if pos['size'] >= 0:  # Same direction or new position
                total_size = pos['size'] + trade.size
                total_cost = (pos['avg_price'] * pos['size']) + (trade.price * trade.size)
                pos['avg_price'] = total_cost / total_size if total_size > 0 else 0
                pos['size'] = total_size
            else:  # Opposite direction - reduce or reverse
                if trade.size >= abs(pos['size']):
                    # Complete reversal
                    realized_pnl = (pos['avg_price'] - trade.price) * abs(pos['size'])
                    pos['realized_pnl'] += realized_pnl
                    pos['size'] = trade.size - abs(pos['size'])
                    pos['avg_price'] = trade.price
                else:
                    # Partial close
                    realized_pnl = (pos['avg_price'] - trade.price) * trade.size
                    pos['realized_pnl'] += realized_pnl
                    pos['size'] += trade.size
                    
        else:  # sell
            if pos['size'] <= 0:  # Same direction or new position
                total_size = pos['size'] - trade.size
                total_cost = (pos['avg_price'] * abs(pos['size'])) - (trade.price * trade.size)
                pos['avg_price'] = total_cost / abs(total_size) if total_size != 0 else 0
                pos['size'] = total_size
            else:  # Opposite direction - reduce or reverse
                if trade.size >= pos['size']:
                    # Complete reversal
                    realized_pnl = (trade.price - pos['avg_price']) * pos['size']
                    pos['realized_pnl'] += realized_pnl
                    pos['size'] = -(trade.size - pos['size'])
                    pos['avg_price'] = trade.price
                else:
                    # Partial close
                    realized_pnl = (trade.price - pos['avg_price']) * trade.size
                    pos['realized_pnl'] += realized_pnl
                    pos['size'] -= trade.size
                    
        # Clean up zero positions
        if abs(pos['size']) < 1e-8:
            pos['size'] = 0.0
            pos['avg_price'] = 0.0
            
    def update_market_prices(self, prices: Dict[str, float], timestamp: datetime):
        """
        Update market prices and calculate unrealized PnL.
        
        Args:
            prices: Dictionary of current prices by instrument
            timestamp: Current timestamp
        """
        total_unrealized_pnl = 0.0
        
        for instrument, position in self.positions.items():
            if position['size'] != 0 and instrument in prices:
                current_price = prices[instrument]
                
                if position['size'] > 0:  # Long position
                    position['unrealized_pnl'] = (current_price - position['avg_price']) * position['size']
                else:  # Short position
                    position['unrealized_pnl'] = (position['avg_price'] - current_price) * abs(position['size'])
                    
                total_unrealized_pnl += position['unrealized_pnl']
                
        # Calculate total portfolio value
        total_value = self.cash + total_unrealized_pnl
        
        # Update drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - total_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
        # Record daily performance
        self.daily_dates.append(timestamp)
        self.daily_pnl.append(total_unrealized_pnl)
        
        if len(self.daily_returns) > 0:
            daily_return = (total_value - self.initial_capital) / self.initial_capital
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)
            
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        return self.cash + total_unrealized_pnl
        
    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        total_realized = sum(pos['realized_pnl'] for pos in self.positions.values())
        total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        return total_realized + total_unrealized
        
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        summary = {}
        for instrument, pos in self.positions.items():
            if pos['size'] != 0:
                summary[instrument] = {
                    'size': pos['size'],
                    'avg_price': pos['avg_price'],
                    'unrealized_pnl': pos['unrealized_pnl'],
                    'realized_pnl': pos['realized_pnl']
                }
        return summary
        
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all trades."""
        if not self.trades:
            return {'total_trades': 0}
            
        df = pd.DataFrame([
            {
                'timestamp': trade.timestamp,
                'instrument': trade.instrument,
                'side': trade.side,
                'size': trade.size,
                'price': trade.price,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'total_cost': trade.total_cost
            }
            for trade in self.trades
        ])
        
        return {
            'total_trades': len(self.trades),
            'total_commission': df['commission'].sum(),
            'total_slippage': df['slippage'].sum(),
            'total_costs': df['total_cost'].sum(),
            'trades_by_instrument': df['instrument'].value_counts().to_dict(),
            'trades_by_side': df['side'].value_counts().to_dict()
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        if not self.daily_returns:
            return {}
            
        returns = pd.Series(self.daily_returns)
        
        # Basic metrics
        total_return = returns.iloc[-1] if len(returns) > 0 else 0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': self.current_drawdown,
            'portfolio_value': self.get_portfolio_value(),
            'cash': self.cash,
            'total_pnl': self.get_total_pnl(),
            'num_positions': len([p for p in self.positions.values() if p['size'] != 0])
        }
        
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.trade_counter = 0
        self.daily_returns = []
        self.daily_pnl = []
        self.daily_dates = []
        self.max_drawdown = 0.0
        self.peak_value = self.initial_capital
        self.current_drawdown = 0.0