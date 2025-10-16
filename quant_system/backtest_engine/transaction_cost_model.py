"""
Transaction Cost Model
=====================

Realistic transaction cost modeling including spreads, commissions, and slippage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Models realistic transaction costs for backtesting.
    
    Cost Components:
    - Spread costs (bid-ask spread)
    - Commission costs (per trade or per unit)
    - Slippage costs (market impact)
    - Funding costs (for leveraged positions)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize transaction cost model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Commission settings
        self.commission_per_trade = config.get('commission_per_trade', 0.0)
        self.commission_per_unit = config.get('commission_per_unit', 0.0)
        self.commission_per_value = config.get('commission_per_value', 0.0)  # As percentage
        
        # Spread settings
        self.spread_bps = config.get('spread_bps', 1.0)  # Basis points
        self.spread_model = config.get('spread_model', 'fixed')  # 'fixed', 'dynamic', 'realistic'
        
        # Slippage settings
        self.slippage_bps = config.get('slippage_bps', 0.5)
        self.slippage_model = config.get('slippage_model', 'linear')  # 'linear', 'sqrt', 'log'
        self.max_slippage_bps = config.get('max_slippage_bps', 10.0)
        
        # Market impact settings
        self.impact_factor = config.get('impact_factor', 0.1)
        self.volume_threshold = config.get('volume_threshold', 1000)
        
        # Funding costs
        self.funding_rate = config.get('funding_rate', 0.0)  # Daily rate
        self.funding_frequency = config.get('funding_frequency', 'daily')
        
    def calculate_costs(self, instrument: str, side: str, size: float, 
                       price: float, market_data: Optional[Dict] = None,
                       timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate total transaction costs for a trade.
        
        Args:
            instrument: Instrument symbol
            side: Trade side ('buy' or 'sell')
            size: Trade size
            price: Trade price
            market_data: Optional market data for dynamic pricing
            timestamp: Optional timestamp for time-based costs
            
        Returns:
            Dictionary with cost breakdown
        """
        costs = {
            'commission': 0.0,
            'spread': 0.0,
            'slippage': 0.0,
            'market_impact': 0.0,
            'funding': 0.0,
            'total': 0.0
        }
        
        # Commission costs
        costs['commission'] = self._calculate_commission(size, price)
        
        # Spread costs
        costs['spread'] = self._calculate_spread_cost(instrument, side, size, price, market_data)
        
        # Slippage costs
        costs['slippage'] = self._calculate_slippage(instrument, side, size, price, market_data)
        
        # Market impact
        costs['market_impact'] = self._calculate_market_impact(instrument, side, size, price, market_data)
        
        # Funding costs (for leveraged positions)
        costs['funding'] = self._calculate_funding_cost(instrument, side, size, price, timestamp)
        
        # Total costs
        costs['total'] = sum(costs.values())
        
        return costs
        
    def _calculate_commission(self, size: float, price: float) -> float:
        """Calculate commission costs."""
        commission = 0.0
        
        # Per trade commission
        commission += self.commission_per_trade
        
        # Per unit commission
        commission += self.commission_per_unit * size
        
        # Per value commission (as percentage)
        commission += self.commission_per_value * size * price
        
        return commission
        
    def _calculate_spread_cost(self, instrument: str, side: str, size: float, 
                              price: float, market_data: Optional[Dict]) -> float:
        """Calculate spread costs."""
        if self.spread_model == 'fixed':
            # Fixed spread in basis points
            spread_bps = self.spread_bps
        elif self.spread_model == 'dynamic' and market_data:
            # Dynamic spread based on market conditions
            spread_bps = self._get_dynamic_spread(instrument, market_data)
        elif self.spread_model == 'realistic' and market_data:
            # Realistic spread based on actual bid-ask data
            spread_bps = self._get_realistic_spread(instrument, market_data)
        else:
            spread_bps = self.spread_bps
            
        # Convert basis points to price
        spread_price = price * (spread_bps / 10000)
        
        # Spread cost is half the spread (we pay half when crossing)
        return spread_price * size * 0.5
        
    def _calculate_slippage(self, instrument: str, side: str, size: float,
                           price: float, market_data: Optional[Dict]) -> float:
        """Calculate slippage costs."""
        if self.slippage_model == 'linear':
            # Linear slippage based on size
            slippage_bps = min(self.slippage_bps * (size / 1000), self.max_slippage_bps)
        elif self.slippage_model == 'sqrt':
            # Square root slippage (more realistic for large orders)
            slippage_bps = min(self.slippage_bps * np.sqrt(size / 1000), self.max_slippage_bps)
        elif self.slippage_model == 'log':
            # Logarithmic slippage
            slippage_bps = min(self.slippage_bps * np.log(1 + size / 1000), self.max_slippage_bps)
        else:
            slippage_bps = self.slippage_bps
            
        # Convert to price
        slippage_price = price * (slippage_bps / 10000)
        
        return slippage_price * size
        
    def _calculate_market_impact(self, instrument: str, side: str, size: float,
                                price: float, market_data: Optional[Dict]) -> float:
        """Calculate market impact costs."""
        if not market_data or 'volume' not in market_data:
            return 0.0
            
        # Market impact increases with order size relative to market volume
        volume = market_data['volume']
        if volume <= 0:
            return 0.0
            
        # Impact factor scales with size/volume ratio
        impact_ratio = size / volume
        impact_bps = self.impact_factor * impact_ratio * 10000  # Convert to basis points
        
        # Cap impact at reasonable level
        impact_bps = min(impact_bps, self.max_slippage_bps)
        
        impact_price = price * (impact_bps / 10000)
        return impact_price * size
        
    def _calculate_funding_cost(self, instrument: str, side: str, size: float,
                               price: float, timestamp: Optional[datetime]) -> float:
        """Calculate funding costs for leveraged positions."""
        if self.funding_rate <= 0:
            return 0.0
            
        # Simple daily funding cost
        # In practice, this would be more complex based on position duration
        position_value = size * price
        daily_funding = position_value * self.funding_rate
        
        return daily_funding
        
    def _get_dynamic_spread(self, instrument: str, market_data: Dict) -> float:
        """Get dynamic spread based on market conditions."""
        base_spread = self.spread_bps
        
        # Adjust based on volatility
        if 'volatility' in market_data:
            vol_multiplier = 1 + (market_data['volatility'] - 0.02) * 10  # Scale volatility
            base_spread *= vol_multiplier
            
        # Adjust based on volume
        if 'volume' in market_data and 'avg_volume' in market_data:
            volume_ratio = market_data['volume'] / market_data['avg_volume']
            if volume_ratio < 0.5:  # Low volume
                base_spread *= 1.5
            elif volume_ratio > 2.0:  # High volume
                base_spread *= 0.8
                
        return max(0.1, base_spread)  # Minimum spread
        
    def _get_realistic_spread(self, instrument: str, market_data: Dict) -> float:
        """Get realistic spread from actual bid-ask data."""
        if 'bid' in market_data and 'ask' in market_data:
            bid = market_data['bid']
            ask = market_data['ask']
            if bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2
                spread_bps = ((ask - bid) / mid_price) * 10000
                return max(0.1, spread_bps)
                
        return self.spread_bps
        
    def get_cost_summary(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Get summary of transaction costs across all trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Cost summary dictionary
        """
        if not trades:
            return {}
            
        df = pd.DataFrame(trades)
        
        return {
            'total_commission': df['commission'].sum(),
            'total_spread': df['spread'].sum(),
            'total_slippage': df['slippage'].sum(),
            'total_market_impact': df['market_impact'].sum(),
            'total_funding': df['funding'].sum(),
            'total_costs': df['total_cost'].sum(),
            'avg_cost_per_trade': df['total_cost'].mean(),
            'cost_breakdown': {
                'commission_pct': (df['commission'].sum() / df['total_cost'].sum()) * 100,
                'spread_pct': (df['spread'].sum() / df['total_cost'].sum()) * 100,
                'slippage_pct': (df['slippage'].sum() / df['total_cost'].sum()) * 100,
                'market_impact_pct': (df['market_impact'].sum() / df['total_cost'].sum()) * 100,
                'funding_pct': (df['funding'].sum() / df['total_cost'].sum()) * 100
            }
        }