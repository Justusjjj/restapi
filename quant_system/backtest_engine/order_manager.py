"""
Order Manager
=============

Order execution and fill simulation for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    instrument: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    filled_price: Optional[float] = None
    strategy: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class OrderManager:
    """
    Manages order execution and fill simulation.
    
    Features:
    - Order lifecycle management
    - Fill simulation
    - Partial fills
    - Order matching
    """
    
    def __init__(self):
        """Initialize order manager."""
        self.orders = {}
        self.order_counter = 0
        self.filled_orders = []
        
    def create_order(self, instrument: str, side: str, order_type: OrderType,
                    size: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    strategy: str = None) -> Order:
        """
        Create a new order.
        
        Args:
            instrument: Instrument symbol
            side: Order side ('buy' or 'sell')
            order_type: Type of order
            size: Order size
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            strategy: Strategy name
            
        Returns:
            Created order
        """
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            instrument=instrument,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            stop_price=stop_price,
            timestamp=datetime.now(),
            strategy=strategy
        )
        
        self.orders[order_id] = order
        logger.debug(f"Created order {order_id}: {side} {size} {instrument}")
        
        return order
        
    def process_market_data(self, market_data: Dict[str, Dict[str, float]]):
        """
        Process market data and check for order fills.
        
        Args:
            market_data: Current market data by instrument
        """
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                continue
                
            if order.instrument not in market_data:
                continue
                
            # Check if order should be filled
            if self._should_fill_order(order, market_data[order.instrument]):
                self._fill_order(order, market_data[order.instrument])
                
    def _should_fill_order(self, order: Order, market_data: Dict[str, float]) -> bool:
        """Check if order should be filled based on market data."""
        if 'bid' not in market_data or 'ask' not in market_data:
            return False
            
        bid = market_data['bid']
        ask = market_data['ask']
        mid = (bid + ask) / 2
        
        if order.order_type == OrderType.MARKET:
            return True
            
        elif order.order_type == OrderType.LIMIT:
            if order.side == 'buy' and order.price >= ask:
                return True
            elif order.side == 'sell' and order.price <= bid:
                return True
                
        elif order.order_type == OrderType.STOP:
            if order.side == 'buy' and order.stop_price <= mid:
                return True
            elif order.side == 'sell' and order.stop_price >= mid:
                return True
                
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == 'buy' and order.stop_price <= mid and order.price >= ask:
                return True
            elif order.side == 'sell' and order.stop_price >= mid and order.price <= bid:
                return True
                
        return False
        
    def _fill_order(self, order: Order, market_data: Dict[str, float]):
        """Fill an order."""
        bid = market_data['bid']
        ask = market_data['ask']
        mid = (bid + ask) / 2
        
        # Determine fill price
        if order.order_type == OrderType.MARKET:
            fill_price = ask if order.side == 'buy' else bid
        elif order.order_type == OrderType.LIMIT:
            fill_price = order.price
        else:
            fill_price = mid
            
        # For now, assume full fills (partial fills would be more complex)
        order.filled_size = order.size
        order.filled_price = fill_price
        order.status = OrderStatus.FILLED
        
        # Move to filled orders
        self.filled_orders.append(order)
        
        logger.debug(f"Filled order {order.order_id}: {order.filled_size} at {order.filled_price}")
        
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled
        """
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
            
        order.status = OrderStatus.CANCELLED
        logger.debug(f"Cancelled order {order_id}")
        
        return True
        
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        if order_id not in self.orders:
            return None
        return self.orders[order_id].status
        
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [order for order in self.orders.values() 
                if order.status == OrderStatus.PENDING]
                
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        return self.filled_orders.copy()
        
    def get_orders_by_strategy(self, strategy: str) -> List[Order]:
        """Get orders by strategy."""
        return [order for order in self.orders.values() 
                if order.strategy == strategy]
                
    def get_order_summary(self) -> Dict[str, Any]:
        """Get order summary statistics."""
        total_orders = len(self.orders)
        pending_orders = len(self.get_pending_orders())
        filled_orders = len(self.filled_orders)
        cancelled_orders = len([o for o in self.orders.values() 
                               if o.status == OrderStatus.CANCELLED])
        
        return {
            'total_orders': total_orders,
            'pending_orders': pending_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0
        }
        
    def reset(self):
        """Reset order manager."""
        self.orders = {}
        self.order_counter = 0
        self.filled_orders = []