import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time


class OrderType(Enum):
	MARKET = "market"
	LIMIT = "limit"
	STOP = "stop"
	STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
	PENDING = "pending"
	FILLED = "filled"
	PARTIAL = "partial"
	CANCELLED = "cancelled"
	REJECTED = "rejected"


@dataclass
class AdvancedOrder:
	"""Advanced order with scaling and partial fill management"""
	symbol: str
	order_type: OrderType
	side: str  # BUY/SELL
	volume: float
	price: Optional[float] = None
	stop_loss: Optional[float] = None
	take_profit: Optional[float] = None
	
	# Advanced features
	scaling_enabled: bool = False
	scale_levels: List[float] = None
	scale_volumes: List[float] = None
	partial_fill_enabled: bool = True
	min_fill_size: float = 0.01
	
	# Order management
	order_id: Optional[int] = None
	status: OrderStatus = OrderStatus.PENDING
	filled_volume: float = 0.0
	remaining_volume: float = 0.0
	created_at: datetime = None
	updated_at: datetime = None
	
	# Risk management
	max_position_size: float = 1.0
	risk_per_trade: float = 0.01
	
	def __post_init__(self):
		if self.created_at is None:
			self.created_at = datetime.now()
		if self.updated_at is None:
			self.updated_at = datetime.now()
		if self.remaining_volume == 0.0:
			self.remaining_volume = self.volume


class AdvancedOrderManager:
	"""
	Advanced order management system with partial fills, scaling, and position management.
	"""
	
	def __init__(self, mt5_connector=None):
		self.mt5_connector = mt5_connector
		self.active_orders: Dict[int, AdvancedOrder] = {}
		self.position_tracker: Dict[str, Dict] = {}
		self.scaling_orders: Dict[str, List[AdvancedOrder]] = {}
		self.order_history: List[AdvancedOrder] = []
		
		# Order management settings
		self.max_orders_per_symbol = 5
		self.max_total_orders = 20
		self.order_timeout_minutes = 30
		
		# Start order monitoring thread
		self.monitoring_active = True
		self.monitor_thread = threading.Thread(target=self._monitor_orders, daemon=True)
		self.monitor_thread.start()
	
	def place_advanced_order(self, order: AdvancedOrder) -> Dict:
		"""
		Place an advanced order with scaling and partial fill management
		"""
		
		# Validate order
		validation_result = self._validate_order(order)
		if not validation_result["valid"]:
			return {
				"success": False,
				"error": validation_result["error"],
				"order_id": None
			}
		
		# Check if scaling is enabled
		if order.scaling_enabled and order.scale_levels:
			return self._place_scaling_order(order)
		else:
			return self._place_single_order(order)
	
	def _place_single_order(self, order: AdvancedOrder) -> Dict:
		"""Place a single order"""
		
		try:
			# Convert to MT5 order type
			mt5_order_type = self._convert_to_mt5_order_type(order)
			
			# Prepare order request
			request = {
				"action": mt5.TRADE_ACTION_DEAL,
				"symbol": order.symbol,
				"volume": order.volume,
				"type": mt5_order_type,
				"price": order.price or self._get_current_price(order.symbol, order.side),
				"sl": order.stop_loss,
				"tp": order.take_profit,
				"deviation": 20,
				"magic": 12345,
				"comment": f"Advanced Order {order.order_type.value}",
				"type_time": mt5.ORDER_TIME_GTC,
				"type_filling": mt5.ORDER_FILLING_IOC if order.partial_fill_enabled else mt5.ORDER_FILLING_FOK,
			}
			
			# Send order
			result = mt5.order_send(request)
			
			if result.retcode == mt5.TRADE_RETCODE_DONE:
				order.order_id = result.order
				order.status = OrderStatus.FILLED
				order.filled_volume = order.volume
				order.remaining_volume = 0.0
				
				# Track the order
				self.active_orders[result.order] = order
				self.order_history.append(order)
				
				# Update position tracker
				self._update_position_tracker(order)
				
				return {
					"success": True,
					"order_id": result.order,
					"filled_volume": order.filled_volume,
					"remaining_volume": order.remaining_volume,
					"status": order.status.value
				}
			else:
				order.status = OrderStatus.REJECTED
				return {
					"success": False,
					"error": f"Order rejected: {result.comment}",
					"retcode": result.retcode,
					"order_id": None
				}
		
		except Exception as e:
			return {
				"success": False,
				"error": f"Order placement error: {str(e)}",
				"order_id": None
			}
	
	def _place_scaling_order(self, order: AdvancedOrder) -> Dict:
		"""Place a scaling order with multiple levels"""
		
		try:
			scaling_orders = []
			total_volume = 0
			
			# Create scaling orders
			for i, (level, volume) in enumerate(zip(order.scale_levels, order.scale_volumes)):
				scale_order = AdvancedOrder(
					symbol=order.symbol,
					order_type=order.order_type,
					side=order.side,
					volume=volume,
					price=level,
					stop_loss=order.stop_loss,
					take_profit=order.take_profit,
					scaling_enabled=False,  # Individual orders are not scaling
					partial_fill_enabled=order.partial_fill_enabled,
					min_fill_size=order.min_fill_size
				)
				
				# Place individual order
				result = self._place_single_order(scale_order)
				
				if result["success"]:
					scaling_orders.append(scale_order)
					total_volume += volume
				else:
					print(f"Scaling order {i+1} failed: {result['error']}")
			
			# Track scaling group
			if scaling_orders:
				scaling_key = f"{order.symbol}_{order.side}_{datetime.now().timestamp()}"
				self.scaling_orders[scaling_key] = scaling_orders
				
				return {
					"success": True,
					"scaling_key": scaling_key,
					"orders_placed": len(scaling_orders),
					"total_volume": total_volume,
					"order_ids": [o.order_id for o in scaling_orders if o.order_id]
				}
			else:
				return {
					"success": False,
					"error": "All scaling orders failed",
					"orders_placed": 0
				}
		
		except Exception as e:
			return {
				"success": False,
				"error": f"Scaling order error: {str(e)}",
				"orders_placed": 0
			}
	
	def _validate_order(self, order: AdvancedOrder) -> Dict:
		"""Validate order before placement"""
		
		# Check symbol
		if not order.symbol:
			return {"valid": False, "error": "Symbol is required"}
		
		# Check volume
		if order.volume <= 0:
			return {"valid": False, "error": "Volume must be positive"}
		
		# Check max position size
		current_position = self.position_tracker.get(order.symbol, {}).get("volume", 0)
		if abs(current_position + order.volume) > order.max_position_size:
			return {"valid": False, "error": f"Position size would exceed maximum ({order.max_position_size})"}
		
		# Check order limits
		symbol_orders = [o for o in self.active_orders.values() if o.symbol == order.symbol]
		if len(symbol_orders) >= self.max_orders_per_symbol:
			return {"valid": False, "error": f"Too many orders for {order.symbol}"}
		
		if len(self.active_orders) >= self.max_total_orders:
			return {"valid": False, "error": "Too many total orders"}
		
		# Check scaling parameters
		if order.scaling_enabled:
			if not order.scale_levels or not order.scale_volumes:
				return {"valid": False, "error": "Scaling levels and volumes required"}
			
			if len(order.scale_levels) != len(order.scale_volumes):
				return {"valid": False, "error": "Scaling levels and volumes must match"}
		
		return {"valid": True}
	
	def _convert_to_mt5_order_type(self, order: AdvancedOrder) -> int:
		"""Convert order type to MT5 format"""
		
		if order.order_type == OrderType.MARKET:
			return mt5.ORDER_TYPE_BUY if order.side == "BUY" else mt5.ORDER_TYPE_SELL
		elif order.order_type == OrderType.LIMIT:
			return mt5.ORDER_TYPE_BUY_LIMIT if order.side == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
		elif order.order_type == OrderType.STOP:
			return mt5.ORDER_TYPE_BUY_STOP if order.side == "BUY" else mt5.ORDER_TYPE_SELL_STOP
		elif order.order_type == OrderType.STOP_LIMIT:
			return mt5.ORDER_TYPE_BUY_STOP_LIMIT if order.side == "BUY" else mt5.ORDER_TYPE_SELL_STOP_LIMIT
		else:
			return mt5.ORDER_TYPE_BUY
	
	def _get_current_price(self, symbol: str, side: str) -> float:
		"""Get current market price"""
		
		try:
			tick = mt5.symbol_info_tick(symbol)
			return tick.ask if side == "BUY" else tick.bid
		except:
			return 0.0
	
	def _update_position_tracker(self, order: AdvancedOrder):
		"""Update position tracker with new order"""
		
		if order.symbol not in self.position_tracker:
			self.position_tracker[order.symbol] = {
				"volume": 0.0,
				"avg_price": 0.0,
				"unrealized_pnl": 0.0,
				"orders": []
			}
		
		position = self.position_tracker[order.symbol]
		
		# Update volume and average price
		if order.side == "BUY":
			new_volume = position["volume"] + order.filled_volume
			if new_volume != 0:
				position["avg_price"] = (
					(position["volume"] * position["avg_price"] + 
					 order.filled_volume * order.price) / new_volume
				)
			position["volume"] = new_volume
		else:  # SELL
			position["volume"] -= order.filled_volume
		
		position["orders"].append(order.order_id)
	
	def _monitor_orders(self):
		"""Monitor active orders for status changes"""
		
		while self.monitoring_active:
			try:
				# Check for order updates
				for order_id, order in list(self.active_orders.items()):
					self._check_order_status(order)
				
				# Clean up old orders
				self._cleanup_old_orders()
				
				# Update position tracker
				self._update_all_positions()
				
				time.sleep(5)  # Check every 5 seconds
				
			except Exception as e:
				print(f"Order monitoring error: {e}")
				time.sleep(10)
	
	def _check_order_status(self, order: AdvancedOrder):
		"""Check and update order status"""
		
		try:
			if order.order_id:
				# Get order info from MT5
				order_info = mt5.orders_get(ticket=order.order_id)
				
				if order_info:
					mt5_order = order_info[0]
					
					# Update status based on MT5 order
					if mt5_order.state == mt5.ORDER_STATE_FILLED:
						order.status = OrderStatus.FILLED
						order.filled_volume = mt5_order.volume_current
						order.remaining_volume = 0.0
					elif mt5_order.state == mt5.ORDER_STATE_PARTIAL:
						order.status = OrderStatus.PARTIAL
						order.filled_volume = mt5_order.volume_current
						order.remaining_volume = mt5_order.volume_initial - mt5_order.volume_current
					elif mt5_order.state == mt5.ORDER_STATE_CANCELED:
						order.status = OrderStatus.CANCELLED
					
					order.updated_at = datetime.now()
		
		except Exception as e:
			print(f"Error checking order status: {e}")
	
	def _cleanup_old_orders(self):
		"""Remove old completed orders"""
		
		current_time = datetime.now()
		
		for order_id, order in list(self.active_orders.items()):
			# Remove orders older than timeout
			if (current_time - order.created_at).total_seconds() > self.order_timeout_minutes * 60:
				if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
					del self.active_orders[order_id]
	
	def _update_all_positions(self):
		"""Update all position P&L"""
		
		for symbol, position in self.position_tracker.items():
			if position["volume"] != 0:
				try:
					# Get current price
					tick = mt5.symbol_info_tick(symbol)
					current_price = (tick.bid + tick.ask) / 2
					
					# Calculate unrealized P&L
					if position["volume"] > 0:  # Long position
						position["unrealized_pnl"] = (current_price - position["avg_price"]) * position["volume"]
					else:  # Short position
						position["unrealized_pnl"] = (position["avg_price"] - current_price) * abs(position["volume"])
				
				except Exception as e:
					print(f"Error updating position for {symbol}: {e}")
	
	def cancel_order(self, order_id: int) -> Dict:
		"""Cancel an active order"""
		
		if order_id not in self.active_orders:
			return {"success": False, "error": "Order not found"}
		
		try:
			# Cancel order in MT5
			request = {
				"action": mt5.TRADE_ACTION_REMOVE,
				"order": order_id,
			}
			
			result = mt5.order_send(request)
			
			if result.retcode == mt5.TRADE_RETCODE_DONE:
				order = self.active_orders[order_id]
				order.status = OrderStatus.CANCELLED
				order.updated_at = datetime.now()
				
				return {"success": True, "order_id": order_id}
			else:
				return {"success": False, "error": f"Cancel failed: {result.comment}"}
		
		except Exception as e:
			return {"success": False, "error": f"Cancel error: {str(e)}"}
	
	def get_position_summary(self, symbol: str = None) -> Dict:
		"""Get position summary"""
		
		if symbol:
			return self.position_tracker.get(symbol, {})
		else:
			return self.position_tracker
	
	def get_order_status(self, order_id: int) -> Dict:
		"""Get detailed order status"""
		
		if order_id not in self.active_orders:
			return {"error": "Order not found"}
		
		order = self.active_orders[order_id]
		
		return {
			"order_id": order_id,
			"symbol": order.symbol,
			"side": order.side,
			"volume": order.volume,
			"filled_volume": order.filled_volume,
			"remaining_volume": order.remaining_volume,
			"status": order.status.value,
			"created_at": order.created_at.isoformat(),
			"updated_at": order.updated_at.isoformat(),
			"price": order.price,
			"stop_loss": order.stop_loss,
			"take_profit": order.take_profit
		}
	
	def get_active_orders(self) -> Dict:
		"""Get all active orders"""
		
		return {
			order_id: {
				"symbol": order.symbol,
				"side": order.side,
				"volume": order.volume,
				"status": order.status.value,
				"created_at": order.created_at.isoformat()
			}
			for order_id, order in self.active_orders.items()
		}
	
	def close_position(self, symbol: str, volume: float = None) -> Dict:
		"""Close position or part of position"""
		
		if symbol not in self.position_tracker:
			return {"success": False, "error": "No position found"}
		
		position = self.position_tracker[symbol]
		current_volume = position["volume"]
		
		if current_volume == 0:
			return {"success": False, "error": "No position to close"}
		
		# Determine close volume
		close_volume = volume if volume else abs(current_volume)
		close_volume = min(close_volume, abs(current_volume))
		
		# Determine close side (opposite to current position)
		close_side = "SELL" if current_volume > 0 else "BUY"
		
		# Create close order
		close_order = AdvancedOrder(
			symbol=symbol,
			order_type=OrderType.MARKET,
			side=close_side,
			volume=close_volume,
			partial_fill_enabled=True
		)
		
		# Place close order
		result = self.place_advanced_order(close_order)
		
		if result["success"]:
			# Update position tracker
			if current_volume > 0:
				position["volume"] -= close_volume
			else:
				position["volume"] += close_volume
			
			return {
				"success": True,
				"closed_volume": close_volume,
				"remaining_volume": position["volume"],
				"order_id": result["order_id"]
			}
		else:
			return result
	
	def stop_monitoring(self):
		"""Stop order monitoring thread"""
		
		self.monitoring_active = False
		if self.monitor_thread.is_alive():
			self.monitor_thread.join(timeout=5)