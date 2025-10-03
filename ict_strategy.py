import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5
from abc import ABC, abstractmethod


class StrategyType(Enum):
	ICT = "ict"
	MOMENTUM = "momentum"
	MEAN_REVERSION = "mean_reversion"
	BREAKOUT = "breakout"
	SCALPING = "scalping"
	RANGE_TRADING = "range_trading"
	MULTI_TIMEFRAME = "multi_timeframe"


class SignalStrength(Enum):
	WEAK = 1
	MEDIUM = 2
	STRONG = 3
	VERY_STRONG = 4


@dataclass
class TradingSignal:
	"""Comprehensive trading signal with all necessary information"""
	symbol: str
	action: str  # BUY, SELL, HOLD
	entry_price: float
	stop_loss: float
	take_profit: float
	volume: float
	confidence: float
	signal_strength: SignalStrength
	strategy_name: str
	timeframe: str
	timestamp: datetime
	reason: str
	risk_reward_ratio: float
	market_regime: str
	additional_info: Dict = None


class BaseStrategy(ABC):
	"""
	Base class for all trading strategies
	"""
	
	def __init__(self, name: str, symbol: str, timeframe: str = "H1"):
		self.name = name
		self.symbol = symbol
		self.timeframe = timeframe
		self.parameters = {}
		self.performance_metrics = {}
		self.last_signal = None
		self.active_positions = []
		
	@abstractmethod
	def generate_signal(self, data: pd.DataFrame, context: Dict = None) -> Optional[TradingSignal]:
		"""Generate trading signal based on strategy logic"""
		pass
	
	@abstractmethod
	def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float = 0.01) -> float:
		"""Calculate optimal position size based on risk management"""
		pass
	
	def update_parameters(self, new_params: Dict):
		"""Update strategy parameters"""
		self.parameters.update(new_params)
	
	def get_performance_metrics(self) -> Dict:
		"""Get strategy performance metrics"""
		return self.performance_metrics
	
	def validate_signal(self, signal: TradingSignal) -> bool:
		"""Validate signal before execution"""
		if not signal:
			return False
		
		# Basic validation
		if signal.action not in ["BUY", "SELL", "HOLD"]:
			return False
		
		if signal.confidence < 0.0 or signal.confidence > 1.0:
			return False
		
		if signal.risk_reward_ratio < 1.0:
			return False
		
		return True


class ICTStrategy(BaseStrategy):
	"""
	ICT (Inner Circle Trader) based trading strategy
	Focuses on Order Blocks, Fair Value Gaps, and Liquidity Sweeps
	"""
	
	def __init__(self, symbol: str, timeframe: str = "H1"):
		super().__init__("ICT Strategy", symbol, timeframe)
		self.parameters = {
			"order_block_lookback": 20,
			"fvg_lookback": 10,
			"liquidity_sweep_threshold": 0.0001,
			"min_confidence": 0.6,
			"risk_reward_min": 1.5
		}
	
	def generate_signal(self, data: pd.DataFrame, context: Dict = None) -> Optional[TradingSignal]:
		"""Generate ICT-based trading signal"""
		
		if len(data) < 50:
			return None
		
		# Get latest price
		current_price = data['close'].iloc[-1]
		current_time = data.index[-1]
		
		# Analyze ICT patterns
		order_blocks = self._detect_order_blocks(data)
		fair_value_gaps = self._detect_fair_value_gaps(data)
		liquidity_sweeps = self._detect_liquidity_sweeps(data)
		market_structure = self._analyze_market_structure(data)
		
		# Combine signals
		signal_strength = self._calculate_signal_strength(order_blocks, fair_value_gaps, liquidity_sweeps, market_structure)
		
		if signal_strength < self.parameters["min_confidence"]:
			return None
		
		# Determine action
		action = self._determine_action(order_blocks, fair_value_gaps, liquidity_sweeps, market_structure)
		
		if action == "HOLD":
			return None
		
		# Calculate entry, stop loss, and take profit
		entry_price = current_price
		stop_loss, take_profit = self._calculate_levels(action, data, order_blocks, fair_value_gaps)
		
		# Calculate risk-reward ratio
		risk = abs(entry_price - stop_loss)
		reward = abs(take_profit - entry_price)
		risk_reward_ratio = reward / risk if risk > 0 else 0
		
		if risk_reward_ratio < self.parameters["risk_reward_min"]:
			return None
		
		# Create signal
		signal = TradingSignal(
			symbol=self.symbol,
			action=action,
			entry_price=entry_price,
			stop_loss=stop_loss,
			take_profit=take_profit,
			volume=0.01,  # Will be calculated later
			confidence=signal_strength,
			signal_strength=SignalStrength.STRONG if signal_strength > 0.8 else SignalStrength.MEDIUM,
			strategy_name=self.name,
			timeframe=self.timeframe,
			timestamp=current_time,
			reason=f"ICT: {action} signal based on Order Blocks, FVGs, and Liquidity",
			risk_reward_ratio=risk_reward_ratio,
			market_regime=context.get("market_regime", "normal") if context else "normal",
			additional_info={
				"order_blocks": order_blocks,
				"fair_value_gaps": fair_value_gaps,
				"liquidity_sweeps": liquidity_sweeps,
				"market_structure": market_structure
			}
		)
		
		return signal
	
	def _detect_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
		"""Detect Order Blocks in the data"""
		
		order_blocks = []
		lookback = self.parameters["order_block_lookback"]
		
		for i in range(lookback, len(data) - 1):
			# Check for bullish order block
			if self._is_bullish_order_block(data, i):
				order_blocks.append({
					"type": "bullish",
					"index": i,
					"high": data['high'].iloc[i],
					"low": data['low'].iloc[i],
					"strength": self._calculate_order_block_strength(data, i, "bullish")
				})
			
			# Check for bearish order block
			elif self._is_bearish_order_block(data, i):
				order_blocks.append({
					"type": "bearish",
					"index": i,
					"high": data['high'].iloc[i],
					"low": data['low'].iloc[i],
					"strength": self._calculate_order_block_strength(data, i, "bearish")
				})
		
		return order_blocks[-3:]  # Return last 3 order blocks
	
	def _is_bullish_order_block(self, data: pd.DataFrame, index: int) -> bool:
		"""Check if current candle is a bullish order block"""
		
		current = data.iloc[index]
		next_candle = data.iloc[index + 1]
		
		# Bullish order block: strong bullish candle followed by pullback
		if (current['close'] > current['open'] and  # Bullish candle
			current['close'] - current['open'] > (current['high'] - current['low']) * 0.6 and  # Strong bullish
			next_candle['close'] < current['close']):  # Pullback
			return True
		
		return False
	
	def _is_bearish_order_block(self, data: pd.DataFrame, index: int) -> bool:
		"""Check if current candle is a bearish order block"""
		
		current = data.iloc[index]
		next_candle = data.iloc[index + 1]
		
		# Bearish order block: strong bearish candle followed by pullback
		if (current['close'] < current['open'] and  # Bearish candle
			current['open'] - current['close'] > (current['high'] - current['low']) * 0.6 and  # Strong bearish
			next_candle['close'] > current['close']):  # Pullback
			return True
		
		return False
	
	def _detect_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict]:
		"""Detect Fair Value Gaps in the data"""
		
		fair_value_gaps = []
		lookback = self.parameters["fvg_lookback"]
		
		for i in range(2, len(data) - 1):
			# Check for bullish FVG
			if self._is_bullish_fvg(data, i):
				fair_value_gaps.append({
					"type": "bullish",
					"index": i,
					"top": data['low'].iloc[i + 1],
					"bottom": data['high'].iloc[i - 1],
					"strength": self._calculate_fvg_strength(data, i, "bullish")
				})
			
			# Check for bearish FVG
			elif self._is_bearish_fvg(data, i):
				fair_value_gaps.append({
					"type": "bearish",
					"index": i,
					"top": data['low'].iloc[i - 1],
					"bottom": data['high'].iloc[i + 1],
					"strength": self._calculate_fvg_strength(data, i, "bearish")
				})
		
		return fair_value_gaps[-2:]  # Return last 2 FVGs
	
	def _is_bullish_fvg(self, data: pd.DataFrame, index: int) -> bool:
		"""Check if there's a bullish Fair Value Gap"""
		
		prev_candle = data.iloc[index - 1]
		current_candle = data.iloc[index]
		next_candle = data.iloc[index + 1]
		
		# Bullish FVG: gap between high of previous candle and low of next candle
		return (prev_candle['high'] < next_candle['low'] and
				current_candle['close'] > current_candle['open'])
	
	def _is_bearish_fvg(self, data: pd.DataFrame, index: int) -> bool:
		"""Check if there's a bearish Fair Value Gap"""
		
		prev_candle = data.iloc[index - 1]
		current_candle = data.iloc[index]
		next_candle = data.iloc[index + 1]
		
		# Bearish FVG: gap between low of previous candle and high of next candle
		return (prev_candle['low'] > next_candle['high'] and
				current_candle['close'] < current_candle['open'])
	
	def _detect_liquidity_sweeps(self, data: pd.DataFrame) -> List[Dict]:
		"""Detect Liquidity Sweeps"""
		
		liquidity_sweeps = []
		threshold = self.parameters["liquidity_sweep_threshold"]
		
		for i in range(20, len(data)):
			# Check for liquidity sweep of highs
			if self._is_liquidity_sweep_high(data, i, threshold):
				liquidity_sweeps.append({
					"type": "high_sweep",
					"index": i,
					"level": data['high'].iloc[i],
					"strength": self._calculate_sweep_strength(data, i, "high")
				})
			
			# Check for liquidity sweep of lows
			elif self._is_liquidity_sweep_low(data, i, threshold):
				liquidity_sweeps.append({
					"type": "low_sweep",
					"index": i,
					"level": data['low'].iloc[i],
					"strength": self._calculate_sweep_strength(data, i, "low")
				})
		
		return liquidity_sweeps[-2:]  # Return last 2 sweeps
	
	def _is_liquidity_sweep_high(self, data: pd.DataFrame, index: int, threshold: float) -> bool:
		"""Check if current candle sweeps liquidity above recent highs"""
		
		recent_highs = data['high'].iloc[index-20:index].max()
		current_high = data['high'].iloc[index]
		
		return current_high > recent_highs + threshold
	
	def _is_liquidity_sweep_low(self, data: pd.DataFrame, index: int, threshold: float) -> bool:
		"""Check if current candle sweeps liquidity below recent lows"""
		
		recent_lows = data['low'].iloc[index-20:index].min()
		current_low = data['low'].iloc[index]
		
		return current_low < recent_lows - threshold
	
	def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
		"""Analyze market structure for trend direction"""
		
		# Calculate higher highs and higher lows
		recent_data = data.tail(50)
		
		# Find swing highs and lows
		swing_highs = []
		swing_lows = []
		
		for i in range(2, len(recent_data) - 2):
			if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
				recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]):
				swing_highs.append(recent_data['high'].iloc[i])
			
			if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
				recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]):
				swing_lows.append(recent_data['low'].iloc[i])
		
		# Determine trend
		if len(swing_highs) >= 2 and len(swing_lows) >= 2:
			trend = "uptrend" if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2] else "downtrend"
		else:
			trend = "sideways"
		
		return {
			"trend": trend,
			"swing_highs": swing_highs[-3:],
			"swing_lows": swing_lows[-3:],
			"structure_break": self._check_structure_break(data)
		}
	
	def _check_structure_break(self, data: pd.DataFrame) -> Optional[str]:
		"""Check for market structure break"""
		
		recent_data = data.tail(20)
		
		# Check for break of recent high
		if recent_data['close'].iloc[-1] > recent_data['high'].iloc[:-1].max():
			return "bullish_break"
		
		# Check for break of recent low
		if recent_data['close'].iloc[-1] < recent_data['low'].iloc[:-1].min():
			return "bearish_break"
		
		return None
	
	def _calculate_signal_strength(self, order_blocks: List, fair_value_gaps: List, 
								liquidity_sweeps: List, market_structure: Dict) -> float:
		"""Calculate overall signal strength"""
		
		strength = 0.0
		
		# Order block strength
		if order_blocks:
			strength += sum(ob['strength'] for ob in order_blocks) * 0.3
		
		# Fair value gap strength
		if fair_value_gaps:
			strength += sum(fvg['strength'] for fvg in fair_value_gaps) * 0.2
		
		# Liquidity sweep strength
		if liquidity_sweeps:
			strength += sum(ls['strength'] for ls in liquidity_sweeps) * 0.2
		
		# Market structure strength
		if market_structure['trend'] != "sideways":
			strength += 0.2
		
		if market_structure['structure_break']:
			strength += 0.1
		
		return min(strength, 1.0)
	
	def _determine_action(self, order_blocks: List, fair_value_gaps: List, 
						liquidity_sweeps: List, market_structure: Dict) -> str:
		"""Determine trading action based on ICT analysis"""
		
		bullish_signals = 0
		bearish_signals = 0
		
		# Count bullish signals
		for ob in order_blocks:
			if ob['type'] == 'bullish':
				bullish_signals += 1
		
		for fvg in fair_value_gaps:
			if fvg['type'] == 'bullish':
				bullish_signals += 1
		
		for ls in liquidity_sweeps:
			if ls['type'] == 'low_sweep':
				bullish_signals += 1
		
		if market_structure['trend'] == 'uptrend':
			bullish_signals += 1
		
		if market_structure['structure_break'] == 'bullish_break':
			bullish_signals += 1
		
		# Count bearish signals
		for ob in order_blocks:
			if ob['type'] == 'bearish':
				bearish_signals += 1
		
		for fvg in fair_value_gaps:
			if fvg['type'] == 'bearish':
				bearish_signals += 1
		
		for ls in liquidity_sweeps:
			if ls['type'] == 'high_sweep':
				bearish_signals += 1
		
		if market_structure['trend'] == 'downtrend':
			bearish_signals += 1
		
		if market_structure['structure_break'] == 'bearish_break':
			bearish_signals += 1
		
		# Determine action
		if bullish_signals > bearish_signals + 1:
			return "BUY"
		elif bearish_signals > bullish_signals + 1:
			return "SELL"
		else:
			return "HOLD"
	
	def _calculate_levels(self, action: str, data: pd.DataFrame, 
						order_blocks: List, fair_value_gaps: List) -> Tuple[float, float]:
		"""Calculate stop loss and take profit levels"""
		
		current_price = data['close'].iloc[-1]
		
		if action == "BUY":
			# Find nearest support level
			support_levels = []
			for ob in order_blocks:
				if ob['type'] == 'bullish':
					support_levels.append(ob['low'])
			
			for fvg in fair_value_gaps:
				if fvg['type'] == 'bullish':
					support_levels.append(fvg['bottom'])
			
			stop_loss = min(support_levels) if support_levels else current_price * 0.995
			
			# Calculate take profit (risk-reward ratio)
			risk = current_price - stop_loss
			take_profit = current_price + (risk * self.parameters["risk_reward_min"])
		
		else:  # SELL
			# Find nearest resistance level
			resistance_levels = []
			for ob in order_blocks:
				if ob['type'] == 'bearish':
					resistance_levels.append(ob['high'])
			
			for fvg in fair_value_gaps:
				if fvg['type'] == 'bearish':
					resistance_levels.append(fvg['top'])
			
			stop_loss = max(resistance_levels) if resistance_levels else current_price * 1.005
			
			# Calculate take profit (risk-reward ratio)
			risk = stop_loss - current_price
			take_profit = current_price - (risk * self.parameters["risk_reward_min"])
		
		return stop_loss, take_profit
	
	def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float = 0.01) -> float:
		"""Calculate position size based on risk management"""
		
		risk_amount = account_balance * risk_per_trade
		risk_per_pip = abs(signal.entry_price - signal.stop_loss)
		
		# Assuming 1 pip = 0.0001 for major pairs
		pip_value = 10.0  # $10 per pip for standard lot
		position_size = risk_amount / (risk_per_pip * pip_value)
		
		return max(0.01, min(position_size, 1.0))  # Min 0.01, Max 1.0 lots
	
	def _calculate_order_block_strength(self, data: pd.DataFrame, index: int, ob_type: str) -> float:
		"""Calculate order block strength"""
		
		candle = data.iloc[index]
		body_size = abs(candle['close'] - candle['open'])
		total_range = candle['high'] - candle['low']
		
		if total_range == 0:
			return 0.0
		
		strength = body_size / total_range
		
		# Additional strength based on volume (if available)
		if 'volume' in data.columns:
			avg_volume = data['volume'].rolling(20).mean().iloc[index]
			if candle['volume'] > avg_volume * 1.5:
				strength += 0.2
		
		return min(strength, 1.0)
	
	def _calculate_fvg_strength(self, data: pd.DataFrame, index: int, fvg_type: str) -> float:
		"""Calculate Fair Value Gap strength"""
		
		prev_candle = data.iloc[index - 1]
		current_candle = data.iloc[index]
		next_candle = data.iloc[index + 1]
		
		if fvg_type == 'bullish':
			gap_size = next_candle['low'] - prev_candle['high']
		else:
			gap_size = prev_candle['low'] - next_candle['high']
		
		# Normalize gap size
		avg_range = data['high'].rolling(20).mean().iloc[index] - data['low'].rolling(20).mean().iloc[index]
		strength = gap_size / avg_range if avg_range > 0 else 0.0
		
		return min(strength, 1.0)
	
	def _calculate_sweep_strength(self, data: pd.DataFrame, index: int, sweep_type: str) -> float:
		"""Calculate liquidity sweep strength"""
		
		if sweep_type == 'high':
			recent_highs = data['high'].iloc[index-20:index].max()
			current_high = data['high'].iloc[index]
			sweep_size = current_high - recent_highs
		else:
			recent_lows = data['low'].iloc[index-20:index].min()
			current_low = data['low'].iloc[index]
			sweep_size = recent_lows - current_low
		
		# Normalize sweep size
		avg_range = data['high'].rolling(20).mean().iloc[index] - data['low'].rolling(20).mean().iloc[index]
		strength = sweep_size / avg_range if avg_range > 0 else 0.0
		
		return min(strength, 1.0)