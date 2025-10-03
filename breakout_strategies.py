import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5
from abc import ABC, abstractmethod
from ict_strategy import BaseStrategy, TradingSignal, SignalStrength


class BreakoutStrategy(BaseStrategy):
	"""
	Breakout Strategy
	Identifies and trades breakouts from consolidation patterns
	"""
	
	def __init__(self, symbol: str, timeframe: str = "H1"):
		super().__init__("Breakout Strategy", symbol, timeframe)
		self.parameters = {
			"consolidation_periods": 20,
			"breakout_threshold": 0.0005,  # 5 pips
			"volume_confirmation": True,
			"min_consolidation_duration": 10,
			"max_consolidation_duration": 50,
			"atr_period": 14,
			"atr_multiplier": 1.5,
			"min_breakout_strength": 0.7,
			"false_breakout_filter": True
		}
	
	def generate_signal(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], context: Dict = None) -> Optional[TradingSignal]:
		"""Generate breakout trading signal"""
		
		# Handle both single DataFrame and multi-timeframe data
		if isinstance(data, dict):
			# Use H1 data for breakout analysis
			if "H1" in data:
				analysis_data = data["H1"]
			else:
				# Use first available timeframe
				analysis_data = list(data.values())[0]
		else:
			analysis_data = data
		
		if len(analysis_data) < 100:
			return None
		
		# Identify consolidation patterns
		consolidation_patterns = self._identify_consolidation_patterns(analysis_data)
		
		if not consolidation_patterns:
			return None
		
		# Check for breakouts
		breakout_signals = self._detect_breakouts(analysis_data, consolidation_patterns)
		
		# Generate signal
		signal = self._generate_breakout_signal(analysis_data, breakout_signals)
		
		return signal
	
	def _identify_consolidation_patterns(self, data: pd.DataFrame) -> List[Dict]:
		"""Identify consolidation patterns in the data"""
		
		consolidation_patterns = []
		lookback = self.parameters["consolidation_periods"]
		
		for i in range(lookback, len(data) - 5):
			# Check for consolidation in the last lookback periods
			consolidation_data = data.iloc[i-lookback:i]
			
			# Calculate consolidation metrics
			consolidation_metrics = self._calculate_consolidation_metrics(consolidation_data)
			
			# Check if it's a valid consolidation
			if self._is_valid_consolidation(consolidation_metrics):
				consolidation_patterns.append({
					"start_index": i - lookback,
					"end_index": i,
					"metrics": consolidation_metrics,
					"support": consolidation_metrics["support_level"],
					"resistance": consolidation_metrics["resistance_level"],
					"range_size": consolidation_metrics["range_size"]
				})
		
		return consolidation_patterns[-3:]  # Return last 3 patterns
	
	def _calculate_consolidation_metrics(self, data: pd.DataFrame) -> Dict:
		"""Calculate consolidation pattern metrics"""
		
		# Price range analysis
		highs = data['high']
		lows = data['low']
		closes = data['close']
		
		# Support and resistance levels
		support_level = lows.min()
		resistance_level = highs.max()
		range_size = resistance_level - support_level
		
		# Range percentage
		avg_price = closes.mean()
		range_percentage = (range_size / avg_price) * 100
		
		# Volatility analysis
		price_changes = closes.pct_change().dropna()
		volatility = price_changes.std()
		
		# Trend analysis within consolidation
		trend_slope = self._calculate_trend_slope(closes)
		
		# Volume analysis (if available)
		volume_analysis = {}
		if 'volume' in data.columns:
			volume_analysis = self._analyze_volume_pattern(data['volume'])
		
		return {
			"support_level": support_level,
			"resistance_level": resistance_level,
			"range_size": range_size,
			"range_percentage": range_percentage,
			"volatility": volatility,
			"trend_slope": trend_slope,
			"volume_analysis": volume_analysis,
			"duration": len(data)
		}
	
	def _calculate_trend_slope(self, prices: pd.Series) -> float:
		"""Calculate trend slope within consolidation"""
		
		if len(prices) < 2:
			return 0.0
		
		x = np.arange(len(prices))
		y = prices.values
		
		# Linear regression slope
		slope = np.polyfit(x, y, 1)[0]
		
		# Normalize slope
		normalized_slope = slope / prices.mean()
		
		return normalized_slope
	
	def _analyze_volume_pattern(self, volume: pd.Series) -> Dict:
		"""Analyze volume pattern during consolidation"""
		
		avg_volume = volume.mean()
		volume_trend = self._calculate_trend_slope(volume)
		
		# Volume confirmation
		recent_volume = volume.tail(5).mean()
		volume_confirmation = recent_volume > avg_volume * 1.2
		
		return {
			"avg_volume": avg_volume,
			"volume_trend": volume_trend,
			"volume_confirmation": volume_confirmation,
			"recent_volume": recent_volume
		}
	
	def _is_valid_consolidation(self, metrics: Dict) -> bool:
		"""Check if consolidation pattern is valid"""
		
		# Range size check
		if metrics["range_percentage"] > 2.0:  # Too wide
			return False
		
		if metrics["range_percentage"] < 0.1:  # Too narrow
			return False
		
		# Duration check
		if (metrics["duration"] < self.parameters["min_consolidation_duration"] or
			metrics["duration"] > self.parameters["max_consolidation_duration"]):
			return False
		
		# Trend slope check (should be relatively flat)
		if abs(metrics["trend_slope"]) > 0.001:  # Too trending
			return False
		
		return True
	
	def _detect_breakouts(self, data: pd.DataFrame, consolidation_patterns: List[Dict]) -> List[Dict]:
		"""Detect breakout signals from consolidation patterns"""
		
		breakout_signals = []
		current_price = data['close'].iloc[-1]
		current_high = data['high'].iloc[-1]
		current_low = data['low'].iloc[-1]
		
		for pattern in consolidation_patterns:
			support = pattern["support"]
			resistance = pattern["resistance"]
			range_size = pattern["range_size"]
			
			# Bullish breakout
			if current_high > resistance + self.parameters["breakout_threshold"]:
				breakout_strength = self._calculate_breakout_strength(
					data, pattern, "bullish", current_price, resistance
				)
				
				if breakout_strength > self.parameters["min_breakout_strength"]:
					breakout_signals.append({
						"type": "bullish",
						"pattern": pattern,
						"breakout_price": resistance,
						"current_price": current_price,
						"strength": breakout_strength,
						"range_size": range_size
					})
			
			# Bearish breakout
			elif current_low < support - self.parameters["breakout_threshold"]:
				breakout_strength = self._calculate_breakout_strength(
					data, pattern, "bearish", current_price, support
				)
				
				if breakout_strength > self.parameters["min_breakout_strength"]:
					breakout_signals.append({
						"type": "bearish",
						"pattern": pattern,
						"breakout_price": support,
						"current_price": current_price,
						"strength": breakout_strength,
						"range_size": range_size
					})
		
		return breakout_signals
	
	def _calculate_breakout_strength(self, data: pd.DataFrame, pattern: Dict, 
								   breakout_type: str, current_price: float, 
								   breakout_level: float) -> float:
		"""Calculate breakout strength"""
		
		strength = 0.0
		
		# Price momentum
		if breakout_type == "bullish":
			momentum = (current_price - breakout_level) / breakout_level
		else:
			momentum = (breakout_level - current_price) / breakout_level
		
		strength += min(momentum * 100, 0.4)  # Max 0.4 for momentum
		
		# Volume confirmation
		if self.parameters["volume_confirmation"] and 'volume' in data.columns:
			recent_volume = data['volume'].tail(3).mean()
			avg_volume = data['volume'].rolling(20).mean().iloc[-1]
			
			if recent_volume > avg_volume * 1.5:
				strength += 0.3
		
		# False breakout filter
		if self.parameters["false_breakout_filter"]:
			if self._is_false_breakout(data, pattern, breakout_type):
				strength -= 0.3
		
		# Pattern duration bonus
		duration_bonus = min(pattern["metrics"]["duration"] / 30.0, 0.2)
		strength += duration_bonus
		
		return max(0.0, min(strength, 1.0))
	
	def _is_false_breakout(self, data: pd.DataFrame, pattern: Dict, breakout_type: str) -> bool:
		"""Check for false breakout patterns"""
		
		# Look at last few candles for false breakout
		recent_data = data.tail(5)
		
		if breakout_type == "bullish":
			# Check if price fell back below resistance
			return recent_data['close'].min() < pattern["resistance"]
		else:
			# Check if price rose back above support
			return recent_data['close'].max() > pattern["support"]
	
	def _generate_breakout_signal(self, data: pd.DataFrame, breakout_signals: List[Dict]) -> Optional[TradingSignal]:
		"""Generate breakout trading signal"""
		
		if not breakout_signals:
			return None
		
		# Select strongest breakout signal
		best_signal = max(breakout_signals, key=lambda x: x["strength"])
		
		current_price = data['close'].iloc[-1]
		current_time = data.index[-1]
		
		action = "BUY" if best_signal["type"] == "bullish" else "SELL"
		confidence = best_signal["strength"]
		
		# Calculate stop loss and take profit
		stop_loss, take_profit = self._calculate_breakout_levels(
			action, current_price, best_signal, data
		)
		
		# Calculate risk-reward ratio
		risk = abs(current_price - stop_loss)
		reward = abs(take_profit - current_price)
		risk_reward_ratio = reward / risk if risk > 0 else 0
		
		if risk_reward_ratio < 1.5:
			return None
		
		# Create signal
		signal = TradingSignal(
			symbol=self.symbol,
			action=action,
			entry_price=current_price,
			stop_loss=stop_loss,
			take_profit=take_profit,
			volume=0.01,  # Will be calculated later
			confidence=confidence,
			signal_strength=SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MEDIUM,
			strategy_name=self.name,
			timeframe=self.timeframe,
			timestamp=current_time,
			reason=f"Breakout: {action} signal from {best_signal['type']} breakout",
			risk_reward_ratio=risk_reward_ratio,
			market_regime=context.get("market_regime", "normal") if context else "normal",
			additional_info={
				"breakout_signal": best_signal,
				"pattern_info": best_signal["pattern"]
			}
		)
		
		return signal
	
	def _calculate_breakout_levels(self, action: str, current_price: float, 
								 breakout_signal: Dict, data: pd.DataFrame) -> Tuple[float, float]:
		"""Calculate stop loss and take profit levels for breakout"""
		
		pattern = breakout_signal["pattern"]
		range_size = pattern["range_size"]
		
		if action == "BUY":
			# Stop loss below the broken resistance (now support)
			stop_loss = pattern["resistance"] - (range_size * 0.1)
			
			# Take profit at 1.5x the range size
			take_profit = current_price + (range_size * 1.5)
		
		else:  # SELL
			# Stop loss above the broken support (now resistance)
			stop_loss = pattern["support"] + (range_size * 0.1)
			
			# Take profit at 1.5x the range size
			take_profit = current_price - (range_size * 1.5)
		
		return stop_loss, take_profit
	
	def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float = 0.01) -> float:
		"""Calculate position size based on risk management"""
		
		risk_amount = account_balance * risk_per_trade
		risk_per_pip = abs(signal.entry_price - signal.stop_loss)
		
		# Assuming 1 pip = 0.0001 for major pairs
		pip_value = 10.0  # $10 per pip for standard lot
		position_size = risk_amount / (risk_per_pip * pip_value)
		
		return max(0.01, min(position_size, 1.0))  # Min 0.01, Max 1.0 lots


class RangeTradingStrategy(BaseStrategy):
	"""
	Range Trading Strategy
	Trades within established ranges, buying at support and selling at resistance
	"""
	
	def __init__(self, symbol: str, timeframe: str = "H1"):
		super().__init__("Range Trading Strategy", symbol, timeframe)
		self.parameters = {
			"range_periods": 50,
			"support_resistance_lookback": 20,
			"min_range_size": 0.001,  # 10 pips
			"max_range_size": 0.005,  # 50 pips
			"bounce_threshold": 0.0002,  # 2 pips
			"min_bounces": 2,
			"rsi_period": 14,
			"rsi_overbought": 70,
			"rsi_oversold": 30,
			"min_range_duration": 20,
			"max_range_duration": 100
		}
	
	def generate_signal(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], context: Dict = None) -> Optional[TradingSignal]:
		"""Generate range trading signal"""
		
		# Handle both single DataFrame and multi-timeframe data
		if isinstance(data, dict):
			# Use H1 data for range trading analysis
			if "H1" in data:
				analysis_data = data["H1"]
			else:
				# Use first available timeframe
				analysis_data = list(data.values())[0]
		else:
			analysis_data = data
		
		if len(analysis_data) < 100:
			return None
		
		# Identify range patterns
		range_patterns = self._identify_range_patterns(analysis_data)
		
		if not range_patterns:
			return None
		
		# Check for range trading opportunities
		range_signals = self._detect_range_signals(analysis_data, range_patterns)
		
		# Generate signal
		signal = self._generate_range_signal(analysis_data, range_signals)
		
		return signal
	
	def _identify_range_patterns(self, data: pd.DataFrame) -> List[Dict]:
		"""Identify range trading patterns"""
		
		range_patterns = []
		lookback = self.parameters["range_periods"]
		
		for i in range(lookback, len(data) - 10):
			# Analyze range in the last lookback periods
			range_data = data.iloc[i-lookback:i]
			
			# Calculate range metrics
			range_metrics = self._calculate_range_metrics(range_data)
			
			# Check if it's a valid range
			if self._is_valid_range(range_metrics):
				range_patterns.append({
					"start_index": i - lookback,
					"end_index": i,
					"metrics": range_metrics,
					"support": range_metrics["support_level"],
					"resistance": range_metrics["resistance_level"],
					"range_size": range_metrics["range_size"]
				})
		
		return range_patterns[-2:]  # Return last 2 patterns
	
	def _calculate_range_metrics(self, data: pd.DataFrame) -> Dict:
		"""Calculate range pattern metrics"""
		
		# Price levels
		highs = data['high']
		lows = data['low']
		closes = data['close']
		
		# Support and resistance levels
		support_level = lows.min()
		resistance_level = highs.max()
		range_size = resistance_level - support_level
		
		# Range analysis
		avg_price = closes.mean()
		range_percentage = (range_size / avg_price) * 100
		
		# Bounce analysis
		bounce_analysis = self._analyze_bounces(data, support_level, resistance_level)
		
		# Volume analysis (if available)
		volume_analysis = {}
		if 'volume' in data.columns:
			volume_analysis = self._analyze_range_volume(data['volume'])
		
		# Trend analysis within range
		trend_slope = self._calculate_trend_slope(closes)
		
		return {
			"support_level": support_level,
			"resistance_level": resistance_level,
			"range_size": range_size,
			"range_percentage": range_percentage,
			"bounce_analysis": bounce_analysis,
			"volume_analysis": volume_analysis,
			"trend_slope": trend_slope,
			"duration": len(data)
		}
	
	def _calculate_trend_slope(self, prices: pd.Series) -> float:
		"""Calculate trend slope within range"""
		
		if len(prices) < 2:
			return 0.0
		
		x = np.arange(len(prices))
		y = prices.values
		
		# Linear regression slope
		slope = np.polyfit(x, y, 1)[0]
		
		# Normalize slope
		normalized_slope = slope / prices.mean()
		
		return normalized_slope
	
	def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
		"""Calculate RSI indicator"""
		
		delta = prices.diff()
		gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
		loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
		
		rs = gain / loss
		rsi = 100 - (100 / (1 + rs))
		
		return rsi
	
	def _analyze_bounces(self, data: pd.DataFrame, support: float, resistance: float) -> Dict:
		"""Analyze bounces off support and resistance"""
		
		threshold = self.parameters["bounce_threshold"]
		
		# Count bounces off support
		support_bounces = 0
		for i in range(1, len(data)):
			if (data['low'].iloc[i] <= support + threshold and
				data['close'].iloc[i] > support + threshold):
				support_bounces += 1
		
		# Count bounces off resistance
		resistance_bounces = 0
		for i in range(1, len(data)):
			if (data['high'].iloc[i] >= resistance - threshold and
				data['close'].iloc[i] < resistance - threshold):
				resistance_bounces += 1
		
		# Calculate bounce strength
		support_bounce_strength = support_bounces / len(data)
		resistance_bounce_strength = resistance_bounces / len(data)
		
		return {
			"support_bounces": support_bounces,
			"resistance_bounces": resistance_bounces,
			"support_bounce_strength": support_bounce_strength,
			"resistance_bounce_strength": resistance_bounce_strength,
			"total_bounces": support_bounces + resistance_bounces
		}
	
	def _analyze_range_volume(self, volume: pd.Series) -> Dict:
		"""Analyze volume patterns within range"""
		
		avg_volume = volume.mean()
		volume_at_support = volume[volume.index.isin(volume.nsmallest(5).index)].mean()
		volume_at_resistance = volume[volume.index.isin(volume.nlargest(5).index)].mean()
		
		return {
			"avg_volume": avg_volume,
			"volume_at_support": volume_at_support,
			"volume_at_resistance": volume_at_resistance,
			"volume_confirmation": volume_at_support > avg_volume and volume_at_resistance > avg_volume
		}
	
	def _is_valid_range(self, metrics: Dict) -> bool:
		"""Check if range pattern is valid"""
		
		# Range size check
		if (metrics["range_percentage"] < self.parameters["min_range_size"] * 100 or
			metrics["range_percentage"] > self.parameters["max_range_size"] * 100):
			return False
		
		# Duration check
		if (metrics["duration"] < self.parameters["min_range_duration"] or
			metrics["duration"] > self.parameters["max_range_duration"]):
			return False
		
		# Bounce check
		if metrics["bounce_analysis"]["total_bounces"] < self.parameters["min_bounces"]:
			return False
		
		# Trend check (should be relatively flat)
		if abs(metrics["trend_slope"]) > 0.0005:
			return False
		
		return True
	
	def _detect_range_signals(self, data: pd.DataFrame, range_patterns: List[Dict]) -> List[Dict]:
		"""Detect range trading signals"""
		
		range_signals = []
		current_price = data['close'].iloc[-1]
		current_low = data['low'].iloc[-1]
		current_high = data['high'].iloc[-1]
		
		# Calculate RSI for confirmation
		rsi = self._calculate_rsi(data['close'], self.parameters["rsi_period"]).iloc[-1]
		
		for pattern in range_patterns:
			support = pattern["support"]
			resistance = pattern["resistance"]
			range_size = pattern["range_size"]
			
			# Buy signal near support
			if (current_low <= support + self.parameters["bounce_threshold"] and
				rsi < self.parameters["rsi_oversold"]):
				
				signal_strength = self._calculate_range_signal_strength(
					data, pattern, "buy", current_price, support
				)
				
				if signal_strength > 0.6:
					range_signals.append({
						"type": "buy",
						"pattern": pattern,
						"entry_price": current_price,
						"target_price": resistance,
						"strength": signal_strength,
						"rsi": rsi
					})
			
			# Sell signal near resistance
			elif (current_high >= resistance - self.parameters["bounce_threshold"] and
				  rsi > self.parameters["rsi_overbought"]):
				
				signal_strength = self._calculate_range_signal_strength(
					data, pattern, "sell", current_price, resistance
				)
				
				if signal_strength > 0.6:
					range_signals.append({
						"type": "sell",
						"pattern": pattern,
						"entry_price": current_price,
						"target_price": support,
						"strength": signal_strength,
						"rsi": rsi
					})
		
		return range_signals
	
	def _calculate_range_signal_strength(self, data: pd.DataFrame, pattern: Dict, 
									   signal_type: str, current_price: float, 
									   level: float) -> float:
		"""Calculate range signal strength"""
		
		strength = 0.0
		
		# Distance from level
		if signal_type == "buy":
			distance = abs(current_price - level) / level
		else:
			distance = abs(current_price - level) / level
		
		strength += max(0, 0.3 - distance * 100)  # Closer to level = stronger
		
		# Bounce history
		bounce_analysis = pattern["metrics"]["bounce_analysis"]
		if signal_type == "buy":
			strength += bounce_analysis["support_bounce_strength"] * 0.3
		else:
			strength += bounce_analysis["resistance_bounce_strength"] * 0.3
		
		# Volume confirmation
		if pattern["metrics"]["volume_analysis"].get("volume_confirmation", False):
			strength += 0.2
		
		# Range duration bonus
		duration_bonus = min(pattern["metrics"]["duration"] / 50.0, 0.2)
		strength += duration_bonus
		
		return max(0.0, min(strength, 1.0))
	
	def _generate_range_signal(self, data: pd.DataFrame, range_signals: List[Dict]) -> Optional[TradingSignal]:
		"""Generate range trading signal"""
		
		if not range_signals:
			return None
		
		# Select strongest signal
		best_signal = max(range_signals, key=lambda x: x["strength"])
		
		current_price = data['close'].iloc[-1]
		current_time = data.index[-1]
		
		action = "BUY" if best_signal["type"] == "buy" else "SELL"
		confidence = best_signal["strength"]
		
		# Calculate stop loss and take profit
		stop_loss, take_profit = self._calculate_range_levels(
			action, current_price, best_signal
		)
		
		# Calculate risk-reward ratio
		risk = abs(current_price - stop_loss)
		reward = abs(take_profit - current_price)
		risk_reward_ratio = reward / risk if risk > 0 else 0
		
		if risk_reward_ratio < 1.0:  # Range trading can have lower R:R
			return None
		
		# Create signal
		signal = TradingSignal(
			symbol=self.symbol,
			action=action,
			entry_price=current_price,
			stop_loss=stop_loss,
			take_profit=take_profit,
			volume=0.01,  # Will be calculated later
			confidence=confidence,
			signal_strength=SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MEDIUM,
			strategy_name=self.name,
			timeframe=self.timeframe,
			timestamp=current_time,
			reason=f"Range Trading: {action} signal at {best_signal['type']} level",
			risk_reward_ratio=risk_reward_ratio,
			market_regime=context.get("market_regime", "normal") if context else "normal",
			additional_info={
				"range_signal": best_signal,
				"pattern_info": best_signal["pattern"]
			}
		)
		
		return signal
	
	def _calculate_range_levels(self, action: str, current_price: float, 
							  range_signal: Dict) -> Tuple[float, float]:
		"""Calculate stop loss and take profit levels for range trading"""
		
		pattern = range_signal["pattern"]
		support = pattern["support"]
		resistance = pattern["resistance"]
		range_size = pattern["range_size"]
		
		if action == "BUY":
			# Stop loss below support
			stop_loss = support - (range_size * 0.1)
			
			# Take profit at resistance
			take_profit = resistance - (range_size * 0.1)
		
		else:  # SELL
			# Stop loss above resistance
			stop_loss = resistance + (range_size * 0.1)
			
			# Take profit at support
			take_profit = support + (range_size * 0.1)
		
		return stop_loss, take_profit
	
	def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float = 0.01) -> float:
		"""Calculate position size based on risk management"""
		
		risk_amount = account_balance * risk_per_trade
		risk_per_pip = abs(signal.entry_price - signal.stop_loss)
		
		# Assuming 1 pip = 0.0001 for major pairs
		pip_value = 10.0  # $10 per pip for standard lot
		position_size = risk_amount / (risk_per_pip * pip_value)
		
		return max(0.01, min(position_size, 0.8))  # Moderate max size for range trading