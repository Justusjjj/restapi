import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5
from abc import ABC, abstractmethod
from ict_strategy import BaseStrategy, TradingSignal, SignalStrength
from ict_strategy import ICTStrategy
from momentum_strategies import MomentumStrategy, MeanReversionStrategy
from breakout_strategies import BreakoutStrategy, RangeTradingStrategy


class MultiTimeframeStrategy(BaseStrategy):
	"""
	Multi-Timeframe Strategy Framework
	Combines signals from multiple timeframes for higher accuracy
	"""
	
	def __init__(self, symbol: str, primary_timeframe: str = "H1"):
		super().__init__("Multi-Timeframe Strategy", symbol, primary_timeframe)
		
		# Define timeframes hierarchy
		self.timeframes = {
			"HTF": ["H4", "D1"],      # Higher Timeframes for bias
			"MTF": ["H1"],            # Medium Timeframe for structure
			"LTF": ["M15", "M5"]      # Lower Timeframes for entry
		}
		
		# Initialize strategies for each timeframe
		self.strategies = {
			"H4": ICTStrategy(symbol, "H4"),
			"D1": ICTStrategy(symbol, "D1"),
			"H1": MomentumStrategy(symbol, "H1"),
			"M15": MeanReversionStrategy(symbol, "M15"),
			"M5": RangeTradingStrategy(symbol, "M5")
		}
		
		self.parameters = {
			"htf_weight": 0.4,        # Higher timeframe weight
			"mtf_weight": 0.4,        # Medium timeframe weight
			"ltf_weight": 0.2,        # Lower timeframe weight
			"min_htf_alignment": 0.6, # Minimum HTF alignment required
			"min_mtf_strength": 0.5,  # Minimum MTF signal strength
			"ltf_confirmation": True,  # Require LTF confirmation
			"bias_filter": True,      # Use HTF bias filter
			"structure_filter": True  # Use MTF structure filter
		}
		
		# Signal history for analysis
		self.signal_history = []
		self.performance_metrics = {}
	
	def generate_signal(self, data: Dict[str, pd.DataFrame], context: Dict = None) -> Optional[TradingSignal]:
		"""Generate multi-timeframe trading signal"""
		
		# Validate data availability
		if not self._validate_data(data):
			return None
		
		# Generate signals for each timeframe
		timeframe_signals = self._generate_timeframe_signals(data, context)
		
		# Analyze signal alignment
		signal_alignment = self._analyze_signal_alignment(timeframe_signals)
		
		# Apply filters
		if not self._apply_filters(signal_alignment):
			return None
		
		# Generate final signal
		final_signal = self._generate_final_signal(data, timeframe_signals, signal_alignment)
		
		# Store signal for analysis
		if final_signal:
			self.signal_history.append(final_signal)
			self._update_performance_metrics(final_signal)
		
		return final_signal
	
	def _validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
		"""Validate data availability for all timeframes"""
		
		required_timeframes = ["H4", "H1", "M15"]
		
		for tf in required_timeframes:
			if tf not in data or len(data[tf]) < 50:
				return False
		
		return True
	
	def _generate_timeframe_signals(self, data: Dict[str, pd.DataFrame], 
								  context: Dict = None) -> Dict[str, Optional[TradingSignal]]:
		"""Generate signals for each timeframe"""
		
		signals = {}
		
		for timeframe, strategy in self.strategies.items():
			if timeframe in data and len(data[timeframe]) >= 50:
				try:
					signal = strategy.generate_signal(data[timeframe], context)
					signals[timeframe] = signal
				except Exception as e:
					print(f"Error generating signal for {timeframe}: {e}")
					signals[timeframe] = None
			else:
				signals[timeframe] = None
		
		return signals
	
	def _analyze_signal_alignment(self, signals: Dict[str, Optional[TradingSignal]]) -> Dict:
		"""Analyze alignment between timeframe signals"""
		
		alignment = {
			"htf_signals": [],
			"mtf_signals": [],
			"ltf_signals": [],
			"overall_alignment": 0.0,
			"bias_direction": "neutral",
			"structure_direction": "neutral",
			"entry_direction": "neutral",
			"confidence_scores": {},
			"conflicts": []
		}
		
		# Categorize signals by timeframe
		for tf, signal in signals.items():
			if signal:
				if tf in self.timeframes["HTF"]:
					alignment["htf_signals"].append(signal)
				elif tf in self.timeframes["MTF"]:
					alignment["mtf_signals"].append(signal)
				elif tf in self.timeframes["LTF"]:
					alignment["ltf_signals"].append(signal)
				
				alignment["confidence_scores"][tf] = signal.confidence
		
		# Determine bias direction (HTF)
		alignment["bias_direction"] = self._determine_bias_direction(alignment["htf_signals"])
		
		# Determine structure direction (MTF)
		alignment["structure_direction"] = self._determine_structure_direction(alignment["mtf_signals"])
		
		# Determine entry direction (LTF)
		alignment["entry_direction"] = self._determine_entry_direction(alignment["ltf_signals"])
		
		# Calculate overall alignment
		alignment["overall_alignment"] = self._calculate_overall_alignment(alignment)
		
		# Identify conflicts
		alignment["conflicts"] = self._identify_conflicts(alignment)
		
		return alignment
	
	def _determine_bias_direction(self, htf_signals: List[TradingSignal]) -> str:
		"""Determine higher timeframe bias direction"""
		
		if not htf_signals:
			return "neutral"
		
		buy_signals = [s for s in htf_signals if s.action == "BUY"]
		sell_signals = [s for s in htf_signals if s.action == "SELL"]
		
		buy_strength = sum(s.confidence for s in buy_signals)
		sell_strength = sum(s.confidence for s in sell_signals)
		
		if buy_strength > sell_strength * 1.2:
			return "bullish"
		elif sell_strength > buy_strength * 1.2:
			return "bearish"
		else:
			return "neutral"
	
	def _determine_structure_direction(self, mtf_signals: List[TradingSignal]) -> str:
		"""Determine medium timeframe structure direction"""
		
		if not mtf_signals:
			return "neutral"
		
		# Use the strongest MTF signal
		strongest_signal = max(mtf_signals, key=lambda x: x.confidence)
		
		return strongest_signal.action.lower() if strongest_signal.action != "HOLD" else "neutral"
	
	def _determine_entry_direction(self, ltf_signals: List[TradingSignal]) -> str:
		"""Determine lower timeframe entry direction"""
		
		if not ltf_signals:
			return "neutral"
		
		# Use the most recent LTF signal
		most_recent_signal = max(ltf_signals, key=lambda x: x.timestamp)
		
		return most_recent_signal.action.lower() if most_recent_signal.action != "HOLD" else "neutral"
	
	def _calculate_overall_alignment(self, alignment: Dict) -> float:
		"""Calculate overall signal alignment score"""
		
		score = 0.0
		
		# HTF bias alignment
		if alignment["bias_direction"] != "neutral":
			htf_confidence = np.mean([s.confidence for s in alignment["htf_signals"]]) if alignment["htf_signals"] else 0.0
			score += htf_confidence * self.parameters["htf_weight"]
		
		# MTF structure alignment
		if alignment["structure_direction"] != "neutral":
			mtf_confidence = np.mean([s.confidence for s in alignment["mtf_signals"]]) if alignment["mtf_signals"] else 0.0
			score += mtf_confidence * self.parameters["mtf_weight"]
		
		# LTF entry alignment
		if alignment["entry_direction"] != "neutral":
			ltf_confidence = np.mean([s.confidence for s in alignment["ltf_signals"]]) if alignment["ltf_signals"] else 0.0
			score += ltf_confidence * self.parameters["ltf_weight"]
		
		# Bonus for alignment between timeframes
		if (alignment["bias_direction"] == alignment["structure_direction"] == alignment["entry_direction"] and
			alignment["bias_direction"] != "neutral"):
			score += 0.2
		
		return min(score, 1.0)
	
	def _identify_conflicts(self, alignment: Dict) -> List[str]:
		"""Identify conflicts between timeframe signals"""
		
		conflicts = []
		
		# Check HTF vs MTF conflict
		if (alignment["bias_direction"] != "neutral" and 
			alignment["structure_direction"] != "neutral" and
			alignment["bias_direction"] != alignment["structure_direction"]):
			conflicts.append("HTF-MTF conflict")
		
		# Check MTF vs LTF conflict
		if (alignment["structure_direction"] != "neutral" and 
			alignment["entry_direction"] != "neutral" and
			alignment["structure_direction"] != alignment["entry_direction"]):
			conflicts.append("MTF-LTF conflict")
		
		# Check HTF vs LTF conflict
		if (alignment["bias_direction"] != "neutral" and 
			alignment["entry_direction"] != "neutral" and
			alignment["bias_direction"] != alignment["entry_direction"]):
			conflicts.append("HTF-LTF conflict")
		
		return conflicts
	
	def _apply_filters(self, alignment: Dict) -> bool:
		"""Apply multi-timeframe filters"""
		
		# HTF bias filter
		if self.parameters["bias_filter"]:
			if alignment["bias_direction"] == "neutral":
				return False
		
		# MTF structure filter
		if self.parameters["structure_filter"]:
			if alignment["structure_direction"] == "neutral":
				return False
		
		# Minimum HTF alignment
		if alignment["overall_alignment"] < self.parameters["min_htf_alignment"]:
			return False
		
		# LTF confirmation filter
		if self.parameters["ltf_confirmation"]:
			if alignment["entry_direction"] == "neutral":
				return False
		
		# Conflict filter
		if len(alignment["conflicts"]) > 1:
			return False
		
		return True
	
	def _generate_final_signal(self, data: Dict[str, pd.DataFrame], 
							 timeframe_signals: Dict[str, Optional[TradingSignal]], 
							 alignment: Dict) -> Optional[TradingSignal]:
		"""Generate final multi-timeframe signal"""
		
		# Use primary timeframe data for price levels
		primary_data = data[self.timeframe]
		current_price = primary_data['close'].iloc[-1]
		current_time = primary_data.index[-1]
		
		# Determine final action
		final_action = self._determine_final_action(alignment)
		
		if final_action == "HOLD":
			return None
		
		# Calculate stop loss and take profit using multiple timeframes
		stop_loss, take_profit = self._calculate_mtf_levels(
			final_action, current_price, data, timeframe_signals
		)
		
		# Calculate risk-reward ratio
		risk = abs(current_price - stop_loss)
		reward = abs(take_profit - current_price)
		risk_reward_ratio = reward / risk if risk > 0 else 0
		
		if risk_reward_ratio < 1.5:
			return None
		
		# Create final signal
		signal = TradingSignal(
			symbol=self.symbol,
			action=final_action,
			entry_price=current_price,
			stop_loss=stop_loss,
			take_profit=take_profit,
			volume=0.01,  # Will be calculated later
			confidence=alignment["overall_alignment"],
			signal_strength=SignalStrength.STRONG if alignment["overall_alignment"] > 0.8 else SignalStrength.MEDIUM,
			strategy_name=self.name,
			timeframe=self.timeframe,
			timestamp=current_time,
			reason=f"Multi-TF: {final_action} signal with {alignment['overall_alignment']:.2f} alignment",
			risk_reward_ratio=risk_reward_ratio,
			market_regime=context.get("market_regime", "normal") if context else "normal",
			additional_info={
				"timeframe_signals": {tf: s.action if s else None for tf, s in timeframe_signals.items()},
				"alignment": alignment,
				"htf_bias": alignment["bias_direction"],
				"mtf_structure": alignment["structure_direction"],
				"ltf_entry": alignment["entry_direction"]
			}
		)
		
		return signal
	
	def _determine_final_action(self, alignment: Dict) -> str:
		"""Determine final trading action based on alignment"""
		
		# If all timeframes align
		if (alignment["bias_direction"] == alignment["structure_direction"] == alignment["entry_direction"] and
			alignment["bias_direction"] != "neutral"):
			return alignment["bias_direction"].upper()
		
		# If HTF and MTF align (strong bias)
		if (alignment["bias_direction"] == alignment["structure_direction"] and
			alignment["bias_direction"] != "neutral"):
			return alignment["bias_direction"].upper()
		
		# If MTF and LTF align (structure + entry)
		if (alignment["structure_direction"] == alignment["entry_direction"] and
			alignment["structure_direction"] != "neutral"):
			return alignment["structure_direction"].upper()
		
		return "HOLD"
	
	def _calculate_mtf_levels(self, action: str, current_price: float, 
							data: Dict[str, pd.DataFrame], 
							timeframe_signals: Dict[str, Optional[TradingSignal]]) -> Tuple[float, float]:
		"""Calculate stop loss and take profit using multiple timeframes"""
		
		# Use HTF for major levels
		htf_data = data.get("H4", data.get("D1"))
		if htf_data is not None and len(htf_data) > 0:
			htf_support = htf_data['low'].tail(50).min()
			htf_resistance = htf_data['high'].tail(50).max()
		else:
			htf_support = current_price * 0.99
			htf_resistance = current_price * 1.01
		
		# Use MTF for precise levels
		mtf_data = data.get("H1")
		if mtf_data is not None and len(mtf_data) > 0:
			mtf_support = mtf_data['low'].tail(20).min()
			mtf_resistance = mtf_data['high'].tail(20).max()
		else:
			mtf_support = current_price * 0.995
			mtf_resistance = current_price * 1.005
		
		if action == "BUY":
			# Stop loss below both HTF and MTF support
			stop_loss = min(htf_support, mtf_support) * 0.999
			
			# Take profit at HTF resistance
			take_profit = htf_resistance * 0.999
		
		else:  # SELL
			# Stop loss above both HTF and MTF resistance
			stop_loss = max(htf_resistance, mtf_resistance) * 1.001
			
			# Take profit at HTF support
			take_profit = htf_support * 1.001
		
		return stop_loss, take_profit
	
	def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float = 0.01) -> float:
		"""Calculate position size based on multi-timeframe risk management"""
		
		# Use conservative position sizing for multi-timeframe signals
		risk_amount = account_balance * risk_per_trade
		risk_per_pip = abs(signal.entry_price - signal.stop_loss)
		
		# Assuming 1 pip = 0.0001 for major pairs
		pip_value = 10.0  # $10 per pip for standard lot
		position_size = risk_amount / (risk_per_pip * pip_value)
		
		# Adjust based on signal confidence
		confidence_multiplier = signal.confidence
		position_size *= confidence_multiplier
		
		return max(0.01, min(position_size, 1.0))  # Min 0.01, Max 1.0 lots
	
	def _update_performance_metrics(self, signal: TradingSignal):
		"""Update performance metrics for multi-timeframe strategy"""
		
		if "total_signals" not in self.performance_metrics:
			self.performance_metrics = {
				"total_signals": 0,
				"buy_signals": 0,
				"sell_signals": 0,
				"avg_confidence": 0.0,
				"avg_risk_reward": 0.0,
				"htf_alignment_rate": 0.0,
				"mtf_alignment_rate": 0.0,
				"ltf_alignment_rate": 0.0
			}
		
		self.performance_metrics["total_signals"] += 1
		
		if signal.action == "BUY":
			self.performance_metrics["buy_signals"] += 1
		elif signal.action == "SELL":
			self.performance_metrics["sell_signals"] += 1
		
		# Update averages
		total = self.performance_metrics["total_signals"]
		self.performance_metrics["avg_confidence"] = (
			(self.performance_metrics["avg_confidence"] * (total - 1) + signal.confidence) / total
		)
		self.performance_metrics["avg_risk_reward"] = (
			(self.performance_metrics["avg_risk_reward"] * (total - 1) + signal.risk_reward_ratio) / total
		)
		
		# Update alignment rates
		additional_info = signal.additional_info
		if additional_info:
			alignment = additional_info.get("alignment", {})
			
			if alignment.get("bias_direction") != "neutral":
				self.performance_metrics["htf_alignment_rate"] += 1
			if alignment.get("structure_direction") != "neutral":
				self.performance_metrics["mtf_alignment_rate"] += 1
			if alignment.get("entry_direction") != "neutral":
				self.performance_metrics["ltf_alignment_rate"] += 1
	
	def get_strategy_summary(self) -> Dict:
		"""Get comprehensive strategy summary"""
		
		return {
			"strategy_name": self.name,
			"symbol": self.symbol,
			"primary_timeframe": self.timeframe,
			"timeframes": self.timeframes,
			"parameters": self.parameters,
			"performance_metrics": self.performance_metrics,
			"total_signals_generated": len(self.signal_history),
			"last_signal": self.signal_history[-1] if self.signal_history else None
		}


class StrategyManager:
	"""
	Strategy Manager for coordinating multiple strategies
	"""
	
	def __init__(self, symbol: str):
		self.symbol = symbol
		self.strategies = {}
		self.active_strategies = []
		self.strategy_weights = {}
		self.performance_tracker = {}
		
		# Initialize all strategies
		self._initialize_strategies()
	
	def _initialize_strategies(self):
		"""Initialize all available strategies"""
		
		self.strategies = {
			"ict": ICTStrategy(self.symbol, "H1"),
			"momentum": MomentumStrategy(self.symbol, "H1"),
			"mean_reversion": MeanReversionStrategy(self.symbol, "H1"),
			"breakout": BreakoutStrategy(self.symbol, "H1"),
			"range_trading": RangeTradingStrategy(self.symbol, "H1"),
			"multi_timeframe": MultiTimeframeStrategy(self.symbol, "H1")
		}
		
		# Set default weights
		self.strategy_weights = {
			"ict": 0.25,
			"momentum": 0.20,
			"mean_reversion": 0.15,
			"breakout": 0.15,
			"range_trading": 0.10,
			"multi_timeframe": 0.15
		}
		
		# Set all strategies as active by default
		self.active_strategies = list(self.strategies.keys())
	
	def generate_ensemble_signal(self, data: Dict[str, pd.DataFrame], 
							   context: Dict = None) -> Optional[TradingSignal]:
		"""Generate ensemble signal from all active strategies"""
		
		signals = {}
		
		# Generate signals from all active strategies
		for strategy_name in self.active_strategies:
			strategy = self.strategies[strategy_name]
			
			try:
				if strategy_name == "multi_timeframe":
					signal = strategy.generate_signal(data, context)
				else:
					primary_data = data.get("H1")
					if primary_data is not None and len(primary_data) >= 50:
						signal = strategy.generate_signal(primary_data, context)
					else:
						signal = None
				
				signals[strategy_name] = signal
				
			except Exception as e:
				print(f"Error generating signal for {strategy_name}: {e}")
				signals[strategy_name] = None
		
		# Combine signals using weighted voting
		ensemble_signal = self._combine_signals(signals)
		
		return ensemble_signal
	
	def _combine_signals(self, signals: Dict[str, Optional[TradingSignal]]) -> Optional[TradingSignal]:
		"""Combine signals using weighted voting"""
		
		# Filter valid signals
		valid_signals = {name: signal for name, signal in signals.items() if signal is not None}
		
		if not valid_signals:
			return None
		
		# Calculate weighted votes
		votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
		total_weight = 0.0
		
		for strategy_name, signal in valid_signals.items():
			weight = self.strategy_weights.get(strategy_name, 0.0)
			votes[signal.action] += weight * signal.confidence
			total_weight += weight
		
		# Normalize votes
		if total_weight > 0:
			for action in votes:
				votes[action] /= total_weight
		
		# Determine final action
		final_action = max(votes.items(), key=lambda x: x[1])[0]
		final_confidence = votes[final_action]
		
		if final_action == "HOLD" or final_confidence < 0.5:
			return None
		
		# Use the strongest signal for price levels
		strongest_signal = max(valid_signals.values(), key=lambda x: x.confidence)
		
		# Create ensemble signal
		ensemble_signal = TradingSignal(
			symbol=self.symbol,
			action=final_action,
			entry_price=strongest_signal.entry_price,
			stop_loss=strongest_signal.stop_loss,
			take_profit=strongest_signal.take_profit,
			volume=strongest_signal.volume,
			confidence=final_confidence,
			signal_strength=SignalStrength.STRONG if final_confidence > 0.8 else SignalStrength.MEDIUM,
			strategy_name="Ensemble Strategy",
			timeframe=strongest_signal.timeframe,
			timestamp=datetime.now(),
			reason=f"Ensemble: {final_action} signal with {final_confidence:.2f} confidence",
			risk_reward_ratio=strongest_signal.risk_reward_ratio,
			market_regime=strongest_signal.market_regime,
			additional_info={
				"individual_signals": {name: signal.action if signal else None for name, signal in signals.items()},
				"votes": votes,
				"active_strategies": self.active_strategies
			}
		)
		
		return ensemble_signal
	
	def update_strategy_weights(self, new_weights: Dict[str, float]):
		"""Update strategy weights based on performance"""
		
		self.strategy_weights.update(new_weights)
		
		# Normalize weights
		total_weight = sum(self.strategy_weights.values())
		if total_weight > 0:
			for strategy_name in self.strategy_weights:
				self.strategy_weights[strategy_name] /= total_weight
	
	def set_active_strategies(self, active_list: List[str]):
		"""Set which strategies are active"""
		
		self.active_strategies = [name for name in active_list if name in self.strategies]
	
	def get_strategy_performance(self) -> Dict:
		"""Get performance summary for all strategies"""
		
		performance = {}
		
		for strategy_name, strategy in self.strategies.items():
			performance[strategy_name] = {
				"weight": self.strategy_weights.get(strategy_name, 0.0),
				"active": strategy_name in self.active_strategies,
				"metrics": strategy.get_performance_metrics()
			}
		
		return performance