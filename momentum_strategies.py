import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5
from abc import ABC, abstractmethod
from ict_strategy import BaseStrategy, TradingSignal, SignalStrength


class MomentumStrategy(BaseStrategy):
	"""
	Momentum and Trend-Following Strategy
	Uses multiple technical indicators to identify and follow trends
	"""
	
	def __init__(self, symbol: str, timeframe: str = "H1"):
		super().__init__("Momentum Strategy", symbol, timeframe)
		self.parameters = {
			"ema_fast": 12,
			"ema_slow": 26,
			"macd_signal": 9,
			"rsi_period": 14,
			"rsi_overbought": 70,
			"rsi_oversold": 30,
			"atr_period": 14,
			"atr_multiplier": 2.0,
			"min_trend_strength": 0.6,
			"min_momentum": 0.5
		}
	
	def generate_signal(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], context: Dict = None) -> Optional[TradingSignal]:
		"""Generate momentum-based trading signal"""
		
		# Handle both single DataFrame and multi-timeframe data
		if isinstance(data, dict):
			# Use H1 data for momentum analysis
			if "H1" in data:
				analysis_data = data["H1"]
			else:
				# Use first available timeframe
				analysis_data = list(data.values())[0]
		else:
			analysis_data = data
		
		if len(analysis_data) < 50:
			return None
		
		# Calculate technical indicators
		indicators = self._calculate_indicators(analysis_data)
		
		# Analyze trend and momentum
		trend_analysis = self._analyze_trend(analysis_data, indicators)
		momentum_analysis = self._analyze_momentum(analysis_data, indicators)
		
		# Generate signal
		signal = self._generate_momentum_signal(analysis_data, indicators, trend_analysis, momentum_analysis)
		
		return signal
	
	def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
		"""Calculate all technical indicators"""
		
		indicators = {}
		
		# EMAs
		indicators['ema_fast'] = data['close'].ewm(span=self.parameters["ema_fast"]).mean()
		indicators['ema_slow'] = data['close'].ewm(span=self.parameters["ema_slow"]).mean()
		
		# MACD
		macd_line = indicators['ema_fast'] - indicators['ema_slow']
		indicators['macd'] = macd_line
		indicators['macd_signal'] = macd_line.ewm(span=self.parameters["macd_signal"]).mean()
		indicators['macd_histogram'] = macd_line - indicators['macd_signal']
		
		# RSI
		indicators['rsi'] = self._calculate_rsi(data['close'], self.parameters["rsi_period"])
		
		# ATR
		indicators['atr'] = self._calculate_atr(data, self.parameters["atr_period"])
		
		# Bollinger Bands
		bb_period = 20
		bb_std = 2
		indicators['bb_middle'] = data['close'].rolling(bb_period).mean()
		indicators['bb_std'] = data['close'].rolling(bb_period).std()
		indicators['bb_upper'] = indicators['bb_middle'] + (indicators['bb_std'] * bb_std)
		indicators['bb_lower'] = indicators['bb_middle'] - (indicators['bb_std'] * bb_std)
		
		# Stochastic
		indicators['stoch_k'], indicators['stoch_d'] = self._calculate_stochastic(data)
		
		# Volume indicators (if available)
		if 'volume' in data.columns:
			indicators['volume_sma'] = data['volume'].rolling(20).mean()
			indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
		
		return indicators
	
	def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
		"""Calculate RSI indicator"""
		
		delta = prices.diff()
		gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
		loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
		
		rs = gain / loss
		rsi = 100 - (100 / (1 + rs))
		
		return rsi
	
	def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
		"""Calculate Average True Range"""
		
		high_low = data['high'] - data['low']
		high_close = np.abs(data['high'] - data['close'].shift())
		low_close = np.abs(data['low'] - data['close'].shift())
		
		ranges = pd.concat([high_low, high_close, low_close], axis=1)
		true_range = ranges.max(axis=1)
		
		atr = true_range.rolling(period).mean()
		
		return atr
	
	def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
		"""Calculate Stochastic Oscillator"""
		
		lowest_low = data['low'].rolling(k_period).min()
		highest_high = data['high'].rolling(k_period).max()
		
		k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
		d_percent = k_percent.rolling(d_period).mean()
		
		return k_percent, d_percent
	
	def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict:
		"""Analyze trend direction and strength"""
		
		current_price = data['close'].iloc[-1]
		ema_fast = indicators['ema_fast'].iloc[-1]
		ema_slow = indicators['ema_slow'].iloc[-1]
		
		# Trend direction
		if ema_fast > ema_slow:
			trend_direction = "uptrend"
		elif ema_fast < ema_slow:
			trend_direction = "downtrend"
		else:
			trend_direction = "sideways"
		
		# Trend strength
		ema_separation = abs(ema_fast - ema_slow) / current_price
		trend_strength = min(ema_separation * 100, 1.0)
		
		# MACD trend confirmation
		macd = indicators['macd'].iloc[-1]
		macd_signal = indicators['macd_signal'].iloc[-1]
		macd_histogram = indicators['macd_histogram'].iloc[-1]
		
		macd_bullish = macd > macd_signal and macd_histogram > 0
		macd_bearish = macd < macd_signal and macd_histogram < 0
		
		# Bollinger Bands trend
		bb_upper = indicators['bb_upper'].iloc[-1]
		bb_lower = indicators['bb_lower'].iloc[-1]
		bb_middle = indicators['bb_middle'].iloc[-1]
		
		bb_position = "upper" if current_price > bb_upper else "lower" if current_price < bb_lower else "middle"
		
		return {
			"direction": trend_direction,
			"strength": trend_strength,
			"macd_bullish": macd_bullish,
			"macd_bearish": macd_bearish,
			"bb_position": bb_position,
			"ema_separation": ema_separation
		}
	
	def _analyze_momentum(self, data: pd.DataFrame, indicators: Dict) -> Dict:
		"""Analyze momentum indicators"""
		
		rsi = indicators['rsi'].iloc[-1]
		stoch_k = indicators['stoch_k'].iloc[-1]
		stoch_d = indicators['stoch_d'].iloc[-1]
		
		# RSI momentum
		rsi_momentum = "bullish" if rsi < self.parameters["rsi_oversold"] else "bearish" if rsi > self.parameters["rsi_overbought"] else "neutral"
		
		# Stochastic momentum
		stoch_momentum = "bullish" if stoch_k < 20 and stoch_k > stoch_d else "bearish" if stoch_k > 80 and stoch_k < stoch_d else "neutral"
		
		# MACD momentum
		macd_histogram = indicators['macd_histogram'].iloc[-1]
		macd_prev_histogram = indicators['macd_histogram'].iloc[-2]
		
		macd_momentum = "bullish" if macd_histogram > macd_prev_histogram else "bearish"
		
		# Overall momentum score
		momentum_score = 0.0
		if rsi_momentum == "bullish":
			momentum_score += 0.3
		elif rsi_momentum == "bearish":
			momentum_score -= 0.3
		
		if stoch_momentum == "bullish":
			momentum_score += 0.3
		elif stoch_momentum == "bearish":
			momentum_score -= 0.3
		
		if macd_momentum == "bullish":
			momentum_score += 0.4
		elif macd_momentum == "bearish":
			momentum_score -= 0.4
		
		return {
			"rsi": rsi,
			"rsi_momentum": rsi_momentum,
			"stoch_k": stoch_k,
			"stoch_d": stoch_d,
			"stoch_momentum": stoch_momentum,
			"macd_momentum": macd_momentum,
			"momentum_score": momentum_score
		}
	
	def _generate_momentum_signal(self, data: pd.DataFrame, indicators: Dict, 
								trend_analysis: Dict, momentum_analysis: Dict) -> Optional[TradingSignal]:
		"""Generate momentum-based trading signal"""
		
		current_price = data['close'].iloc[-1]
		current_time = data.index[-1]
		
		# Check minimum requirements
		if (trend_analysis["strength"] < self.parameters["min_trend_strength"] or
			abs(momentum_analysis["momentum_score"]) < self.parameters["min_momentum"]):
			return None
		
		# Determine action based on trend and momentum alignment
		action = None
		confidence = 0.0
		
		if (trend_analysis["direction"] == "uptrend" and 
			momentum_analysis["momentum_score"] > 0 and
			trend_analysis["macd_bullish"]):
			action = "BUY"
			confidence = (trend_analysis["strength"] + abs(momentum_analysis["momentum_score"])) / 2
		
		elif (trend_analysis["direction"] == "downtrend" and 
			  momentum_analysis["momentum_score"] < 0 and
			  trend_analysis["macd_bearish"]):
			action = "SELL"
			confidence = (trend_analysis["strength"] + abs(momentum_analysis["momentum_score"])) / 2
		
		if not action or confidence < 0.6:
			return None
		
		# Calculate stop loss and take profit
		atr = indicators['atr'].iloc[-1]
		stop_loss, take_profit = self._calculate_momentum_levels(action, current_price, atr, data)
		
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
			reason=f"Momentum: {action} signal based on trend and momentum alignment",
			risk_reward_ratio=risk_reward_ratio,
			market_regime=context.get("market_regime", "normal") if context else "normal",
			additional_info={
				"trend_analysis": trend_analysis,
				"momentum_analysis": momentum_analysis,
				"indicators": {k: v.iloc[-1] if hasattr(v, 'iloc') else v for k, v in indicators.items()}
			}
		)
		
		return signal
	
	def _calculate_momentum_levels(self, action: str, current_price: float, 
								 atr: float, data: pd.DataFrame) -> Tuple[float, float]:
		"""Calculate stop loss and take profit levels for momentum strategy"""
		
		atr_multiplier = self.parameters["atr_multiplier"]
		
		if action == "BUY":
			# Stop loss below recent low or ATR-based
			recent_low = data['low'].tail(20).min()
			atr_stop = current_price - (atr * atr_multiplier)
			stop_loss = max(recent_low, atr_stop)
			
			# Take profit at 2:1 risk-reward ratio
			risk = current_price - stop_loss
			take_profit = current_price + (risk * 2)
		
		else:  # SELL
			# Stop loss above recent high or ATR-based
			recent_high = data['high'].tail(20).max()
			atr_stop = current_price + (atr * atr_multiplier)
			stop_loss = min(recent_high, atr_stop)
			
			# Take profit at 2:1 risk-reward ratio
			risk = stop_loss - current_price
			take_profit = current_price - (risk * 2)
		
		return stop_loss, take_profit
	
	def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float = 0.01) -> float:
		"""Calculate position size based on risk management"""
		
		risk_amount = account_balance * risk_per_trade
		risk_per_pip = abs(signal.entry_price - signal.stop_loss)
		
		# Assuming 1 pip = 0.0001 for major pairs
		pip_value = 10.0  # $10 per pip for standard lot
		position_size = risk_amount / (risk_per_pip * pip_value)
		
		return max(0.01, min(position_size, 1.0))  # Min 0.01, Max 1.0 lots


class MeanReversionStrategy(BaseStrategy):
	"""
	Mean Reversion Strategy
	Identifies overbought/oversold conditions and trades reversals
	"""
	
	def __init__(self, symbol: str, timeframe: str = "H1"):
		super().__init__("Mean Reversion Strategy", symbol, timeframe)
		self.parameters = {
			"bb_period": 20,
			"bb_std": 2.0,
			"rsi_period": 14,
			"rsi_overbought": 70,
			"rsi_oversold": 30,
			"stoch_period": 14,
			"stoch_overbought": 80,
			"stoch_oversold": 20,
			"min_reversal_strength": 0.6,
			"max_risk_per_trade": 0.02
		}
	
	def generate_signal(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], context: Dict = None) -> Optional[TradingSignal]:
		"""Generate mean reversion trading signal"""
		
		# Handle both single DataFrame and multi-timeframe data
		if isinstance(data, dict):
			# Use H1 data for mean reversion analysis
			if "H1" in data:
				analysis_data = data["H1"]
			else:
				# Use first available timeframe
				analysis_data = list(data.values())[0]
		else:
			analysis_data = data
		
		if len(analysis_data) < 50:
			return None
		
		# Calculate indicators
		indicators = self._calculate_reversion_indicators(analysis_data)
		
		# Analyze oversold/overbought conditions
		reversion_signals = self._analyze_reversion_signals(analysis_data, indicators)
		
		# Generate signal
		signal = self._generate_reversion_signal(analysis_data, indicators, reversion_signals)
		
		return signal
	
	def _calculate_reversion_indicators(self, data: pd.DataFrame) -> Dict:
		"""Calculate mean reversion indicators"""
		
		indicators = {}
		
		# Bollinger Bands
		indicators['bb_middle'] = data['close'].rolling(self.parameters["bb_period"]).mean()
		indicators['bb_std'] = data['close'].rolling(self.parameters["bb_period"]).std()
		indicators['bb_upper'] = indicators['bb_middle'] + (indicators['bb_std'] * self.parameters["bb_std"])
		indicators['bb_lower'] = indicators['bb_middle'] - (indicators['bb_std'] * self.parameters["bb_std"])
		indicators['bb_position'] = (data['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
		
		# RSI
		indicators['rsi'] = self._calculate_rsi(data['close'], self.parameters["rsi_period"])
		
		# Stochastic
		indicators['stoch_k'], indicators['stoch_d'] = self._calculate_stochastic(data, self.parameters["stoch_period"])
		
		# Williams %R
		indicators['williams_r'] = self._calculate_williams_r(data)
		
		# CCI (Commodity Channel Index)
		indicators['cci'] = self._calculate_cci(data)
		
		return indicators
	
	def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
		"""Calculate RSI indicator"""
		
		delta = prices.diff()
		gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
		loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
		
		rs = gain / loss
		rsi = 100 - (100 / (1 + rs))
		
		return rsi
	
	def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
		"""Calculate Stochastic Oscillator"""
		
		lowest_low = data['low'].rolling(k_period).min()
		highest_high = data['high'].rolling(k_period).max()
		
		k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
		d_percent = k_percent.rolling(d_period).mean()
		
		return k_percent, d_percent
	
	def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
		"""Calculate Williams %R"""
		
		highest_high = data['high'].rolling(period).max()
		lowest_low = data['low'].rolling(period).min()
		
		williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
		
		return williams_r
	
	def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
		"""Calculate Commodity Channel Index"""
		
		tp = (data['high'] + data['low'] + data['close']) / 3
		sma_tp = tp.rolling(period).mean()
		mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
		
		cci = (tp - sma_tp) / (0.015 * mad)
		
		return cci
	
	def _analyze_reversion_signals(self, data: pd.DataFrame, indicators: Dict) -> Dict:
		"""Analyze mean reversion signals"""
		
		current_price = data['close'].iloc[-1]
		bb_position = indicators['bb_position'].iloc[-1]
		rsi = indicators['rsi'].iloc[-1]
		stoch_k = indicators['stoch_k'].iloc[-1]
		williams_r = indicators['williams_r'].iloc[-1]
		cci = indicators['cci'].iloc[-1]
		
		# Oversold conditions
		oversold_signals = 0
		if bb_position < 0.2:  # Below lower Bollinger Band
			oversold_signals += 1
		if rsi < self.parameters["rsi_oversold"]:
			oversold_signals += 1
		if stoch_k < self.parameters["stoch_oversold"]:
			oversold_signals += 1
		if williams_r < -80:
			oversold_signals += 1
		if cci < -100:
			oversold_signals += 1
		
		# Overbought conditions
		overbought_signals = 0
		if bb_position > 0.8:  # Above upper Bollinger Band
			overbought_signals += 1
		if rsi > self.parameters["rsi_overbought"]:
			overbought_signals += 1
		if stoch_k > self.parameters["stoch_overbought"]:
			overbought_signals += 1
		if williams_r > -20:
			overbought_signals += 1
		if cci > 100:
			overbought_signals += 1
		
		# Reversal confirmation
		reversal_confirmation = self._check_reversal_confirmation(data, indicators)
		
		return {
			"oversold_signals": oversold_signals,
			"overbought_signals": overbought_signals,
			"reversal_confirmation": reversal_confirmation,
			"bb_position": bb_position,
			"rsi": rsi,
			"stoch_k": stoch_k,
			"williams_r": williams_r,
			"cci": cci
		}
	
	def _check_reversal_confirmation(self, data: pd.DataFrame, indicators: Dict) -> Dict:
		"""Check for reversal confirmation signals"""
		
		# Price action confirmation
		current_candle = data.iloc[-1]
		prev_candle = data.iloc[-2]
		
		# Bullish reversal confirmation
		bullish_confirmation = (
			current_candle['close'] > current_candle['open'] and  # Bullish candle
			current_candle['close'] > prev_candle['close'] and    # Higher close
			current_candle['low'] < prev_candle['low']           # Lower low (hammer pattern)
		)
		
		# Bearish reversal confirmation
		bearish_confirmation = (
			current_candle['close'] < current_candle['open'] and  # Bearish candle
			current_candle['close'] < prev_candle['close'] and    # Lower close
			current_candle['high'] > prev_candle['high']         # Higher high (shooting star pattern)
		)
		
		# Divergence confirmation
		price_divergence = self._check_divergence(data, indicators)
		
		return {
			"bullish_confirmation": bullish_confirmation,
			"bearish_confirmation": bearish_confirmation,
			"price_divergence": price_divergence
		}
	
	def _check_divergence(self, data: pd.DataFrame, indicators: Dict) -> Dict:
		"""Check for price-indicator divergence"""
		
		# Look at last 10 periods for divergence
		recent_data = data.tail(10)
		recent_rsi = indicators['rsi'].tail(10)
		
		# Price highs and lows
		price_highs = recent_data['high'].rolling(3).max()
		price_lows = recent_data['low'].rolling(3).min()
		rsi_highs = recent_rsi.rolling(3).max()
		rsi_lows = recent_rsi.rolling(3).min()
		
		# Bullish divergence: price makes lower lows, RSI makes higher lows
		bullish_divergence = (
			price_lows.iloc[-1] < price_lows.iloc[-3] and
			rsi_lows.iloc[-1] > rsi_lows.iloc[-3]
		)
		
		# Bearish divergence: price makes higher highs, RSI makes lower highs
		bearish_divergence = (
			price_highs.iloc[-1] > price_highs.iloc[-3] and
			rsi_highs.iloc[-1] < rsi_highs.iloc[-3]
		)
		
		return {
			"bullish_divergence": bullish_divergence,
			"bearish_divergence": bearish_divergence
		}
	
	def _generate_reversion_signal(self, data: pd.DataFrame, indicators: Dict, 
								reversion_signals: Dict) -> Optional[TradingSignal]:
		"""Generate mean reversion trading signal"""
		
		current_price = data['close'].iloc[-1]
		current_time = data.index[-1]
		
		# Check minimum requirements
		min_signals = 3  # Need at least 3 indicators to confirm
		
		action = None
		confidence = 0.0
		
		# Oversold condition - potential BUY
		if (reversion_signals["oversold_signals"] >= min_signals and
			reversion_signals["reversal_confirmation"]["bullish_confirmation"]):
			action = "BUY"
			confidence = min(reversion_signals["oversold_signals"] / 5.0, 1.0)
		
		# Overbought condition - potential SELL
		elif (reversion_signals["overbought_signals"] >= min_signals and
			  reversion_signals["reversal_confirmation"]["bearish_confirmation"]):
			action = "SELL"
			confidence = min(reversion_signals["overbought_signals"] / 5.0, 1.0)
		
		# Add divergence bonus
		if action == "BUY" and reversion_signals["reversal_confirmation"]["price_divergence"]["bullish_divergence"]:
			confidence += 0.2
		elif action == "SELL" and reversion_signals["reversal_confirmation"]["price_divergence"]["bearish_divergence"]:
			confidence += 0.2
		
		if not action or confidence < self.parameters["min_reversal_strength"]:
			return None
		
		# Calculate stop loss and take profit
		stop_loss, take_profit = self._calculate_reversion_levels(action, current_price, data, indicators)
		
		# Calculate risk-reward ratio
		risk = abs(current_price - stop_loss)
		reward = abs(take_profit - current_price)
		risk_reward_ratio = reward / risk if risk > 0 else 0
		
		if risk_reward_ratio < 1.0:  # Mean reversion can have lower R:R
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
			reason=f"Mean Reversion: {action} signal based on oversold/overbought conditions",
			risk_reward_ratio=risk_reward_ratio,
			market_regime=context.get("market_regime", "normal") if context else "normal",
			additional_info={
				"reversion_signals": reversion_signals,
				"indicators": {k: v.iloc[-1] if hasattr(v, 'iloc') else v for k, v in indicators.items()}
			}
		)
		
		return signal
	
	def _calculate_reversion_levels(self, action: str, current_price: float, 
								  data: pd.DataFrame, indicators: Dict) -> Tuple[float, float]:
		"""Calculate stop loss and take profit levels for mean reversion"""
		
		bb_upper = indicators['bb_upper'].iloc[-1]
		bb_lower = indicators['bb_lower'].iloc[-1]
		bb_middle = indicators['bb_middle'].iloc[-1]
		
		if action == "BUY":
			# Stop loss below recent low or Bollinger Band
			recent_low = data['low'].tail(10).min()
			stop_loss = min(recent_low, bb_lower * 0.999)  # Slightly below BB lower
			
			# Take profit at Bollinger Band middle or upper
			take_profit = bb_middle
		
		else:  # SELL
			# Stop loss above recent high or Bollinger Band
			recent_high = data['high'].tail(10).max()
			stop_loss = max(recent_high, bb_upper * 1.001)  # Slightly above BB upper
			
			# Take profit at Bollinger Band middle or lower
			take_profit = bb_middle
		
		return stop_loss, take_profit
	
	def calculate_position_size(self, signal: TradingSignal, account_balance: float, risk_per_trade: float = 0.01) -> float:
		"""Calculate position size based on risk management"""
		
		# Use smaller position size for mean reversion
		risk_amount = account_balance * min(risk_per_trade, self.parameters["max_risk_per_trade"])
		risk_per_pip = abs(signal.entry_price - signal.stop_loss)
		
		# Assuming 1 pip = 0.0001 for major pairs
		pip_value = 10.0  # $10 per pip for standard lot
		position_size = risk_amount / (risk_per_pip * pip_value)
		
		return max(0.01, min(position_size, 0.5))  # Smaller max size for mean reversion