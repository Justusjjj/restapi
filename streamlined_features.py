import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from ict_price_action import ICTPriceActionAnalyzer


class StreamlinedICTFeatures:
	"""
	Extract only the most reliable ICT/price action signals as explicit counts and levels.
	Focus on order blocks, fair value gaps, liquidity sweeps, and market structure breaks.
	"""
	
	def __init__(self):
		self.ict = ICTPriceActionAnalyzer()
	
	def extract_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
		"""Count bullish/bearish order blocks in recent bars"""
		if df.empty or len(df) < 10:
			return {"bullish_obs": 0, "bearish_obs": 0, "latest_ob_price": None, "ob_strength": 0.0}
		
		# Use ICT analyzer to get order blocks
		ict_data = self.ict.analyze_ict_levels(df.tail(lookback))
		
		bullish_obs = int(ict_data['order_block_bullish'].sum())
		bearish_obs = int(ict_data['order_block_bearish'].sum())
		
		# Get latest order block price
		latest_ob_price = None
		ob_strength = 0.0
		
		if bullish_obs > 0:
			latest_bullish = ict_data[ict_data['order_block_bullish'] > 0]['order_block_bullish'].iloc[-1]
			if latest_bullish > 0:
				latest_ob_price = latest_bullish
				ob_strength = 0.5
		
		if bearish_obs > 0:
			latest_bearish = ict_data[ict_data['order_block_bearish'] > 0]['order_block_bearish'].iloc[-1]
			if latest_bearish > 0:
				latest_ob_price = latest_bearish
				ob_strength = -0.5
		
		return {
			"bullish_obs": bullish_obs,
			"bearish_obs": bearish_obs,
			"latest_ob_price": latest_ob_price,
			"ob_strength": ob_strength
		}
	
	def extract_fair_value_gaps(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
		"""Count and analyze fair value gaps"""
		if df.empty or len(df) < 10:
			return {"bullish_fvgs": 0, "bearish_fvgs": 0, "latest_fvg_price": None, "fvg_strength": 0.0}
		
		ict_data = self.ict.analyze_ict_levels(df.tail(lookback))
		
		bullish_fvgs = int(ict_data['fvg_bullish'].sum())
		bearish_fvgs = int(ict_data['fvg_bearish'].sum())
		
		latest_fvg_price = None
		fvg_strength = 0.0
		
		if bullish_fvgs > 0:
			latest_bullish = ict_data[ict_data['fvg_bullish'] > 0]['fvg_bullish'].iloc[-1]
			if latest_bullish > 0:
				latest_fvg_price = latest_bullish
				fvg_strength = 0.3
		
		if bearish_fvgs > 0:
			latest_bearish = ict_data[ict_data['fvg_bearish'] > 0]['fvg_bearish'].iloc[-1]
			if latest_bearish > 0:
				latest_fvg_price = latest_bearish
				fvg_strength = -0.3
		
		return {
			"bullish_fvgs": bullish_fvgs,
			"bearish_fvgs": bearish_fvgs,
			"latest_fvg_price": latest_fvg_price,
			"fvg_strength": fvg_strength
		}
	
	def extract_liquidity_sweeps(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
		"""Detect liquidity sweeps (break of recent highs/lows)"""
		if df.empty or len(df) < 20:
			return {"sweep_high": False, "sweep_low": False, "sweep_strength": 0.0}
		
		recent_data = df.tail(lookback)
		current_high = recent_data['high'].iloc[-1]
		current_low = recent_data['low'].iloc[-1]
		
		# Check if current bar swept previous highs/lows
		prev_highs = recent_data['high'].iloc[:-1]
		prev_lows = recent_data['low'].iloc[:-1]
		
		sweep_high = current_high > prev_highs.max() and current_high > recent_data['high'].iloc[-2]
		sweep_low = current_low < prev_lows.min() and current_low < recent_data['low'].iloc[-2]
		
		sweep_strength = 0.0
		if sweep_high:
			sweep_strength = 0.4
		elif sweep_low:
			sweep_strength = -0.4
		
		return {
			"sweep_high": sweep_high,
			"sweep_low": sweep_low,
			"sweep_strength": sweep_strength
		}
	
	def extract_market_structure(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
		"""Detect market structure breaks (higher highs, lower lows)"""
		if df.empty or len(df) < 20:
			return {"structure": "UNKNOWN", "break_bullish": False, "break_bearish": False, "structure_strength": 0.0}
		
		recent_data = df.tail(lookback)
		
		# Simple structure detection
		highs = recent_data['high'].rolling(5).max()
		lows = recent_data['low'].rolling(5).min()
		
		# Check for structure breaks
		latest_high = highs.iloc[-1]
		latest_low = lows.iloc[-1]
		prev_high = highs.iloc[-6] if len(highs) > 5 else highs.iloc[0]
		prev_low = lows.iloc[-6] if len(lows) > 5 else lows.iloc[0]
		
		break_bullish = latest_high > prev_high
		break_bearish = latest_low < prev_low
		
		structure = "NEUTRAL"
		structure_strength = 0.0
		
		if break_bullish and not break_bearish:
			structure = "BULLISH"
			structure_strength = 0.6
		elif break_bearish and not break_bullish:
			structure = "BEARISH"
			structure_strength = -0.6
		
		return {
			"structure": structure,
			"break_bullish": break_bullish,
			"break_bearish": break_bearish,
			"structure_strength": structure_strength
		}
	
	def get_composite_signal(self, df: pd.DataFrame) -> Dict:
		"""Combine all ICT signals into a composite score"""
		ob_data = self.extract_order_blocks(df)
		fvg_data = self.extract_fair_value_gaps(df)
		liquidity_data = self.extract_liquidity_sweeps(df)
		structure_data = self.extract_market_structure(df)
		
		# Weighted composite score
		composite_score = (
			ob_data['ob_strength'] * 0.4 +
			fvg_data['fvg_strength'] * 0.2 +
			liquidity_data['sweep_strength'] * 0.2 +
			structure_data['structure_strength'] * 0.2
		)
		
		signal = "NEUTRAL"
		if composite_score > 0.3:
			signal = "BULLISH"
		elif composite_score < -0.3:
			signal = "BEARISH"
		
		return {
			"signal": signal,
			"composite_score": composite_score,
			"order_blocks": ob_data,
			"fair_value_gaps": fvg_data,
			"liquidity_sweeps": liquidity_data,
			"market_structure": structure_data,
			"confidence": min(abs(composite_score) * 2, 1.0)
		}


def get_economic_calendar_proxy(symbol: str, timestamp: datetime) -> Dict:
	"""
	Simplified economic calendar proxy based on time patterns.
	In production, integrate with real economic calendar API.
	"""
	hour = timestamp.hour
	day_of_week = timestamp.weekday()
	
	# High impact times (simplified)
	high_impact_hours = [8, 9, 13, 14, 15]  # London/NY overlap, major releases
	high_impact_days = [0, 1, 2, 3]  # Monday-Thursday
	
	impact_score = 0.0
	if hour in high_impact_hours:
		impact_score += 0.5
	if day_of_week in high_impact_days:
		impact_score += 0.3
	
	# Currency-specific events (simplified)
	base = symbol[:3]
	if base in ['USD'] and hour in [13, 14]:  # US session
		impact_score += 0.2
	elif base in ['EUR'] and hour in [8, 9]:  # London session
		impact_score += 0.2
	elif base in ['GBP'] and hour in [8, 9]:  # London session
		impact_score += 0.2
	
	return {
		"impact_score": min(impact_score, 1.0),
		"high_impact_time": impact_score > 0.5,
		"session": "LONDON" if 8 <= hour < 16 else "NY" if 13 <= hour < 21 else "ASIA"
	}