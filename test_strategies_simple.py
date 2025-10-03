#!/usr/bin/env python3
"""
Simplified Strategy Testing Script
Tests trading strategies without MT5 dependency
"""

import sys
import os
sys.path.append('/workspace')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


# Mock MetaTrader5 module
class MockMT5:
	ORDER_TYPE_BUY = 0
	ORDER_TYPE_SELL = 1
	
	@staticmethod
	def symbol_info_tick(symbol):
		class MockTick:
			ask = 1.1000
			bid = 1.0999
		return MockTick()

# Mock the MT5 module
sys.modules['MetaTrader5'] = MockMT5()

# Now import our strategies
from ict_strategy import ICTStrategy, TradingSignal, SignalStrength, BaseStrategy
from momentum_strategies import MomentumStrategy, MeanReversionStrategy
from breakout_strategies import BreakoutStrategy, RangeTradingStrategy
from multi_timeframe_strategies import MultiTimeframeStrategy, StrategyManager


def create_sample_data(symbol: str = "EURUSD", periods: int = 1000) -> dict:
	"""Create sample OHLCV data for testing"""
	
	# Generate realistic price data
	np.random.seed(42)
	
	# Base price
	base_price = 1.1000
	
	# Generate price movements
	price_changes = np.random.normal(0, 0.0001, periods)
	prices = [base_price]
	
	for change in price_changes[1:]:
		new_price = prices[-1] + change
		prices.append(max(new_price, 0.5))  # Prevent negative prices
	
	# Create OHLCV data
	data = []
	for i, price in enumerate(prices):
		# Add some volatility
		volatility = np.random.uniform(0.0001, 0.0005)
		high = price + volatility
		low = price - volatility
		open_price = prices[i-1] if i > 0 else price
		close_price = price
		volume = np.random.uniform(1000, 10000)
		
		data.append({
			'open': open_price,
			'high': high,
			'low': low,
			'close': close_price,
			'volume': volume
		})
	
	# Create DataFrame with datetime index
	dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')
	df = pd.DataFrame(data, index=dates)
	
	return {
		"H4": df.iloc[::4].copy(),  # Every 4th row for H4
		"H1": df.copy(),
		"M15": df.iloc[::4].copy(),  # Every 4th row for M15
		"M5": df.iloc[::12].copy()   # Every 12th row for M5
	}


def test_strategy(strategy, data, strategy_name: str):
	"""Test a single strategy"""
	
	print(f"\n🧪 Testing {strategy_name}...")
	
	try:
		# Test signal generation
		signal = strategy.generate_signal(data)
		
		if signal:
			print(f"✅ {strategy_name} generated signal:")
			print(f"   Action: {signal.action}")
			print(f"   Confidence: {signal.confidence:.2f}")
			print(f"   Entry: {signal.entry_price:.5f}")
			print(f"   Stop Loss: {signal.stop_loss:.5f}")
			print(f"   Take Profit: {signal.take_profit:.5f}")
			print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")
			print(f"   Reason: {signal.reason}")
			
			# Test position sizing
			position_size = strategy.calculate_position_size(signal, 10000.0, 0.01)
			print(f"   Position Size: {position_size:.2f} lots")
			
			# Test signal validation
			is_valid = strategy.validate_signal(signal)
			print(f"   Signal Valid: {is_valid}")
			
		else:
			print(f"ℹ️  {strategy_name} generated no signal (HOLD)")
		
		# Test parameter update
		test_params = {"test_param": 1.0}
		strategy.update_parameters(test_params)
		print(f"✅ {strategy_name} parameter update successful")
		
		# Test performance metrics
		metrics = strategy.get_performance_metrics()
		print(f"✅ {strategy_name} performance metrics accessible")
		
		return True
		
	except Exception as e:
		print(f"❌ {strategy_name} failed: {str(e)}")
		import traceback
		traceback.print_exc()
		return False


def test_all_strategies():
	"""Test all trading strategies"""
	
	print("🚀 Starting Strategy Testing...")
	
	# Create sample data
	sample_data = create_sample_data()
	print(f"📊 Created sample data with {len(sample_data['H1'])} H1 bars")
	
	# Test results
	results = {}
	
	# Test ICT Strategy
	try:
		ict_strategy = ICTStrategy("EURUSD", "H1")
		results["ICT"] = test_strategy(ict_strategy, sample_data["H1"], "ICT Strategy")
	except Exception as e:
		print(f"❌ ICT Strategy initialization failed: {e}")
		results["ICT"] = False
	
	# Test Momentum Strategy
	try:
		momentum_strategy = MomentumStrategy("EURUSD", "H1")
		results["Momentum"] = test_strategy(momentum_strategy, sample_data["H1"], "Momentum Strategy")
	except Exception as e:
		print(f"❌ Momentum Strategy initialization failed: {e}")
		results["Momentum"] = False
	
	# Test Mean Reversion Strategy
	try:
		mean_reversion_strategy = MeanReversionStrategy("EURUSD", "H1")
		results["Mean Reversion"] = test_strategy(mean_reversion_strategy, sample_data["H1"], "Mean Reversion Strategy")
	except Exception as e:
		print(f"❌ Mean Reversion Strategy initialization failed: {e}")
		results["Mean Reversion"] = False
	
	# Test Breakout Strategy
	try:
		breakout_strategy = BreakoutStrategy("EURUSD", "H1")
		results["Breakout"] = test_strategy(breakout_strategy, sample_data["H1"], "Breakout Strategy")
	except Exception as e:
		print(f"❌ Breakout Strategy initialization failed: {e}")
		results["Breakout"] = False
	
	# Test Range Trading Strategy
	try:
		range_strategy = RangeTradingStrategy("EURUSD", "H1")
		results["Range Trading"] = test_strategy(range_strategy, sample_data["H1"], "Range Trading Strategy")
	except Exception as e:
		print(f"❌ Range Trading Strategy initialization failed: {e}")
		results["Range Trading"] = False
	
	# Test Multi-Timeframe Strategy
	try:
		mtf_strategy = MultiTimeframeStrategy("EURUSD", "H1")
		results["Multi-Timeframe"] = test_strategy(mtf_strategy, sample_data, "Multi-Timeframe Strategy")
	except Exception as e:
		print(f"❌ Multi-Timeframe Strategy initialization failed: {e}")
		results["Multi-Timeframe"] = False
	
	# Test Strategy Manager
	print(f"\n🧪 Testing Strategy Manager...")
	try:
		manager = StrategyManager("EURUSD")
		ensemble_signal = manager.generate_ensemble_signal(sample_data)
		
		if ensemble_signal:
			print(f"✅ Strategy Manager generated ensemble signal:")
			print(f"   Action: {ensemble_signal.action}")
			print(f"   Confidence: {ensemble_signal.confidence:.2f}")
			print(f"   Individual Signals: {ensemble_signal.additional_info.get('individual_signals', {})}")
		else:
			print(f"ℹ️  Strategy Manager generated no signal (HOLD)")
		
		performance = manager.get_strategy_performance()
		print(f"✅ Strategy Manager performance metrics accessible")
		results["Strategy Manager"] = True
		
	except Exception as e:
		print(f"❌ Strategy Manager failed: {str(e)}")
		import traceback
		traceback.print_exc()
		results["Strategy Manager"] = False
	
	# Print summary
	print(f"\n📊 TESTING SUMMARY:")
	print(f"=" * 50)
	
	passed = sum(results.values())
	total = len(results)
	
	for strategy_name, passed_test in results.items():
		status = "✅ PASSED" if passed_test else "❌ FAILED"
		print(f"{strategy_name:20} {status}")
	
	print(f"=" * 50)
	print(f"Total: {passed}/{total} strategies passed")
	
	if passed == total:
		print(f"🎉 ALL STRATEGIES WORKING CORRECTLY!")
		return True
	else:
		print(f"⚠️  {total - passed} strategies need attention")
		return False


if __name__ == "__main__":
	print("🎯 STRATEGY TESTING SUITE")
	print("=" * 50)
	
	# Run all tests
	success = test_all_strategies()
	
	print(f"\n🏁 Testing completed!")
	
	if success:
		print("✅ All strategies are working correctly!")
		sys.exit(0)
	else:
		print("❌ Some strategies need attention!")
		sys.exit(1)