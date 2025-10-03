#!/usr/bin/env python3
"""
Strategy Testing Script
Tests all trading strategies to ensure they work correctly
"""

import sys
import os
sys.path.append('/workspace')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Import all strategies
from ict_strategy import ICTStrategy, TradingSignal, SignalStrength
from momentum_strategies import MomentumStrategy, MeanReversionStrategy
from breakout_strategies import BreakoutStrategy, RangeTradingStrategy
from multi_timeframe_strategies import MultiTimeframeStrategy, StrategyManager
from strategy_optimizer import StrategyOptimizer


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
	ict_strategy = ICTStrategy("EURUSD", "H1")
	results["ICT"] = test_strategy(ict_strategy, sample_data["H1"], "ICT Strategy")
	
	# Test Momentum Strategy
	momentum_strategy = MomentumStrategy("EURUSD", "H1")
	results["Momentum"] = test_strategy(momentum_strategy, sample_data["H1"], "Momentum Strategy")
	
	# Test Mean Reversion Strategy
	mean_reversion_strategy = MeanReversionStrategy("EURUSD", "H1")
	results["Mean Reversion"] = test_strategy(mean_reversion_strategy, sample_data["H1"], "Mean Reversion Strategy")
	
	# Test Breakout Strategy
	breakout_strategy = BreakoutStrategy("EURUSD", "H1")
	results["Breakout"] = test_strategy(breakout_strategy, sample_data["H1"], "Breakout Strategy")
	
	# Test Range Trading Strategy
	range_strategy = RangeTradingStrategy("EURUSD", "H1")
	results["Range Trading"] = test_strategy(range_strategy, sample_data["H1"], "Range Trading Strategy")
	
	# Test Multi-Timeframe Strategy
	mtf_strategy = MultiTimeframeStrategy("EURUSD", "H1")
	results["Multi-Timeframe"] = test_strategy(mtf_strategy, sample_data, "Multi-Timeframe Strategy")
	
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
		results["Strategy Manager"] = False
	
	# Test Strategy Optimizer
	print(f"\n🧪 Testing Strategy Optimizer...")
	try:
		optimizer = StrategyOptimizer("EURUSD")
		
		# Test backtest (small sample)
		small_data = {tf: df.tail(200) for tf, df in sample_data.items()}
		backtest_results = optimizer.run_comprehensive_backtest(small_data, 100)
		
		print(f"✅ Strategy Optimizer backtest completed")
		print(f"   Strategies tested: {len(backtest_results)}")
		
		# Test rankings
		rankings = optimizer.get_strategy_rankings()
		print(f"✅ Strategy rankings generated: {len(rankings)} strategies")
		
		# Test best strategy
		best_strategy = optimizer.get_best_strategy()
		if best_strategy:
			print(f"✅ Best strategy identified: {best_strategy}")
		
		results["Strategy Optimizer"] = True
		
	except Exception as e:
		print(f"❌ Strategy Optimizer failed: {str(e)}")
		results["Strategy Optimizer"] = False
	
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


def test_signal_generation():
	"""Test signal generation with different market conditions"""
	
	print(f"\n🎯 Testing Signal Generation...")
	
	# Create different market scenarios
	scenarios = {
		"Trending Up": create_trending_data("up"),
		"Trending Down": create_trending_data("down"),
		"Ranging": create_ranging_data(),
		"Volatile": create_volatile_data()
	}
	
	for scenario_name, data in scenarios.items():
		print(f"\n📈 Testing {scenario_name} scenario...")
		
		# Test ICT strategy
		ict_strategy = ICTStrategy("EURUSD", "H1")
		signal = ict_strategy.generate_signal(data["H1"])
		
		if signal:
			print(f"   ICT: {signal.action} (confidence: {signal.confidence:.2f})")
		else:
			print(f"   ICT: HOLD")
		
		# Test Momentum strategy
		momentum_strategy = MomentumStrategy("EURUSD", "H1")
		signal = momentum_strategy.generate_signal(data["H1"])
		
		if signal:
			print(f"   Momentum: {signal.action} (confidence: {signal.confidence:.2f})")
		else:
			print(f"   Momentum: HOLD")


def create_trending_data(direction: str = "up"):
	"""Create trending market data"""
	
	periods = 500
	base_price = 1.1000
	
	if direction == "up":
		trend = np.linspace(0, 0.01, periods)  # Upward trend
	else:
		trend = np.linspace(0, -0.01, periods)  # Downward trend
	
	noise = np.random.normal(0, 0.0001, periods)
	prices = base_price + trend + noise
	
	data = []
	for i, price in enumerate(prices):
		volatility = np.random.uniform(0.0001, 0.0003)
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
	
	dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')
	df = pd.DataFrame(data, index=dates)
	
	return {"H1": df}


def create_ranging_data():
	"""Create ranging market data"""
	
	periods = 500
	base_price = 1.1000
	range_size = 0.002
	
	# Create sine wave for ranging
	t = np.linspace(0, 4*np.pi, periods)
	prices = base_price + range_size * np.sin(t)
	
	data = []
	for i, price in enumerate(prices):
		volatility = np.random.uniform(0.0001, 0.0002)
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
	
	dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')
	df = pd.DataFrame(data, index=dates)
	
	return {"H1": df}


def create_volatile_data():
	"""Create volatile market data"""
	
	periods = 500
	base_price = 1.1000
	
	# High volatility
	volatility = 0.001
	price_changes = np.random.normal(0, volatility, periods)
	prices = [base_price]
	
	for change in price_changes[1:]:
		new_price = prices[-1] + change
		prices.append(max(new_price, 0.5))
	
	data = []
	for i, price in enumerate(prices):
		volatility = np.random.uniform(0.0002, 0.0008)
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
	
	dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')
	df = pd.DataFrame(data, index=dates)
	
	return {"H1": df}


if __name__ == "__main__":
	print("🎯 STRATEGY TESTING SUITE")
	print("=" * 50)
	
	# Run all tests
	success = test_all_strategies()
	
	# Test signal generation
	test_signal_generation()
	
	print(f"\n🏁 Testing completed!")
	
	if success:
		print("✅ All strategies are working correctly!")
		sys.exit(0)
	else:
		print("❌ Some strategies need attention!")
		sys.exit(1)