import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5
from abc import ABC, abstractmethod
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

from ict_strategy import BaseStrategy, TradingSignal, SignalStrength
from ict_strategy import ICTStrategy
from momentum_strategies import MomentumStrategy, MeanReversionStrategy
from breakout_strategies import BreakoutStrategy, RangeTradingStrategy
from multi_timeframe_strategies import MultiTimeframeStrategy, StrategyManager


@dataclass
class StrategyPerformance:
	"""Comprehensive strategy performance metrics"""
	strategy_name: str
	total_signals: int
	winning_signals: int
	losing_signals: int
	win_rate: float
	profit_factor: float
	total_pnl: float
	max_drawdown: float
	sharpe_ratio: float
	sortino_ratio: float
	calmar_ratio: float
	avg_risk_reward: float
	avg_confidence: float
	avg_trade_duration: float
	best_period: str
	worst_period: str
	regime_performance: Dict[str, Dict]
	monthly_returns: Dict[str, float]
	equity_curve: List[float]
	trade_history: List[Dict]


class StrategyOptimizer:
	"""
	Strategy Performance Comparison and Optimization
	"""
	
	def __init__(self, symbol: str):
		self.symbol = symbol
		self.strategies = {}
		self.performance_results = {}
		self.optimization_results = {}
		self.backtest_data = None
		
		# Initialize all strategies
		self._initialize_strategies()
	
	def _initialize_strategies(self):
		"""Initialize all available strategies"""
		
		self.strategies = {
			"ICT": ICTStrategy(self.symbol, "H1"),
			"Momentum": MomentumStrategy(self.symbol, "H1"),
			"Mean_Reversion": MeanReversionStrategy(self.symbol, "H1"),
			"Breakout": BreakoutStrategy(self.symbol, "H1"),
			"Range_Trading": RangeTradingStrategy(self.symbol, "H1"),
			"Multi_Timeframe": MultiTimeframeStrategy(self.symbol, "H1")
		}
	
	def run_comprehensive_backtest(self, data: Dict[str, pd.DataFrame], 
								 lookback_periods: int = 1000) -> Dict[str, StrategyPerformance]:
		"""
		Run comprehensive backtest for all strategies
		"""
		
		print("Running comprehensive strategy backtest...")
		
		results = {}
		
		for strategy_name, strategy in self.strategies.items():
			print(f"Backtesting {strategy_name}...")
			
			try:
				performance = self._backtest_strategy(strategy, data, lookback_periods)
				results[strategy_name] = performance
				
			except Exception as e:
				print(f"Error backtesting {strategy_name}: {e}")
				continue
		
		self.performance_results = results
		return results
	
	def _backtest_strategy(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame], 
						 lookback_periods: int) -> StrategyPerformance:
		"""Backtest a single strategy"""
		
		# Handle different strategy types
		if isinstance(strategy, MultiTimeframeStrategy):
			# Multi-timeframe strategy needs all timeframes
			backtest_data = data
		else:
			# Single timeframe strategies use H1 data
			primary_data = data.get("H1")
			if primary_data is None or len(primary_data) < lookback_periods:
				raise ValueError("Insufficient data for backtesting")
			
			# Prepare backtest data
			backtest_data = primary_data.tail(lookback_periods).copy()
		
		# Initialize backtest variables
		balance = 10000.0
		equity_curve = [balance]
		trade_history = []
		positions = {}
		max_balance = balance
		max_drawdown = 0.0
		
		# Process each bar
		if isinstance(strategy, MultiTimeframeStrategy):
			# Multi-timeframe strategy processing
			for i in range(50, len(backtest_data["H1"])):
				current_bar = backtest_data["H1"].iloc[i]
				
				# Generate signal
				try:
					signal = strategy.generate_signal(backtest_data, {"current_bar": current_bar})
				except Exception as e:
					print(f"Signal generation error: {e}")
					continue
				
				# Execute trades
				if signal and signal.action in ["BUY", "SELL"]:
					trade_result = self._execute_backtest_trade(
						signal, current_bar, balance, positions
					)
					
					if trade_result:
						trade_history.append(trade_result)
						balance += trade_result['pnl']
						
						# Update max drawdown
						if balance > max_balance:
							max_balance = balance
						
						current_drawdown = (max_balance - balance) / max_balance
						max_drawdown = max(max_drawdown, current_drawdown)
				
				# Update equity curve
				equity_curve.append(balance)
		else:
			# Single timeframe strategy processing
			for i in range(50, len(backtest_data)):
				current_bar = backtest_data.iloc[i]
				historical_data = backtest_data.iloc[:i+1]
				
				# Generate signal
				try:
					signal = strategy.generate_signal(historical_data, {"current_bar": current_bar})
				except Exception as e:
					print(f"Signal generation error: {e}")
					continue
				
				# Execute trades
				if signal and signal.action in ["BUY", "SELL"]:
					trade_result = self._execute_backtest_trade(
						signal, current_bar, balance, positions
					)
					
					if trade_result:
						trade_history.append(trade_result)
						balance += trade_result['pnl']
						
						# Update max drawdown
						if balance > max_balance:
							max_balance = balance
						
						current_drawdown = (max_balance - balance) / max_balance
						max_drawdown = max(max_drawdown, current_drawdown)
				
				# Update equity curve
				equity_curve.append(balance)
		
		# Calculate performance metrics
		performance = self._calculate_performance_metrics(
			strategy.name, trade_history, equity_curve, max_drawdown
		)
		
		return performance
	
	def _execute_backtest_trade(self, signal: TradingSignal, current_bar: pd.Series, 
							  balance: float, positions: Dict) -> Optional[Dict]:
		"""Execute a backtest trade"""
		
		action = signal.action
		entry_price = signal.entry_price
		stop_loss = signal.stop_loss
		take_profit = signal.take_profit
		
		# Calculate position size
		position_size = signal.volume  # Use signal volume
		
		# Simple trade execution
		if action == "BUY":
			# Close any existing short position
			if "SHORT" in positions:
				short_trade = positions["SHORT"]
				pnl = short_trade['entry_price'] - entry_price
				positions.pop("SHORT")
				
				return {
					"action": "CLOSE_SHORT",
					"entry_price": short_trade['entry_price'],
					"exit_price": entry_price,
					"volume": short_trade['volume'],
					"pnl": pnl * short_trade['volume'],
					"confidence": signal.confidence,
					"strategy": signal.strategy_name,
					"timestamp": current_bar.name
				}
			
			# Open long position
			positions["LONG"] = {
				"entry_price": entry_price,
				"volume": position_size,
				"confidence": signal.confidence,
				"stop_loss": stop_loss,
				"take_profit": take_profit
			}
			
			return {
				"action": "OPEN_LONG",
				"entry_price": entry_price,
				"volume": position_size,
				"pnl": 0.0,
				"confidence": signal.confidence,
				"strategy": signal.strategy_name,
				"timestamp": current_bar.name
			}
		
		elif action == "SELL":
			# Close any existing long position
			if "LONG" in positions:
				long_trade = positions["LONG"]
				pnl = entry_price - long_trade['entry_price']
				positions.pop("LONG")
				
				return {
					"action": "CLOSE_LONG",
					"entry_price": long_trade['entry_price'],
					"exit_price": entry_price,
					"volume": long_trade['volume'],
					"pnl": pnl * long_trade['volume'],
					"confidence": signal.confidence,
					"strategy": signal.strategy_name,
					"timestamp": current_bar.name
				}
			
			# Open short position
			positions["SHORT"] = {
				"entry_price": entry_price,
				"volume": position_size,
				"confidence": signal.confidence,
				"stop_loss": stop_loss,
				"take_profit": take_profit
			}
			
			return {
				"action": "OPEN_SHORT",
				"entry_price": entry_price,
				"volume": position_size,
				"pnl": 0.0,
				"confidence": signal.confidence,
				"strategy": signal.strategy_name,
				"timestamp": current_bar.name
			}
		
		return None
	
	def _calculate_performance_metrics(self, strategy_name: str, trade_history: List[Dict], 
									 equity_curve: List[float], max_drawdown: float) -> StrategyPerformance:
		"""Calculate comprehensive performance metrics"""
		
		if not trade_history:
			return StrategyPerformance(
				strategy_name=strategy_name,
				total_signals=0,
				winning_signals=0,
				losing_signals=0,
				win_rate=0.0,
				profit_factor=0.0,
				total_pnl=0.0,
				max_drawdown=0.0,
				sharpe_ratio=0.0,
				sortino_ratio=0.0,
				calmar_ratio=0.0,
				avg_risk_reward=0.0,
				avg_confidence=0.0,
				avg_trade_duration=0.0,
				best_period="N/A",
				worst_period="N/A",
				regime_performance={},
				monthly_returns={},
				equity_curve=equity_curve,
				trade_history=trade_history
			)
		
		# Basic metrics
		total_signals = len(trade_history)
		winning_signals = len([t for t in trade_history if t['pnl'] > 0])
		losing_signals = len([t for t in trade_history if t['pnl'] < 0])
		win_rate = winning_signals / total_signals if total_signals > 0 else 0.0
		
		# P&L metrics
		total_pnl = sum(t['pnl'] for t in trade_history)
		total_wins = sum(t['pnl'] for t in trade_history if t['pnl'] > 0)
		total_losses = abs(sum(t['pnl'] for t in trade_history if t['pnl'] < 0))
		profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
		
		# Risk metrics
		equity_series = pd.Series(equity_curve)
		returns = equity_series.pct_change().dropna()
		
		if len(returns) > 1:
			sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
			
			# Sortino ratio
			downside_returns = returns[returns < 0]
			sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0.0
			
			# Calmar ratio
			annual_return = (equity_curve[-1] / equity_curve[0]) ** (252 / len(equity_curve)) - 1
			calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
		else:
			sharpe_ratio = sortino_ratio = calmar_ratio = 0.0
		
		# Additional metrics
		avg_confidence = np.mean([t['confidence'] for t in trade_history])
		avg_risk_reward = 0.0  # Would need to calculate from signal data
		
		# Monthly returns
		monthly_returns = self._calculate_monthly_returns(equity_curve)
		
		# Regime performance
		regime_performance = self._calculate_regime_performance(trade_history)
		
		# Best/worst periods
		best_period, worst_period = self._find_best_worst_periods(monthly_returns)
		
		return StrategyPerformance(
			strategy_name=strategy_name,
			total_signals=total_signals,
			winning_signals=winning_signals,
			losing_signals=losing_signals,
			win_rate=win_rate,
			profit_factor=profit_factor,
			total_pnl=total_pnl,
			max_drawdown=max_drawdown,
			sharpe_ratio=sharpe_ratio,
			sortino_ratio=sortino_ratio,
			calmar_ratio=calmar_ratio,
			avg_risk_reward=avg_risk_reward,
			avg_confidence=avg_confidence,
			avg_trade_duration=0.0,  # Would need to calculate from trade data
			best_period=best_period,
			worst_period=worst_period,
			regime_performance=regime_performance,
			monthly_returns=monthly_returns,
			equity_curve=equity_curve,
			trade_history=trade_history
		)
	
	def _calculate_monthly_returns(self, equity_curve: List[float]) -> Dict[str, float]:
		"""Calculate monthly returns"""
		
		if len(equity_curve) < 2:
			return {}
		
		# Create DataFrame with dates (simplified)
		dates = pd.date_range(start='2023-01-01', periods=len(equity_curve), freq='H')
		df = pd.DataFrame({'equity': equity_curve}, index=dates)
		
		# Resample to monthly and calculate returns
		monthly_equity = df.resample('M').last()
		monthly_returns = monthly_equity.pct_change().dropna()
		
		return {
			month.strftime('%Y-%m'): float(return_val)
			for month, return_val in monthly_returns['equity'].items()
		}
	
	def _calculate_regime_performance(self, trade_history: List[Dict]) -> Dict[str, Dict]:
		"""Calculate performance by market regime"""
		
		# Simplified regime analysis
		regimes = {
			"trending": {"trades": 0, "pnl": 0.0},
			"ranging": {"trades": 0, "pnl": 0.0},
			"volatile": {"trades": 0, "pnl": 0.0}
		}
		
		for trade in trade_history:
			# Simple regime classification based on trade characteristics
			if abs(trade['pnl']) > 100:  # Large P&L = volatile
				regimes["volatile"]["trades"] += 1
				regimes["volatile"]["pnl"] += trade['pnl']
			elif trade['pnl'] > 0:  # Profitable = trending
				regimes["trending"]["trades"] += 1
				regimes["trending"]["pnl"] += trade['pnl']
			else:  # Loss = ranging
				regimes["ranging"]["trades"] += 1
				regimes["ranging"]["pnl"] += trade['pnl']
		
		return regimes
	
	def _find_best_worst_periods(self, monthly_returns: Dict[str, float]) -> Tuple[str, str]:
		"""Find best and worst performing periods"""
		
		if not monthly_returns:
			return "N/A", "N/A"
		
		best_period = max(monthly_returns.items(), key=lambda x: x[1])[0]
		worst_period = min(monthly_returns.items(), key=lambda x: x[1])[0]
		
		return best_period, worst_period
	
	def optimize_strategy_parameters(self, strategy_name: str, data: Dict[str, pd.DataFrame], 
								  parameter_grid: Dict) -> Dict:
		"""
		Optimize strategy parameters using grid search
		"""
		
		if strategy_name not in self.strategies:
			raise ValueError(f"Strategy {strategy_name} not found")
		
		strategy = self.strategies[strategy_name]
		best_params = None
		best_performance = -float('inf')
		optimization_results = []
		
		print(f"Optimizing {strategy_name} parameters...")
		
		# Generate parameter combinations
		param_combinations = list(ParameterGrid(parameter_grid))
		
		for i, params in enumerate(param_combinations):
			print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
			
			# Update strategy parameters
			strategy.update_parameters(params)
			
			# Run backtest
			try:
				performance = self._backtest_strategy(strategy, data, 500)  # Smaller sample for optimization
				
				# Use Sharpe ratio as optimization metric
				optimization_score = performance.sharpe_ratio
				
				optimization_results.append({
					"parameters": params,
					"sharpe_ratio": performance.sharpe_ratio,
					"win_rate": performance.win_rate,
					"profit_factor": performance.profit_factor,
					"total_pnl": performance.total_pnl,
					"max_drawdown": performance.max_drawdown
				})
				
				# Update best parameters
				if optimization_score > best_performance:
					best_performance = optimization_score
					best_params = params
				
			except Exception as e:
				print(f"Error testing parameters {params}: {e}")
				continue
		
		# Store optimization results
		self.optimization_results[strategy_name] = {
			"best_parameters": best_params,
			"best_performance": best_performance,
			"all_results": optimization_results
		}
		
		return self.optimization_results[strategy_name]
	
	def generate_performance_report(self, output_path: str = "strategy_performance_report.html"):
		"""Generate comprehensive performance report"""
		
		if not self.performance_results:
			print("No performance results available. Run backtest first.")
			return
		
		# Create HTML report
		html_content = self._create_performance_html_report()
		
		with open(output_path, 'w') as f:
			f.write(html_content)
		
		print(f"Performance report saved to: {output_path}")
	
	def _create_performance_html_report(self) -> str:
		"""Create HTML performance report"""
		
		html = """
		<!DOCTYPE html>
		<html>
		<head>
			<title>Strategy Performance Report</title>
			<style>
				body { font-family: Arial, sans-serif; margin: 20px; }
				table { border-collapse: collapse; width: 100%; margin: 20px 0; }
				th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
				th { background-color: #f2f2f2; }
				.metric { background-color: #e8f4fd; }
				.positive { color: green; font-weight: bold; }
				.negative { color: red; font-weight: bold; }
				.rank-1 { background-color: #d4edda; }
				.rank-2 { background-color: #fff3cd; }
				.rank-3 { background-color: #f8d7da; }
			</style>
		</head>
		<body>
			<h1>Strategy Performance Report</h1>
			<p>Generated on: {}</p>
		""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		
		# Strategy comparison table
		html += "<h2>Strategy Comparison</h2>"
		html += "<table>"
		html += "<tr><th>Strategy</th><th>Total Signals</th><th>Win Rate</th><th>Profit Factor</th><th>Sharpe Ratio</th><th>Max Drawdown</th><th>Total P&L</th></tr>"
		
		# Sort strategies by Sharpe ratio
		sorted_strategies = sorted(
			self.performance_results.items(),
			key=lambda x: x[1].sharpe_ratio,
			reverse=True
		)
		
		for i, (strategy_name, performance) in enumerate(sorted_strategies):
			rank_class = f"rank-{min(i+1, 3)}" if i < 3 else ""
			
			html += f"<tr class='{rank_class}'>"
			html += f"<td>{strategy_name}</td>"
			html += f"<td>{performance.total_signals}</td>"
			html += f"<td>{performance.win_rate:.1%}</td>"
			html += f"<td>{performance.profit_factor:.2f}</td>"
			html += f"<td>{performance.sharpe_ratio:.2f}</td>"
			html += f"<td>{performance.max_drawdown:.1%}</td>"
			html += f"<td class={'positive' if performance.total_pnl > 0 else 'negative'}>${performance.total_pnl:.2f}</td>"
			html += "</tr>"
		
		html += "</table>"
		
		# Detailed strategy analysis
		html += "<h2>Detailed Strategy Analysis</h2>"
		
		for strategy_name, performance in self.performance_results.items():
			html += f"<h3>{strategy_name}</h3>"
			html += "<table>"
			html += "<tr><th>Metric</th><th>Value</th></tr>"
			
			metrics = [
				("Total Signals", performance.total_signals),
				("Winning Signals", performance.winning_signals),
				("Losing Signals", performance.losing_signals),
				("Win Rate", f"{performance.win_rate:.1%}"),
				("Profit Factor", f"{performance.profit_factor:.2f}"),
				("Total P&L", f"${performance.total_pnl:.2f}"),
				("Max Drawdown", f"{performance.max_drawdown:.1%}"),
				("Sharpe Ratio", f"{performance.sharpe_ratio:.2f}"),
				("Sortino Ratio", f"{performance.sortino_ratio:.2f}"),
				("Calmar Ratio", f"{performance.calmar_ratio:.2f}"),
				("Avg Confidence", f"{performance.avg_confidence:.2f}"),
				("Best Period", performance.best_period),
				("Worst Period", performance.worst_period)
			]
			
			for metric_name, metric_value in metrics:
				html += f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>"
			
			html += "</table>"
		
		# Optimization results
		if self.optimization_results:
			html += "<h2>Parameter Optimization Results</h2>"
			
			for strategy_name, opt_results in self.optimization_results.items():
				html += f"<h3>{strategy_name} Optimization</h3>"
				html += f"<p><strong>Best Parameters:</strong> {opt_results['best_parameters']}</p>"
				html += f"<p><strong>Best Sharpe Ratio:</strong> {opt_results['best_performance']:.2f}</p>"
		
		html += "</body></html>"
		
		return html
	
	def get_strategy_rankings(self) -> List[Tuple[str, float]]:
		"""Get strategy rankings by Sharpe ratio"""
		
		if not self.performance_results:
			return []
		
		rankings = [
			(strategy_name, performance.sharpe_ratio)
			for strategy_name, performance in self.performance_results.items()
		]
		
		return sorted(rankings, key=lambda x: x[1], reverse=True)
	
	def get_best_strategy(self) -> Optional[str]:
		"""Get the best performing strategy"""
		
		rankings = self.get_strategy_rankings()
		return rankings[0][0] if rankings else None
	
	def export_results(self, output_path: str = "strategy_results.json"):
		"""Export results to JSON"""
		
		results = {
			"performance_results": {
				name: {
					"strategy_name": perf.strategy_name,
					"total_signals": perf.total_signals,
					"winning_signals": perf.winning_signals,
					"losing_signals": perf.losing_signals,
					"win_rate": perf.win_rate,
					"profit_factor": perf.profit_factor,
					"total_pnl": perf.total_pnl,
					"max_drawdown": perf.max_drawdown,
					"sharpe_ratio": perf.sharpe_ratio,
					"sortino_ratio": perf.sortino_ratio,
					"calmar_ratio": perf.calmar_ratio,
					"avg_confidence": perf.avg_confidence,
					"best_period": perf.best_period,
					"worst_period": perf.worst_period
				}
				for name, perf in self.performance_results.items()
			},
			"optimization_results": self.optimization_results,
			"rankings": self.get_strategy_rankings(),
			"generated_at": datetime.now().isoformat()
		}
		
		with open(output_path, 'w') as f:
			json.dump(results, f, indent=2, default=str)
		
		print(f"Results exported to: {output_path}")
		return results