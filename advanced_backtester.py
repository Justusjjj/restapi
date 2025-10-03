import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
	"""Comprehensive backtest result"""
	total_trades: int
	winning_trades: int
	losing_trades: int
	win_rate: float
	profit_factor: float
	total_pnl: float
	max_drawdown: float
	sharpe_ratio: float
	sortino_ratio: float
	calmar_ratio: float
	avg_trade_duration: float
	avg_win: float
	avg_loss: float
	largest_win: float
	largest_loss: float
	trade_history: List[Dict]
	equity_curve: List[float]
	drawdown_curve: List[float]
	monthly_returns: Dict[str, float]
	regime_performance: Dict[str, Dict]


class AdvancedBacktester:
	"""
	Advanced backtesting system with walk-forward analysis, regime detection, and comprehensive metrics.
	"""
	
	def __init__(self, symbols: List[str], lookback_period: int = 1000):
		self.symbols = symbols
		self.lookback_period = lookback_period
		self.walk_forward_windows = []
		self.regime_detector = None
		self.performance_metrics = {}
		
	def setup_walk_forward_analysis(self, 
								   training_period: int = 252,  # 1 year
								   testing_period: int = 63,   # 3 months
								   step_size: int = 21):       # 1 month
		"""
		Setup walk-forward analysis parameters
		"""
		
		self.training_period = training_period
		self.testing_period = testing_period
		self.step_size = step_size
		
		print(f"Walk-forward analysis setup:")
		print(f"  Training period: {training_period} days")
		print(f"  Testing period: {testing_period} days")
		print(f"  Step size: {step_size} days")
	
	def run_walk_forward_backtest(self, 
								mt5_connector,
								strategy_function,
								model_path: Optional[str] = None) -> Dict:
		"""
		Run walk-forward backtesting analysis
		"""
		
		# Fetch historical data
		historical_data = self._fetch_historical_data(mt5_connector)
		
		if historical_data.empty:
			return {"error": "No historical data available"}
		
		# Setup walk-forward windows
		self._setup_walk_forward_windows(historical_data)
		
		# Run backtests for each window
		walk_forward_results = []
		
		for i, window in enumerate(self.walk_forward_windows):
			print(f"Processing window {i+1}/{len(self.walk_forward_windows)}")
			
			# Extract training and testing data
			train_data = historical_data.iloc[window['train_start']:window['train_end']]
			test_data = historical_data.iloc[window['test_start']:window['test_end']]
			
			# Run backtest for this window
			window_result = self._run_single_backtest(
				train_data, test_data, strategy_function, model_path
			)
			
			window_result['window'] = i
			window_result['train_period'] = f"{train_data.index[0]} to {train_data.index[-1]}"
			window_result['test_period'] = f"{test_data.index[0]} to {test_data.index[-1]}"
			
			walk_forward_results.append(window_result)
		
		# Aggregate results
		aggregated_results = self._aggregate_walk_forward_results(walk_forward_results)
		
		return {
			"walk_forward_results": walk_forward_results,
			"aggregated_results": aggregated_results,
			"total_windows": len(self.walk_forward_windows),
			"analysis_period": f"{historical_data.index[0]} to {historical_data.index[-1]}"
		}
	
	def _fetch_historical_data(self, mt5_connector) -> pd.DataFrame:
		"""Fetch historical data for all symbols"""
		
		all_data = {}
		
		for symbol in self.symbols:
			try:
				data = mt5_connector.fetch_rates(symbol, "H1", self.lookback_period)
				if not data.empty:
					all_data[symbol] = data
			except Exception as e:
				print(f"Error fetching data for {symbol}: {e}")
				continue
		
		if not all_data:
			return pd.DataFrame()
		
		# Use the first symbol as primary (assuming all have same index)
		primary_symbol = list(all_data.keys())[0]
		historical_data = all_data[primary_symbol].copy()
		
		# Add other symbols as additional columns
		for symbol, data in all_data.items():
			if symbol != primary_symbol:
				historical_data[f"{symbol}_close"] = data['close']
				historical_data[f"{symbol}_high"] = data['high']
				historical_data[f"{symbol}_low"] = data['low']
		
		return historical_data
	
	def _setup_walk_forward_windows(self, data: pd.DataFrame):
		"""Setup walk-forward analysis windows"""
		
		total_length = len(data)
		self.walk_forward_windows = []
		
		start_idx = 0
		
		while start_idx + self.training_period + self.testing_period <= total_length:
			train_start = start_idx
			train_end = start_idx + self.training_period
			test_start = train_end
			test_end = test_start + self.testing_period
			
			self.walk_forward_windows.append({
				'train_start': train_start,
				'train_end': train_end,
				'test_start': test_start,
				'test_end': test_end
			})
			
			start_idx += self.step_size
	
	def _run_single_backtest(self, 
						   train_data: pd.DataFrame,
						   test_data: pd.DataFrame,
						   strategy_function,
						   model_path: Optional[str] = None) -> Dict:
		"""Run backtest for a single window"""
		
		# Initialize backtest variables
		balance = 10000.0
		equity_curve = [balance]
		trade_history = []
		positions = {}
		max_balance = balance
		max_drawdown = 0.0
		
		# Process each bar in test data
		for i, (timestamp, bar) in enumerate(test_data.iterrows()):
			current_price = bar['close']
			
			# Get strategy signal
			try:
				signal = strategy_function(
					train_data.iloc[:i] if i > 0 else train_data.iloc[:1],
					bar,
					model_path
				)
			except Exception as e:
				print(f"Strategy error at {timestamp}: {e}")
				signal = {"action": "HOLD", "confidence": 0.0}
			
			# Execute trades based on signal
			if signal.get("action") in ["BUY", "SELL"]:
				trade_result = self._execute_trade(
					signal, current_price, balance, positions
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
		metrics = self._calculate_performance_metrics(
			trade_history, equity_curve, max_drawdown
		)
		
		return {
			"metrics": metrics,
			"trade_history": trade_history,
			"equity_curve": equity_curve,
			"final_balance": balance,
			"initial_balance": 10000.0
		}
	
	def _execute_trade(self, signal: Dict, price: float, 
					  balance: float, positions: Dict) -> Optional[Dict]:
		"""Execute a trade based on signal"""
		
		action = signal.get("action")
		confidence = signal.get("confidence", 0.5)
		volume = signal.get("volume", 0.01)
		
		# Simple trade execution (in real implementation, use advanced order manager)
		if action == "BUY":
			# Close any existing short position
			if "SHORT" in positions:
				short_trade = positions["SHORT"]
				pnl = short_trade['entry_price'] - price
				positions.pop("SHORT")
				
				return {
					"action": "CLOSE_SHORT",
					"entry_price": short_trade['entry_price'],
					"exit_price": price,
					"volume": short_trade['volume'],
					"pnl": pnl * short_trade['volume'],
					"confidence": confidence
				}
			
			# Open long position
			positions["LONG"] = {
				"entry_price": price,
				"volume": volume,
				"confidence": confidence
			}
			
			return {
				"action": "OPEN_LONG",
				"entry_price": price,
				"volume": volume,
				"pnl": 0.0,
				"confidence": confidence
			}
		
		elif action == "SELL":
			# Close any existing long position
			if "LONG" in positions:
				long_trade = positions["LONG"]
				pnl = price - long_trade['entry_price']
				positions.pop("LONG")
				
				return {
					"action": "CLOSE_LONG",
					"entry_price": long_trade['entry_price'],
					"exit_price": price,
					"volume": long_trade['volume'],
					"pnl": pnl * long_trade['volume'],
					"confidence": confidence
				}
			
			# Open short position
			positions["SHORT"] = {
				"entry_price": price,
				"volume": volume,
				"confidence": confidence
			}
			
			return {
				"action": "OPEN_SHORT",
				"entry_price": price,
				"volume": volume,
				"pnl": 0.0,
				"confidence": confidence
			}
		
		return None
	
	def _calculate_performance_metrics(self, 
									  trade_history: List[Dict],
									  equity_curve: List[float],
									  max_drawdown: float) -> Dict:
		"""Calculate comprehensive performance metrics"""
		
		if not trade_history:
			return {
				"total_trades": 0,
				"win_rate": 0.0,
				"profit_factor": 0.0,
				"total_pnl": 0.0,
				"max_drawdown": 0.0,
				"sharpe_ratio": 0.0,
				"sortino_ratio": 0.0,
				"calmar_ratio": 0.0
			}
		
		# Basic trade statistics
		total_trades = len(trade_history)
		winning_trades = [t for t in trade_history if t['pnl'] > 0]
		losing_trades = [t for t in trade_history if t['pnl'] < 0]
		
		win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
		
		# P&L statistics
		total_pnl = sum(t['pnl'] for t in trade_history)
		total_wins = sum(t['pnl'] for t in winning_trades)
		total_losses = abs(sum(t['pnl'] for t in losing_trades))
		
		profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
		
		# Risk metrics
		equity_series = pd.Series(equity_curve)
		returns = equity_series.pct_change().dropna()
		
		if len(returns) > 1:
			sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
			
			# Sortino ratio (downside deviation)
			downside_returns = returns[returns < 0]
			sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0.0
			
			# Calmar ratio
			annual_return = (equity_curve[-1] / equity_curve[0]) ** (252 / len(equity_curve)) - 1
			calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
		else:
			sharpe_ratio = sortino_ratio = calmar_ratio = 0.0
		
		return {
			"total_trades": total_trades,
			"winning_trades": len(winning_trades),
			"losing_trades": len(losing_trades),
			"win_rate": win_rate,
			"profit_factor": profit_factor,
			"total_pnl": total_pnl,
			"max_drawdown": max_drawdown,
			"sharpe_ratio": sharpe_ratio,
			"sortino_ratio": sortino_ratio,
			"calmar_ratio": calmar_ratio,
			"avg_win": np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0,
			"avg_loss": np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0,
			"largest_win": max([t['pnl'] for t in winning_trades]) if winning_trades else 0.0,
			"largest_loss": min([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
		}
	
	def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
		"""Aggregate results from all walk-forward windows"""
		
		if not results:
			return {}
		
		# Extract metrics from all windows
		all_metrics = [r['metrics'] for r in results]
		
		# Calculate aggregated statistics
		aggregated = {}
		
		for metric in all_metrics[0].keys():
			values = [m[metric] for m in all_metrics if not np.isnan(m[metric]) and not np.isinf(m[metric])]
			
			if values:
				aggregated[metric] = {
					"mean": np.mean(values),
					"std": np.std(values),
					"min": np.min(values),
					"max": np.max(values),
					"median": np.median(values)
				}
		
		# Calculate stability metrics
		win_rates = [m['win_rate'] for m in all_metrics]
		sharpe_ratios = [m['sharpe_ratio'] for m in all_metrics]
		
		aggregated['stability'] = {
			"win_rate_consistency": 1.0 - np.std(win_rates) if win_rates else 0.0,
			"sharpe_consistency": 1.0 - np.std(sharpe_ratios) if sharpe_ratios else 0.0,
			"positive_windows": sum(1 for m in all_metrics if m['total_pnl'] > 0),
			"total_windows": len(all_metrics)
		}
		
		return aggregated
	
	def run_regime_aware_backtest(self, 
								mt5_connector,
								strategy_function,
								regime_detector,
								model_path: Optional[str] = None) -> Dict:
		"""
		Run backtest with regime awareness
		"""
		
		# Fetch historical data
		historical_data = self._fetch_historical_data(mt5_connector)
		
		if historical_data.empty:
			return {"error": "No historical data available"}
		
		# Detect regimes
		regimes = self._detect_regimes(historical_data, regime_detector)
		
		# Run backtest for each regime
		regime_results = {}
		
		for regime_name, regime_periods in regimes.items():
			print(f"Backtesting regime: {regime_name}")
			
			regime_trades = []
			regime_equity = [10000.0]
			
			for start_idx, end_idx in regime_periods:
				regime_data = historical_data.iloc[start_idx:end_idx]
				
				# Run backtest for this regime period
				period_result = self._run_single_backtest(
					regime_data.iloc[:len(regime_data)//2],  # Training
					regime_data.iloc[len(regime_data)//2:],  # Testing
					strategy_function,
					model_path
				)
				
				regime_trades.extend(period_result['trade_history'])
				regime_equity.extend(period_result['equity_curve'][1:])
			
			# Calculate regime-specific metrics
			regime_metrics = self._calculate_performance_metrics(
				regime_trades, regime_equity, 0.0
			)
			
			regime_results[regime_name] = {
				"metrics": regime_metrics,
				"trade_count": len(regime_trades),
				"periods": len(regime_periods)
			}
		
		return {
			"regime_results": regime_results,
			"total_regimes": len(regimes),
			"analysis_period": f"{historical_data.index[0]} to {historical_data.index[-1]}"
		}
	
	def _detect_regimes(self, data: pd.DataFrame, regime_detector) -> Dict:
		"""Detect market regimes in historical data"""
		
		regimes = {}
		current_regime = None
		regime_start = 0
		
		for i in range(len(data)):
			# Get regime for current period
			price_data = data.iloc[max(0, i-100):i+1]['close'].values
			
			if len(price_data) > 50:
				regime = regime_detector.detect_regime(price_data.reshape(-1, 1))
				
				if regime != current_regime:
					# End previous regime
					if current_regime is not None:
						if current_regime not in regimes:
							regimes[current_regime] = []
						regimes[current_regime].append((regime_start, i))
					
					# Start new regime
					current_regime = regime
					regime_start = i
		
		# Add final regime
		if current_regime is not None:
			if current_regime not in regimes:
				regimes[current_regime] = []
			regimes[current_regime].append((regime_start, len(data)))
		
		return regimes
	
	def generate_backtest_report(self, results: Dict, output_path: str = "backtest_report.html"):
		"""Generate comprehensive backtest report"""
		
		# Create HTML report
		html_content = self._create_html_report(results)
		
		with open(output_path, 'w') as f:
			f.write(html_content)
		
		print(f"Backtest report saved to: {output_path}")
	
	def _create_html_report(self, results: Dict) -> str:
		"""Create HTML report content"""
		
		html = """
		<!DOCTYPE html>
		<html>
		<head>
			<title>Advanced Backtest Report</title>
			<style>
				body { font-family: Arial, sans-serif; margin: 20px; }
				table { border-collapse: collapse; width: 100%; margin: 20px 0; }
				th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
				th { background-color: #f2f2f2; }
				.metric { background-color: #e8f4fd; }
				.positive { color: green; }
				.negative { color: red; }
			</style>
		</head>
		<body>
			<h1>Advanced Backtest Report</h1>
		"""
		
		# Add aggregated results
		if "aggregated_results" in results:
			html += "<h2>Walk-Forward Analysis Results</h2>"
			html += "<table>"
			html += "<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th></tr>"
			
			for metric, stats in results["aggregated_results"].items():
				if isinstance(stats, dict) and "mean" in stats:
					html += f"<tr><td>{metric}</td>"
					html += f"<td>{stats['mean']:.4f}</td>"
					html += f"<td>{stats['std']:.4f}</td>"
					html += f"<td>{stats['min']:.4f}</td>"
					html += f"<td>{stats['max']:.4f}</td>"
					html += f"<td>{stats['median']:.4f}</td></tr>"
			
			html += "</table>"
		
		# Add regime results
		if "regime_results" in results:
			html += "<h2>Regime-Specific Performance</h2>"
			html += "<table>"
			html += "<tr><th>Regime</th><th>Win Rate</th><th>Sharpe Ratio</th><th>Total P&L</th><th>Trade Count</th></tr>"
			
			for regime, data in results["regime_results"].items():
				metrics = data["metrics"]
				html += f"<tr><td>{regime}</td>"
				html += f"<td>{metrics['win_rate']:.2%}</td>"
				html += f"<td>{metrics['sharpe_ratio']:.2f}</td>"
				html += f"<td class={'positive' if metrics['total_pnl'] > 0 else 'negative'}>{metrics['total_pnl']:.2f}</td>"
				html += f"<td>{data['trade_count']}</td></tr>"
			
			html += "</table>"
		
		html += "</body></html>"
		
		return html