import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import cvxpy as cp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class AdvancedPortfolioOptimizer:
	"""
	Advanced portfolio optimization with risk parity, correlation analysis, and dynamic hedging.
	"""
	
	def __init__(self, symbols: List[str], lookback_period: int = 252):
		self.symbols = symbols
		self.lookback_period = lookback_period
		self.correlation_matrix = None
		self.covariance_matrix = None
		self.expected_returns = None
		self.risk_parity_weights = None
		self.hedge_ratios = {}
		
	def fetch_multi_asset_data(self, mt5_connector) -> pd.DataFrame:
		"""Fetch data for all symbols"""
		
		all_data = {}
		
		for symbol in self.symbols:
			try:
				data = mt5_connector.fetch_rates(symbol, "H1", self.lookback_period)
				if not data.empty:
					all_data[symbol] = data['close'].values
			except Exception as e:
				print(f"Error fetching data for {symbol}: {e}")
				continue
		
		# Create DataFrame with aligned data
		max_length = max(len(data) for data in all_data.values()) if all_data else 0
		
		df = pd.DataFrame()
		for symbol, prices in all_data.items():
			# Pad shorter series with NaN
			padded_prices = np.full(max_length, np.nan)
			padded_prices[:len(prices)] = prices
			df[symbol] = padded_prices
		
		# Forward fill missing values
		df = df.fillna(method='ffill').dropna()
		
		return df
	
	def calculate_correlation_analysis(self, price_data: pd.DataFrame) -> Dict:
		"""Calculate comprehensive correlation analysis"""
		
		# Calculate returns
		returns = price_data.pct_change().dropna()
		
		# Correlation matrix
		self.correlation_matrix = returns.corr()
		
		# Covariance matrix
		self.covariance_matrix = returns.cov()
		
		# Expected returns (annualized)
		self.expected_returns = returns.mean() * 252
		
		# Correlation clusters
		clusters = self._identify_correlation_clusters()
		
		# Hedge ratios
		self.hedge_ratios = self._calculate_hedge_ratios(returns)
		
		return {
			"correlation_matrix": self.correlation_matrix.to_dict(),
			"covariance_matrix": self.covariance_matrix.to_dict(),
			"expected_returns": self.expected_returns.to_dict(),
			"correlation_clusters": clusters,
			"hedge_ratios": self.hedge_ratios,
			"avg_correlation": self.correlation_matrix.mean().mean(),
			"max_correlation": self.correlation_matrix.max().max(),
			"min_correlation": self.correlation_matrix.min().min()
		}
	
	def _identify_correlation_clusters(self) -> Dict:
		"""Identify highly correlated asset clusters"""
		
		clusters = {}
		processed = set()
		
		for symbol1 in self.correlation_matrix.columns:
			if symbol1 in processed:
				continue
			
			cluster = [symbol1]
			processed.add(symbol1)
			
			for symbol2 in self.correlation_matrix.columns:
				if symbol2 in processed:
					continue
				
				correlation = abs(self.correlation_matrix.loc[symbol1, symbol2])
				if correlation > 0.7:  # High correlation threshold
					cluster.append(symbol2)
					processed.add(symbol2)
			
			if len(cluster) > 1:
				clusters[f"cluster_{len(clusters)}"] = cluster
		
		return clusters
	
	def _calculate_hedge_ratios(self, returns: pd.DataFrame) -> Dict:
		"""Calculate optimal hedge ratios between pairs"""
		
		hedge_ratios = {}
		
		for i, symbol1 in enumerate(returns.columns):
			for j, symbol2 in enumerate(returns.columns):
				if i >= j:  # Avoid duplicates
					continue
				
				# Calculate hedge ratio using regression
				y = returns[symbol1].values
				x = returns[symbol2].values
				
				# Remove NaN values
				mask = ~(np.isnan(x) | np.isnan(y))
				x_clean = x[mask]
				y_clean = y[mask]
				
				if len(x_clean) < 10:  # Need minimum data points
					continue
				
				# Linear regression: y = alpha + beta * x
				beta = np.cov(x_clean, y_clean)[0, 1] / np.var(x_clean)
				alpha = np.mean(y_clean) - beta * np.mean(x_clean)
				
				# R-squared
				y_pred = alpha + beta * x_clean
				r_squared = 1 - np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2)
				
				if r_squared > 0.5:  # Only store significant relationships
					hedge_ratios[f"{symbol1}_{symbol2}"] = {
						"hedge_ratio": beta,
						"alpha": alpha,
						"r_squared": r_squared,
						"correlation": np.corrcoef(x_clean, y_clean)[0, 1]
					}
		
		return hedge_ratios
	
	def optimize_risk_parity(self, price_data: pd.DataFrame) -> Dict:
		"""Optimize portfolio using risk parity approach"""
		
		returns = price_data.pct_change().dropna()
		
		# Calculate covariance matrix
		cov_matrix = returns.cov().values
		
		# Risk parity optimization
		n_assets = len(self.symbols)
		weights = cp.Variable(n_assets)
		
		# Risk parity objective: minimize sum of squared deviations from equal risk contribution
		risk_contrib = cp.multiply(weights, cov_matrix @ weights)
		target_risk = cp.sum(risk_contrib) / n_assets
		
		objective = cp.Minimize(cp.sum_squares(risk_contrib - target_risk))
		
		# Constraints
		constraints = [
			cp.sum(weights) == 1,  # Weights sum to 1
			weights >= 0,  # No short selling
			weights <= 0.4  # Max 40% in any single asset
		]
		
		# Solve optimization
		problem = cp.Problem(objective, constraints)
		problem.solve()
		
		if problem.status == cp.OPTIMAL:
			self.risk_parity_weights = dict(zip(self.symbols, weights.value))
			
			# Calculate risk contributions
			risk_contributions = {}
			for i, symbol in enumerate(self.symbols):
				risk_contributions[symbol] = float(weights.value[i] * (cov_matrix[i] @ weights.value))
			
			return {
				"weights": self.risk_parity_weights,
				"risk_contributions": risk_contributions,
				"total_risk": float(np.sum(risk_contributions.values())),
				"optimization_status": "optimal"
			}
		else:
			return {
				"weights": {symbol: 1.0/len(self.symbols) for symbol in self.symbols},
				"risk_contributions": {},
				"total_risk": 0.0,
				"optimization_status": "failed"
			}
	
	def optimize_mean_variance(self, price_data: pd.DataFrame, 
							  risk_aversion: float = 1.0) -> Dict:
		"""Optimize portfolio using mean-variance approach"""
		
		returns = price_data.pct_change().dropna()
		
		# Calculate expected returns and covariance
		mu = returns.mean().values * 252  # Annualized
		Sigma = returns.cov().values * 252  # Annualized
		
		n_assets = len(self.symbols)
		weights = cp.Variable(n_assets)
		
		# Mean-variance objective: maximize return - risk_aversion * variance
		expected_return = mu.T @ weights
		portfolio_variance = cp.quad_form(weights, Sigma)
		
		objective = cp.Maximize(expected_return - risk_aversion * portfolio_variance)
		
		# Constraints
		constraints = [
			cp.sum(weights) == 1,
			weights >= 0,
			weights <= 0.4
		]
		
		# Solve
		problem = cp.Problem(objective, constraints)
		problem.solve()
		
		if problem.status == cp.OPTIMAL:
			mv_weights = dict(zip(self.symbols, weights.value))
			
			return {
				"weights": mv_weights,
				"expected_return": float(expected_return.value),
				"portfolio_variance": float(portfolio_variance.value),
				"sharpe_ratio": float(expected_return.value / np.sqrt(portfolio_variance.value)),
				"optimization_status": "optimal"
			}
		else:
			return {
				"weights": {symbol: 1.0/len(self.symbols) for symbol in self.symbols},
				"expected_return": 0.0,
				"portfolio_variance": 0.0,
				"sharpe_ratio": 0.0,
				"optimization_status": "failed"
			}
	
	def calculate_portfolio_metrics(self, weights: Dict, 
								  price_data: pd.DataFrame) -> Dict:
		"""Calculate comprehensive portfolio metrics"""
		
		returns = price_data.pct_change().dropna()
		
		# Portfolio returns
		portfolio_returns = pd.Series(0, index=returns.index)
		for symbol, weight in weights.items():
			if symbol in returns.columns:
				portfolio_returns += weight * returns[symbol]
		
		# Calculate metrics
		annual_return = portfolio_returns.mean() * 252
		annual_volatility = portfolio_returns.std() * np.sqrt(252)
		sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
		
		# Drawdown analysis
		cumulative_returns = (1 + portfolio_returns).cumprod()
		running_max = cumulative_returns.expanding().max()
		drawdown = (cumulative_returns - running_max) / running_max
		max_drawdown = drawdown.min()
		
		# VaR and CVaR (95% confidence)
		var_95 = np.percentile(portfolio_returns, 5)
		cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
		
		return {
			"annual_return": annual_return,
			"annual_volatility": annual_volatility,
			"sharpe_ratio": sharpe_ratio,
			"max_drawdown": max_drawdown,
			"var_95": var_95,
			"cvar_95": cvar_95,
			"skewness": portfolio_returns.skew(),
			"kurtosis": portfolio_returns.kurtosis()
		}
	
	def generate_hedging_signals(self, current_positions: Dict, 
							   price_data: pd.DataFrame) -> Dict:
		"""Generate hedging signals based on correlation analysis"""
		
		hedging_signals = {}
		
		for pair, hedge_info in self.hedge_ratios.items():
			symbol1, symbol2 = pair.split('_')
			
			if symbol1 not in price_data.columns or symbol2 not in price_data.columns:
				continue
			
			# Current prices
			price1 = price_data[symbol1].iloc[-1]
			price2 = price_data[symbol2].iloc[-1]
			
			# Current positions
			pos1 = current_positions.get(symbol1, 0)
			pos2 = current_positions.get(symbol2, 0)
			
			# Calculate hedge ratio
			hedge_ratio = hedge_info['hedge_ratio']
			r_squared = hedge_info['r_squared']
			
			# Only hedge if relationship is strong
			if r_squared > 0.6:
				# Calculate optimal hedge position
				optimal_hedge = -pos1 * hedge_ratio
				hedge_adjustment = optimal_hedge - pos2
				
				if abs(hedge_adjustment) > 0.01:  # Minimum adjustment threshold
					hedging_signals[pair] = {
						"symbol": symbol2,
						"action": "BUY" if hedge_adjustment > 0 else "SELL",
						"size": abs(hedge_adjustment),
						"hedge_ratio": hedge_ratio,
						"r_squared": r_squared,
						"reason": f"Hedge {symbol1} position"
					}
		
		return hedging_signals
	
	def rebalance_portfolio(self, current_weights: Dict, 
						  target_weights: Dict, 
						  rebalance_threshold: float = 0.05) -> Dict:
		"""Generate rebalancing signals"""
		
		rebalance_signals = {}
		
		for symbol in self.symbols:
			current_weight = current_weights.get(symbol, 0)
			target_weight = target_weights.get(symbol, 0)
			
			weight_diff = target_weight - current_weight
			
			if abs(weight_diff) > rebalance_threshold:
				rebalance_signals[symbol] = {
					"action": "BUY" if weight_diff > 0 else "SELL",
					"weight_adjustment": weight_diff,
					"current_weight": current_weight,
					"target_weight": target_weight,
					"reason": "Portfolio rebalancing"
				}
		
		return rebalance_signals


class MultiAssetCorrelationAnalyzer:
	"""
	Advanced multi-asset correlation analysis and hedging
	"""
	
	def __init__(self, symbols: List[str]):
		self.symbols = symbols
		self.correlation_history = []
		self.regime_correlations = {}
		
	def analyze_correlation_regimes(self, price_data: pd.DataFrame) -> Dict:
		"""Analyze correlation regimes over time"""
		
		returns = price_data.pct_change().dropna()
		
		# Rolling correlation analysis
		window = 60  # 60-period rolling window
		rolling_correlations = {}
		
		for i, symbol1 in enumerate(self.symbols):
			for j, symbol2 in enumerate(self.symbols):
				if i >= j:
					continue
				
				if symbol1 in returns.columns and symbol2 in returns.columns:
					rolling_corr = returns[symbol1].rolling(window).corr(returns[symbol2])
					rolling_correlations[f"{symbol1}_{symbol2}"] = rolling_corr
		
		# Identify correlation regimes
		regimes = self._identify_correlation_regimes(rolling_correlations)
		
		# Calculate regime-specific statistics
		regime_stats = {}
		for regime, periods in regimes.items():
			regime_stats[regime] = {
				"periods": len(periods),
				"avg_correlation": np.mean([rolling_correlations[pair].iloc[periods].mean() 
										   for pair in rolling_correlations.keys()]),
				"correlation_volatility": np.std([rolling_correlations[pair].iloc[periods].mean() 
												for pair in rolling_correlations.keys()])
			}
		
		return {
			"rolling_correlations": {k: v.tolist() for k, v in rolling_correlations.items()},
			"correlation_regimes": regimes,
			"regime_statistics": regime_stats,
			"current_regime": self._get_current_regime(rolling_correlations)
		}
	
	def _identify_correlation_regimes(self, rolling_correlations: Dict) -> Dict:
		"""Identify distinct correlation regimes"""
		
		regimes = {
			"high_correlation": [],
			"low_correlation": [],
			"negative_correlation": []
		}
		
		# Analyze each pair
		for pair, corr_series in rolling_correlations.items():
			corr_values = corr_series.dropna()
			
			if len(corr_values) == 0:
				continue
			
			# Define thresholds
			high_threshold = 0.7
			low_threshold = 0.3
			negative_threshold = -0.3
			
			# Classify periods
			for i, corr in enumerate(corr_values):
				if corr > high_threshold:
					regimes["high_correlation"].append(i)
				elif corr < negative_threshold:
					regimes["negative_correlation"].append(i)
				elif corr < low_threshold:
					regimes["low_correlation"].append(i)
		
		return regimes
	
	def _get_current_regime(self, rolling_correlations: Dict) -> str:
		"""Determine current correlation regime"""
		
		current_correlations = []
		
		for pair, corr_series in rolling_correlations.items():
			if not corr_series.empty:
				current_correlations.append(corr_series.iloc[-1])
		
		if not current_correlations:
			return "unknown"
		
		avg_correlation = np.mean(current_correlations)
		
		if avg_correlation > 0.7:
			return "high_correlation"
		elif avg_correlation < -0.3:
			return "negative_correlation"
		elif avg_correlation < 0.3:
			return "low_correlation"
		else:
			return "moderate_correlation"
	
	def generate_diversification_signals(self, current_positions: Dict, 
									   correlation_matrix: pd.DataFrame) -> Dict:
		"""Generate diversification signals based on correlation analysis"""
		
		signals = {}
		
		# Find highly correlated positions
		correlated_pairs = []
		for i, symbol1 in enumerate(self.symbols):
			for j, symbol2 in enumerate(self.symbols):
				if i >= j:
					continue
				
				if symbol1 in correlation_matrix.columns and symbol2 in correlation_matrix.columns:
					correlation = abs(correlation_matrix.loc[symbol1, symbol2])
					if correlation > 0.8:  # High correlation threshold
						correlated_pairs.append((symbol1, symbol2, correlation))
		
		# Generate diversification signals
		for symbol1, symbol2, correlation in correlated_pairs:
			pos1 = current_positions.get(symbol1, 0)
			pos2 = current_positions.get(symbol2, 0)
			
			# If both positions are in same direction, suggest diversification
			if pos1 * pos2 > 0 and abs(pos1) > 0.1 and abs(pos2) > 0.1:
				signals[f"diversify_{symbol1}_{symbol2}"] = {
					"action": "DIVERSIFY",
					"symbols": [symbol1, symbol2],
					"correlation": correlation,
					"reason": f"High correlation ({correlation:.2f}) detected",
					"suggestion": "Consider reducing one position or hedging"
				}
		
		return signals