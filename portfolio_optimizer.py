#!/usr/bin/env python3
"""
Advanced Portfolio Optimizer for Forex Trading Bot
Features:
- Risk Parity optimization
- Mean-Variance optimization
- Kelly Criterion position sizing
- Dynamic correlation analysis
- Portfolio rebalancing
- Risk metrics calculation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import logging

class PortfolioOptimizer:
    def __init__(self, config: Dict):
        """Initialize portfolio optimizer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.portfolio_weights = {}
        self.correlation_matrix = pd.DataFrame()
        self.volatility_matrix = pd.DataFrame()
        self.expected_returns = pd.Series()
        
    def optimize_portfolio(self, 
                         returns: pd.DataFrame, 
                         method: str = "risk_parity") -> Dict:
        """Optimize portfolio using specified method"""
        try:
            if method == "risk_parity":
                return self._risk_parity_optimization(returns)
            elif method == "mean_variance":
                return self._mean_variance_optimization(returns)
            elif method == "kelly":
                return self._kelly_criterion_optimization(returns)
            elif method == "minimum_variance":
                return self._minimum_variance_optimization(returns)
            else:
                self.logger.warning(f"Unknown optimization method: {method}")
                return self._risk_parity_optimization(returns)
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return self._get_default_weights(returns.columns)
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> Dict:
        """Risk parity optimization - equal risk contribution"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Risk parity objective function
            def risk_parity_objective(weights):
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                risk_contrib = weights * (np.dot(cov_matrix, weights)) / portfolio_risk
                risk_diff = risk_contrib - risk_contrib.mean()
                return np.sum(risk_diff ** 2)
            
            # Constraints
            n_assets = len(returns.columns)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            ]
            
            # Bounds
            min_weight = self.config.get("min_weight", 0.05)
            max_weight = self.config.get("max_weight", 0.3)
            bounds = [(min_weight, max_weight)] * n_assets
            
            # Initial weights
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=returns.columns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                return {
                    'weights': weights,
                    'portfolio_risk': portfolio_risk,
                    'method': 'risk_parity',
                    'success': True
                }
            else:
                self.logger.warning("Risk parity optimization failed, using default weights")
                return self._get_default_weights(returns.columns)
                
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {e}")
            return self._get_default_weights(returns.columns)
    
    def _mean_variance_optimization(self, returns: pd.DataFrame) -> Dict:
        """Mean-variance optimization (Markowitz)"""
        try:
            # Calculate expected returns and covariance
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252
            
            # Risk aversion parameter
            risk_aversion = 1.0
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = portfolio_return / portfolio_risk
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            # Constraints
            n_assets = len(returns.columns)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            ]
            
            # Bounds
            min_weight = self.config.get("min_weight", 0.05)
            max_weight = self.config.get("max_weight", 0.3)
            bounds = [(min_weight, max_weight)] * n_assets
            
            # Initial weights
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=returns.columns)
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = portfolio_return / portfolio_risk
                
                return {
                    'weights': weights,
                    'portfolio_return': portfolio_return,
                    'portfolio_risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'method': 'mean_variance',
                    'success': True
                }
            else:
                self.logger.warning("Mean-variance optimization failed, using default weights")
                return self._get_default_weights(returns.columns)
                
        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {e}")
            return self._get_default_weights(returns.columns)
    
    def _kelly_criterion_optimization(self, returns: pd.DataFrame) -> Dict:
        """Kelly Criterion optimization for position sizing"""
        try:
            # Calculate Kelly fractions
            kelly_fractions = {}
            
            for asset in returns.columns:
                asset_returns = returns[asset].dropna()
                
                if len(asset_returns) > 0:
                    # Calculate win rate and average win/loss
                    positive_returns = asset_returns[asset_returns > 0]
                    negative_returns = asset_returns[asset_returns < 0]
                    
                    if len(positive_returns) > 0 and len(negative_returns) > 0:
                        win_rate = len(positive_returns) / len(asset_returns)
                        avg_win = positive_returns.mean()
                        avg_loss = abs(negative_returns.mean())
                        
                        # Kelly fraction: f = (bp - q) / b
                        # where b = odds received, p = probability of win, q = probability of loss
                        if avg_loss > 0:
                            b = avg_win / avg_loss
                            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
                            kelly_fractions[asset] = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                        else:
                            kelly_fractions[asset] = 0.1
                    else:
                        kelly_fractions[asset] = 0.1
                else:
                    kelly_fractions[asset] = 0.1
            
            # Normalize weights to sum to 1
            total_fraction = sum(kelly_fractions.values())
            if total_fraction > 0:
                weights = pd.Series({k: v/total_fraction for k, v in kelly_fractions.items()})
            else:
                weights = self._get_default_weights(returns.columns)
            
            return {
                'weights': weights,
                'method': 'kelly_criterion',
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in Kelly criterion optimization: {e}")
            return self._get_default_weights(returns.columns)
    
    def _minimum_variance_optimization(self, returns: pd.DataFrame) -> Dict:
        """Minimum variance optimization"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            n_assets = len(returns.columns)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            ]
            
            # Bounds
            min_weight = self.config.get("min_weight", 0.05)
            max_weight = self.config.get("max_weight", 0.3)
            bounds = [(min_weight, max_weight)] * n_assets
            
            # Initial weights
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=returns.columns)
                portfolio_risk = np.sqrt(result.fun)
                
                return {
                    'weights': weights,
                    'portfolio_risk': portfolio_risk,
                    'method': 'minimum_variance',
                    'success': True
                }
            else:
                self.logger.warning("Minimum variance optimization failed, using default weights")
                return self._get_default_weights(returns.columns)
                
        except Exception as e:
            self.logger.error(f"Error in minimum variance optimization: {e}")
            return self._get_default_weights(returns.columns)
    
    def _get_default_weights(self, assets: List[str]) -> Dict:
        """Get default equal-weight portfolio"""
        n_assets = len(assets)
        weights = pd.Series(1/n_assets, index=assets)
        
        return {
            'weights': weights,
            'method': 'equal_weight',
            'success': True
        }
    
    def calculate_portfolio_metrics(self, 
                                 returns: pd.DataFrame, 
                                 weights: pd.Series) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Basic metrics
            total_return = portfolio_returns.sum()
            annualized_return = portfolio_returns.mean() * 252
            annualized_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Risk metrics
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Correlation analysis
            correlation_matrix = returns.corr()
            avg_correlation = (correlation_matrix.sum().sum() - len(returns.columns)) / (len(returns.columns) ** 2 - len(returns.columns))
            
            # Diversification ratio
            individual_vols = returns.std() * np.sqrt(252)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            weighted_vol = np.dot(weights, individual_vols)
            diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'avg_correlation': avg_correlation,
                'diversification_ratio': diversification_ratio,
                'portfolio_weights': weights.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception:
            return 0.0
    
    def rebalance_portfolio(self, 
                          current_weights: Dict, 
                          target_weights: Dict, 
                          threshold: float = 0.05) -> Dict:
        """Determine rebalancing trades"""
        try:
            rebalance_trades = {}
            
            for asset in target_weights:
                current_weight = current_weights.get(asset, 0)
                target_weight = target_weights[asset]
                weight_diff = target_weight - current_weight
                
                # Only rebalance if difference exceeds threshold
                if abs(weight_diff) > threshold:
                    rebalance_trades[asset] = {
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_change': weight_diff,
                        'action': 'buy' if weight_diff > 0 else 'sell'
                    }
            
            return rebalance_trades
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")
            return {}
    
    def calculate_position_sizes(self, 
                               account_balance: float, 
                               weights: pd.Series, 
                               risk_per_trade: float = 0.02) -> Dict:
        """Calculate position sizes based on portfolio weights and risk"""
        try:
            position_sizes = {}
            
            for asset, weight in weights.items():
                # Calculate position size based on weight and risk
                position_value = account_balance * weight
                position_size = position_value / 100000  # Assuming 100k lot size
                
                # Apply risk limits
                max_position_value = account_balance * risk_per_trade
                if position_value > max_position_value:
                    position_value = max_position_value
                    position_size = position_value / 100000
                
                position_sizes[asset] = {
                    'weight': weight,
                    'position_value': position_value,
                    'position_size': position_size,
                    'risk_amount': position_value * risk_per_trade
                }
            
            return position_sizes
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {e}")
            return {}
    
    def update_correlation_matrix(self, returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Update rolling correlation matrix"""
        try:
            # Calculate rolling correlations
            correlation_matrix = returns.rolling(window).corr()
            
            # Get latest correlation matrix
            latest_correlation = correlation_matrix.iloc[-len(returns.columns):]
            
            self.correlation_matrix = latest_correlation
            return latest_correlation
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
            return pd.DataFrame()
    
    def detect_correlation_breaks(self, 
                                returns: pd.DataFrame, 
                                threshold: float = 0.8) -> List[str]:
        """Detect assets with high correlation that may need hedging"""
        try:
            correlation_matrix = returns.corr()
            high_correlation_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    correlation = abs(correlation_matrix.iloc[i, j])
                    if correlation > threshold:
                        asset1 = correlation_matrix.columns[i]
                        asset2 = correlation_matrix.columns[j]
                        high_correlation_pairs.append(f"{asset1}-{asset2}")
            
            return high_correlation_pairs
            
        except Exception as e:
            self.logger.error(f"Error detecting correlation breaks: {e}")
            return []