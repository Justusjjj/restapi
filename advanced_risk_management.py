#!/usr/bin/env python3
"""
Advanced Risk Management Module for Forex Trading Bot
Features:
- Value at Risk (VaR) calculations
- Conditional VaR (CVaR)
- Stress testing and scenario analysis
- Risk-adjusted performance metrics
- Position sizing and risk limits
- Portfolio risk decomposition
- Real-time risk monitoring
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    def __init__(self, config: Dict):
        """Initialize advanced risk manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_limits = config.get("risk", {})
        self.risk_metrics = {}
        self.position_limits = {}
        self.stress_scenarios = self._initialize_stress_scenarios()
        
    def _initialize_stress_scenarios(self) -> Dict:
        """Initialize stress testing scenarios"""
        return {
            "market_crash": {
                "description": "2008-style market crash",
                "equity_shock": -0.40,
                "volatility_multiplier": 3.0,
                "correlation_shift": 0.3,
                "liquidity_dry_up": True
            },
            "flash_crash": {
                "description": "2010 flash crash scenario",
                "equity_shock": -0.20,
                "volatility_multiplier": 5.0,
                "correlation_shift": 0.5,
                "liquidity_dry_up": True
            },
            "currency_crisis": {
                "description": "Emerging market currency crisis",
                "equity_shock": -0.30,
                "volatility_multiplier": 4.0,
                "correlation_shift": 0.4,
                "liquidity_dry_up": False
            },
            "central_bank_intervention": {
                "description": "Major central bank policy shift",
                "equity_shock": -0.15,
                "volatility_multiplier": 2.5,
                "correlation_shift": 0.2,
                "liquidity_dry_up": False
            }
        }
    
    def calculate_var(self, 
                     returns: pd.Series, 
                     confidence_level: float = 0.95, 
                     method: str = "historical") -> Dict:
        """Calculate Value at Risk using multiple methods"""
        try:
            results = {}
            
            if method == "historical" or method == "all":
                results["historical"] = self._historical_var(returns, confidence_level)
            
            if method == "parametric" or method == "all":
                results["parametric"] = self._parametric_var(returns, confidence_level)
            
            if method == "monte_carlo" or method == "all":
                results["monte_carlo"] = self._monte_carlo_var(returns, confidence_level)
            
            if method == "cornish_fisher" or method == "all":
                results["cornish_fisher"] = self._cornish_fisher_var(returns, confidence_level)
            
            # Calculate average VaR if multiple methods
            if len(results) > 1:
                var_values = [result["var"] for result in results.values()]
                results["average"] = {
                    "var": np.mean(var_values),
                    "method": "average",
                    "confidence_level": confidence_level
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return {}
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Calculate historical VaR"""
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return {
                "var": var,
                "method": "historical",
                "confidence_level": confidence_level,
                "observations": len(returns)
            }
        except Exception as e:
            self.logger.error(f"Error in historical VaR: {e}")
            return {"var": 0, "method": "historical", "error": str(e)}
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Calculate parametric VaR assuming normal distribution"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(confidence_level)
            var = mean_return - z_score * std_return
            
            return {
                "var": var,
                "method": "parametric",
                "confidence_level": confidence_level,
                "mean": mean_return,
                "std": std_return,
                "z_score": z_score
            }
        except Exception as e:
            self.logger.error(f"Error in parametric VaR: {e}")
            return {"var": 0, "method": "parametric", "error": str(e)}
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float, n_simulations: int = 10000) -> Dict:
        """Calculate Monte Carlo VaR"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random returns
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
            return {
                "var": var,
                "method": "monte_carlo",
                "confidence_level": confidence_level,
                "simulations": n_simulations,
                "mean": mean_return,
                "std": std_return
            }
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo VaR: {e}")
            return {"var": 0, "method": "monte_carlo", "error": str(e)}
    
    def _cornish_fisher_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Calculate Cornish-Fisher VaR (accounts for skewness and kurtosis)"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Cornish-Fisher expansion
            z_score = stats.norm.ppf(confidence_level)
            z_cf = z_score + (z_score**2 - 1) * skewness / 6 + (z_score**3 - 3*z_score) * (kurtosis - 3) / 24
            
            var = mean_return - z_cf * std_return
            
            return {
                "var": var,
                "method": "cornish_fisher",
                "confidence_level": confidence_level,
                "mean": mean_return,
                "std": std_return,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "z_cf": z_cf
            }
        except Exception as e:
            self.logger.error(f"Error in Cornish-Fisher VaR: {e}")
            return {"var": 0, "method": "cornish_fisher", "error": str(e)}
    
    def calculate_cvar(self, 
                      returns: pd.Series, 
                      confidence_level: float = 0.95) -> Dict:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            # Calculate VaR first
            var_result = self.calculate_var(returns, confidence_level, "historical")
            var = var_result["historical"]["var"]
            
            # Calculate CVaR (average of returns below VaR)
            tail_returns = returns[returns <= var]
            cvar = tail_returns.mean() if len(tail_returns) > 0 else var
            
            return {
                "var": var,
                "cvar": cvar,
                "confidence_level": confidence_level,
                "tail_observations": len(tail_returns),
                "tail_probability": len(tail_returns) / len(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {e}")
            return {"var": 0, "cvar": 0, "error": str(e)}
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            metrics = {}
            
            # Basic statistics
            metrics["mean"] = returns.mean()
            metrics["std"] = returns.std()
            metrics["skewness"] = returns.skew()
            metrics["kurtosis"] = returns.kurtosis()
            
            # Risk metrics
            metrics["var_95"] = self.calculate_var(returns, 0.95, "historical")["historical"]["var"]
            metrics["var_99"] = self.calculate_var(returns, 0.99, "historical")["historical"]["var"]
            metrics["cvar_95"] = self.calculate_cvar(returns, 0.95)["cvar"]
            metrics["cvar_99"] = self.calculate_cvar(returns, 0.99)["cvar"]
            
            # Drawdown metrics
            metrics["max_drawdown"] = self._calculate_max_drawdown(returns)
            metrics["avg_drawdown"] = self._calculate_average_drawdown(returns)
            
            # Volatility metrics
            metrics["annualized_volatility"] = returns.std() * np.sqrt(252)
            metrics["downside_deviation"] = self._calculate_downside_deviation(returns)
            
            # Risk-adjusted returns
            metrics["sharpe_ratio"] = self._calculate_sharpe_ratio(returns)
            metrics["sortino_ratio"] = self._calculate_sortino_ratio(returns)
            metrics["calmar_ratio"] = self._calculate_calmar_ratio(returns)
            
            # Distribution tests
            metrics["normality_test"] = self._test_normality(returns)
            metrics["stationarity_test"] = self._test_stationarity(returns)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
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
    
    def _calculate_average_drawdown(self, returns: pd.Series) -> float:
        """Calculate average drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown[drawdown < 0].mean()
        except Exception:
            return 0.0
    
    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate downside deviation"""
        try:
            downside_returns = returns[returns < target]
            return np.sqrt(np.mean(downside_returns**2))
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - risk_free_rate/252
            return excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - risk_free_rate/252
            downside_dev = self._calculate_downside_deviation(returns)
            return excess_returns.mean() / downside_dev if downside_dev > 0 else 0
        except Exception:
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Calmar ratio"""
        try:
            excess_returns = returns - risk_free_rate/252
            max_dd = abs(self._calculate_max_drawdown(returns))
            return excess_returns.mean() / max_dd if max_dd > 0 else 0
        except Exception:
            return 0.0
    
    def _test_normality(self, returns: pd.Series) -> Dict:
        """Test for normality using multiple tests"""
        try:
            from scipy.stats import shapiro, normaltest, jarque_bera
            
            tests = {}
            tests["shapiro"] = shapiro(returns)
            tests["normaltest"] = normaltest(returns)
            tests["jarque_bera"] = jarque_bera(returns)
            
            # Determine if normal (p-value > 0.05)
            is_normal = all(test[1] > 0.05 for test in tests.values())
            
            return {
                "is_normal": is_normal,
                "tests": tests
            }
        except Exception as e:
            self.logger.error(f"Error in normality test: {e}")
            return {"is_normal": False, "error": str(e)}
    
    def _test_stationarity(self, returns: pd.Series) -> Dict:
        """Test for stationarity using ADF test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(returns.dropna())
            
            return {
                "is_stationary": adf_result[1] < 0.05,
                "adf_statistic": adf_result[0],
                "p_value": adf_result[1],
                "critical_values": adf_result[4]
            }
        except Exception as e:
            self.logger.error(f"Error in stationarity test: {e}")
            return {"is_stationary": False, "error": str(e)}
    
    def calculate_portfolio_risk(self, 
                               returns: pd.DataFrame, 
                               weights: pd.Series) -> Dict:
        """Calculate portfolio-level risk metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Portfolio risk metrics
            portfolio_metrics = self.calculate_risk_metrics(portfolio_returns)
            
            # Risk decomposition
            risk_decomposition = self._decompose_portfolio_risk(returns, weights)
            
            # Correlation analysis
            correlation_analysis = self._analyze_portfolio_correlations(returns)
            
            return {
                "portfolio_metrics": portfolio_metrics,
                "risk_decomposition": risk_decomposition,
                "correlation_analysis": correlation_analysis,
                "weights": weights.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return {}
    
    def _decompose_portfolio_risk(self, returns: pd.DataFrame, weights: pd.Series) -> Dict:
        """Decompose portfolio risk into individual asset contributions"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252
            
            # Portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Marginal risk contribution
            marginal_risk = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)
            
            # Risk contribution
            risk_contribution = weights * marginal_risk
            
            # Percentage contribution
            percentage_contribution = risk_contribution / np.sqrt(portfolio_variance)
            
            decomposition = {}
            for i, asset in enumerate(returns.columns):
                decomposition[asset] = {
                    "weight": weights[asset],
                    "marginal_risk": marginal_risk[i],
                    "risk_contribution": risk_contribution[i],
                    "percentage_contribution": percentage_contribution[i]
                }
            
            return decomposition
            
        except Exception as e:
            self.logger.error(f"Error decomposing portfolio risk: {e}")
            return {}
    
    def _analyze_portfolio_correlations(self, returns: pd.DataFrame) -> Dict:
        """Analyze portfolio correlations"""
        try:
            correlation_matrix = returns.corr()
            
            # Average correlation
            n_assets = len(returns.columns)
            total_correlation = correlation_matrix.sum().sum() - n_assets
            avg_correlation = total_correlation / (n_assets * n_assets - n_assets)
            
            # High correlation pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append({
                            "asset1": correlation_matrix.columns[i],
                            "asset2": correlation_matrix.columns[j],
                            "correlation": corr
                        })
            
            return {
                "correlation_matrix": correlation_matrix.to_dict(),
                "average_correlation": avg_correlation,
                "high_correlation_pairs": high_corr_pairs
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    def run_stress_test(self, 
                       portfolio_returns: pd.Series, 
                       scenario: str = "market_crash") -> Dict:
        """Run stress test on portfolio"""
        try:
            if scenario not in self.stress_scenarios:
                raise ValueError(f"Unknown stress scenario: {scenario}")
            
            scenario_config = self.stress_scenarios[scenario]
            
            # Apply stress scenario
            stressed_returns = portfolio_returns.copy()
            
            # Equity shock
            if scenario_config["equity_shock"] != 0:
                stressed_returns = stressed_returns + scenario_config["equity_shock"] / 252
            
            # Volatility multiplier
            if scenario_config["volatility_multiplier"] != 1.0:
                stressed_returns = stressed_returns * scenario_config["volatility_multiplier"]
            
            # Calculate stressed risk metrics
            stressed_metrics = self.calculate_risk_metrics(stressed_returns)
            
            # Compare with baseline
            baseline_metrics = self.calculate_risk_metrics(portfolio_returns)
            
            # Calculate impact
            impact = {}
            for key in baseline_metrics:
                if key in stressed_metrics and isinstance(baseline_metrics[key], (int, float)):
                    baseline_val = baseline_metrics[key]
                    stressed_val = stressed_metrics[key]
                    if baseline_val != 0:
                        impact[key] = (stressed_val - baseline_val) / abs(baseline_val)
                    else:
                        impact[key] = 0
            
            return {
                "scenario": scenario,
                "scenario_config": scenario_config,
                "baseline_metrics": baseline_metrics,
                "stressed_metrics": stressed_metrics,
                "impact": impact
            }
            
        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            return {}
    
    def calculate_position_limits(self, 
                                account_balance: float, 
                                risk_per_trade: float = 0.02,
                                max_portfolio_risk: float = 0.05) -> Dict:
        """Calculate position size limits based on risk parameters"""
        try:
            # Maximum risk per trade
            max_risk_amount = account_balance * risk_per_trade
            
            # Maximum portfolio risk
            max_portfolio_risk_amount = account_balance * max_portfolio_risk
            
            # Position size limits
            limits = {
                "max_risk_per_trade": max_risk_amount,
                "max_portfolio_risk": max_portfolio_risk_amount,
                "max_position_size": max_risk_amount * 50,  # 50:1 leverage
                "max_daily_loss": account_balance * 0.02,
                "max_drawdown": account_balance * 0.05
            }
            
            return limits
            
        except Exception as e:
            self.logger.error(f"Error calculating position limits: {e}")
            return {}
    
    def check_risk_limits(self, 
                          current_positions: Dict, 
                          new_trade: Dict, 
                          account_balance: float) -> Dict:
        """Check if new trade violates risk limits"""
        try:
            limits = self.calculate_position_limits(account_balance)
            violations = []
            warnings = []
            
            # Check position size
            if new_trade.get("size", 0) > limits["max_position_size"]:
                violations.append("Position size exceeds maximum")
            
            # Check daily loss limit
            current_daily_pnl = sum(pos.get("daily_pnl", 0) for pos in current_positions.values())
            if abs(current_daily_pnl) > limits["max_daily_loss"]:
                violations.append("Daily loss limit exceeded")
            
            # Check portfolio risk
            total_exposure = sum(pos.get("exposure", 0) for pos in current_positions.values())
            if total_exposure > limits["max_portfolio_risk"]:
                violations.append("Portfolio risk limit exceeded")
            
            # Check correlation limits
            if len(current_positions) > 1:
                correlation_violations = self._check_correlation_limits(current_positions, new_trade)
                violations.extend(correlation_violations)
            
            return {
                "trade_allowed": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "limits": limits
            }
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return {"trade_allowed": False, "violations": ["Error checking limits"], "warnings": [], "limits": {}}
    
    def _check_correlation_limits(self, current_positions: Dict, new_trade: Dict) -> List[str]:
        """Check correlation limits between positions"""
        try:
            violations = []
            
            # This is a simplified check - in practice you'd use actual correlation data
            if len(current_positions) >= 3:
                # Check if adding new trade would create too many correlated positions
                violations.append("Maximum correlated positions limit reached")
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error checking correlation limits: {e}")
            return []
    
    def generate_risk_report(self, 
                           portfolio_data: Dict, 
                           time_period: str = "daily") -> Dict:
        """Generate comprehensive risk report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "time_period": time_period,
                "summary": {},
                "detailed_metrics": {},
                "risk_decomposition": {},
                "stress_test_results": {},
                "recommendations": []
            }
            
            # Generate recommendations based on risk metrics
            recommendations = self._generate_risk_recommendations(portfolio_data)
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {"error": str(e)}
    
    def _generate_risk_recommendations(self, portfolio_data: Dict) -> List[str]:
        """Generate risk management recommendations"""
        try:
            recommendations = []
            
            # Check VaR limits
            if portfolio_data.get("var_95", 0) < -0.02:
                recommendations.append("Consider reducing position sizes - VaR approaching limits")
            
            # Check correlation
            if portfolio_data.get("avg_correlation", 0) > 0.7:
                recommendations.append("High portfolio correlation detected - consider diversification")
            
            # Check drawdown
            if portfolio_data.get("max_drawdown", 0) < -0.05:
                recommendations.append("Maximum drawdown limit reached - review risk management")
            
            # Check volatility
            if portfolio_data.get("annualized_volatility", 0) > 0.25:
                recommendations.append("High volatility detected - consider hedging strategies")
            
            if not recommendations:
                recommendations.append("Portfolio risk metrics within acceptable ranges")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]