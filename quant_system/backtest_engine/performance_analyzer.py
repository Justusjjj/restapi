"""
Performance Analyzer
===================

Comprehensive performance analysis and risk metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Performance analysis and risk metrics calculation.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        pass
        
    def analyze(self, returns: List[float], dates: List[datetime]) -> Dict[str, Any]:
        """
        Comprehensive performance analysis.
        
        Args:
            returns: List of daily returns
            dates: List of corresponding dates
            
        Returns:
            Dictionary with performance metrics
        """
        if not returns or len(returns) < 2:
            return {}
            
        returns_series = pd.Series(returns, index=dates)
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(returns_series)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns_series)
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(returns_series)
        
        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(returns_series)
        
        # Return distribution
        distribution_metrics = self._calculate_distribution_metrics(returns_series)
        
        return {
            'basic_metrics': basic_metrics,
            'risk_metrics': risk_metrics,
            'drawdown_metrics': drawdown_metrics,
            'rolling_metrics': rolling_metrics,
            'distribution_metrics': distribution_metrics
        }
        
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        total_return = returns.iloc[-1] if len(returns) > 0 else 0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'num_periods': len(returns),
            'start_date': returns.index[0].isoformat() if len(returns) > 0 else None,
            'end_date': returns.index[-1].isoformat() if len(returns) > 0 else None
        }
        
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'cvar_99': returns[returns <= returns.quantile(0.01)].mean()
        }
        
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdowns.min()
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdowns)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'current_drawdown': drawdowns.iloc[-1] if len(drawdowns) > 0 else 0,
            'drawdown_periods': len(drawdowns[drawdowns < 0])
        }
        
    def _calculate_rolling_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate rolling metrics."""
        rolling_30d = returns.rolling(30)
        rolling_90d = returns.rolling(90)
        rolling_252d = returns.rolling(252)
        
        return {
            'rolling_30d_vol': rolling_30d.std().iloc[-1] * np.sqrt(252) if len(returns) >= 30 else np.nan,
            'rolling_90d_vol': rolling_90d.std().iloc[-1] * np.sqrt(252) if len(returns) >= 90 else np.nan,
            'rolling_252d_vol': rolling_252d.std().iloc[-1] * np.sqrt(252) if len(returns) >= 252 else np.nan,
            'rolling_30d_sharpe': self._calculate_sharpe_ratio(returns.tail(30)) if len(returns) >= 30 else np.nan,
            'rolling_90d_sharpe': self._calculate_sharpe_ratio(returns.tail(90)) if len(returns) >= 90 else np.nan
        }
        
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return distribution metrics."""
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = self._jarque_bera_test(returns)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        }
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return np.nan
            
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return np.nan
            
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
        
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return np.nan
            
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.nan
            
        return (excess_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))
        
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) < 2:
            return np.nan
            
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_drawdown = self._calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return np.nan
            
        return annualized_return / abs(max_drawdown)
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()
        
    def _calculate_max_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(drawdowns) == 0:
            return 0
            
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
                
        if current_period > 0:
            drawdown_periods.append(current_period)
            
        return max(drawdown_periods) if drawdown_periods else 0
        
    def _jarque_bera_test(self, returns: pd.Series) -> Tuple[float, float]:
        """Jarque-Bera test for normality."""
        try:
            from scipy import stats
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
            return jb_stat, jb_pvalue
        except ImportError:
            return np.nan, np.nan