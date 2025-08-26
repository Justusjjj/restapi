#!/usr/bin/env python3
"""
Advanced Time Series Analysis Module for Forex Trading Bot
Features:
- ARIMA and SARIMA models
- GARCH volatility modeling
- Vector Autoregression (VAR)
- Cointegration analysis
- Structural breaks detection
- Forecasting and backtesting
- Model selection and validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss, coint
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.regime_switching import MarkovRegression
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, GJR-GARCH
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.structural import UnobservedComponents
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False
    logging.warning("Some time series libraries not available. Install statsmodels and arch for full functionality.")

class AdvancedTimeSeriesAnalyzer:
    def __init__(self, config: Dict):
        """Initialize advanced time series analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.forecasts = {}
        self.model_metrics = {}
        
        if not TS_AVAILABLE:
            self.logger.error("Time series libraries not available. Install statsmodels and arch.")
    
    def fit_arima_model(self, 
                       data: pd.Series, 
                       order: Tuple[int, int, int] = None,
                       seasonal_order: Tuple[int, int, int, int] = None,
                       auto_select: bool = True) -> Dict:
        """Fit ARIMA/SARIMA model to time series data"""
        try:
            if not TS_AVAILABLE:
                return {"error": "Time series libraries not available"}
            
            # Auto-select order if not specified
            if auto_select and order is None:
                order = self._auto_select_arima_order(data)
            
            if order is None:
                order = (1, 1, 1)  # Default order
            
            # Fit ARIMA model
            if seasonal_order:
                model = ARIMA(data, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(data, order=order)
            
            fitted_model = model.fit()
            
            # Model diagnostics
            diagnostics = self._arima_diagnostics(fitted_model, data)
            
            # Store model
            model_key = f"arima_{order}_{seasonal_order if seasonal_order else 'None'}"
            self.models[model_key] = fitted_model
            
            return {
                "model": fitted_model,
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": fitted_model.aic,
                "bic": fitted_model.bic,
                "diagnostics": diagnostics,
                "residuals": fitted_model.resid,
                "fitted_values": fitted_model.fittedvalues
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting ARIMA model: {e}")
            return {"error": str(e)}
    
    def _auto_select_arima_order(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Auto-select optimal ARIMA order using AIC"""
        try:
            best_aic = np.inf
            best_order = (1, 1, 1)
            
            # Test different orders
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            return best_order
            
        except Exception as e:
            self.logger.error(f"Error auto-selecting ARIMA order: {e}")
            return (1, 1, 1)
    
    def _arima_diagnostics(self, model, data: pd.Series) -> Dict:
        """Perform ARIMA model diagnostics"""
        try:
            residuals = model.resid
            
            # Ljung-Box test for autocorrelation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            # Jarque-Bera test for normality
            jb_test = stats.jarque_bera(residuals)
            
            # ADF test on residuals
            adf_test = adfuller(residuals.dropna())
            
            return {
                "ljung_box": {
                    "statistic": lb_test.iloc[-1]["lb_stat"],
                    "p_value": lb_test.iloc[-1]["lb_pvalue"]
                },
                "jarque_bera": {
                    "statistic": jb_test[0],
                    "p_value": jb_test[1]
                },
                "adf_residuals": {
                    "statistic": adf_test[0],
                    "p_value": adf_test[1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA diagnostics: {e}")
            return {}
    
    def fit_garch_model(self, 
                       returns: pd.Series, 
                       model_type: str = "GARCH",
                       p: int = 1, 
                       q: int = 1) -> Dict:
        """Fit GARCH model for volatility modeling"""
        try:
            if not TS_AVAILABLE:
                return {"error": "Time series libraries not available"}
            
            # Select GARCH model type
            if model_type.upper() == "GARCH":
                garch_model = GARCH(returns, p=p, q=q)
            elif model_type.upper() == "EGARCH":
                garch_model = EGARCH(returns, p=p, q=q)
            elif model_type.upper() == "GJR-GARCH":
                garch_model = GJR-GARCH(returns, p=p, q=q)
            else:
                garch_model = GARCH(returns, p=p, q=q)
            
            # Fit model
            fitted_model = garch_model.fit(disp="off")
            
            # Model diagnostics
            diagnostics = self._garch_diagnostics(fitted_model, returns)
            
            # Store model
            model_key = f"garch_{model_type}_{p}_{q}"
            self.models[model_key] = fitted_model
            
            return {
                "model": fitted_model,
                "model_type": model_type,
                "order": (p, q),
                "aic": fitted_model.aic,
                "bic": fitted_model.bic,
                "diagnostics": diagnostics,
                "conditional_volatility": fitted_model.conditional_volatility,
                "residuals": fitted_model.resid
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting GARCH model: {e}")
            return {"error": str(e)}
    
    def _garch_diagnostics(self, model, returns: pd.Series) -> Dict:
        """Perform GARCH model diagnostics"""
        try:
            residuals = model.resid
            standardized_residuals = residuals / model.conditional_volatility
            
            # Ljung-Box test on residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_residuals = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            # Ljung-Box test on squared residuals
            lb_squared = acorr_ljungbox(residuals**2, lags=10, return_df=True)
            
            # Jarque-Bera test
            jb_test = stats.jarque_bera(standardized_residuals)
            
            return {
                "ljung_box_residuals": {
                    "statistic": lb_residuals.iloc[-1]["lb_stat"],
                    "p_value": lb_residuals.iloc[-1]["lb_pvalue"]
                },
                "ljung_box_squared": {
                    "statistic": lb_squared.iloc[-1]["lb_stat"],
                    "p_value": lb_squared.iloc[-1]["lb_pvalue"]
                },
                "jarque_bera": {
                    "statistic": jb_test[0],
                    "p_value": jb_test[1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in GARCH diagnostics: {e}")
            return {}
    
    def fit_var_model(self, 
                     data: pd.DataFrame, 
                     maxlags: int = 10,
                     auto_select: bool = True) -> Dict:
        """Fit Vector Autoregression (VAR) model"""
        try:
            if not TS_AVAILABLE:
                return {"error": "Time series libraries not available"}
            
            # Auto-select lag order if requested
            if auto_select:
                optimal_lags = self._select_var_lags(data, maxlags)
            else:
                optimal_lags = maxlags
            
            # Fit VAR model
            var_model = VAR(data)
            fitted_model = var_model.fit(optimal_lags)
            
            # Model diagnostics
            diagnostics = self._var_diagnostics(fitted_model, data)
            
            # Store model
            model_key = f"var_{optimal_lags}"
            self.models[model_key] = fitted_model
            
            return {
                "model": fitted_model,
                "lags": optimal_lags,
                "aic": fitted_model.aic,
                "bic": fitted_model.bic,
                "diagnostics": diagnostics,
                "residuals": fitted_model.resid,
                "fitted_values": fitted_model.fittedvalues
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting VAR model: {e}")
            return {"error": str(e)}
    
    def _select_var_lags(self, data: pd.DataFrame, maxlags: int) -> int:
        """Select optimal lag order for VAR model"""
        try:
            var_model = VAR(data)
            results = var_model.select_order(maxlags=maxlags)
            return results.selected_orders['aic']
        except Exception as e:
            self.logger.error(f"Error selecting VAR lags: {e}")
            return 1
    
    def _var_diagnostics(self, model, data: pd.DataFrame) -> Dict:
        """Perform VAR model diagnostics"""
        try:
            # Test for serial correlation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            diagnostics = {}
            for i, col in enumerate(data.columns):
                residuals = model.resid[col]
                
                # Ljung-Box test
                lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
                diagnostics[col] = {
                    "ljung_box": {
                        "statistic": lb_test.iloc[-1]["lb_stat"],
                        "p_value": lb_test.iloc[-1]["lb_pvalue"]
                    }
                }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Error in VAR diagnostics: {e}")
            return {}
    
    def test_cointegration(self, 
                          series1: pd.Series, 
                          series2: pd.Series,
                          significance_level: float = 0.05) -> Dict:
        """Test for cointegration between two time series"""
        try:
            if not TS_AVAILABLE:
                return {"error": "Time series libraries not available"}
            
            # Engle-Granger cointegration test
            coint_result = coint(series1, series2)
            
            # ADF test on residuals
            residuals = series1 - series2
            adf_result = adfuller(residuals.dropna())
            
            is_cointegrated = coint_result[1] < significance_level
            
            return {
                "is_cointegrated": is_cointegrated,
                "cointegration_test": {
                    "statistic": coint_result[0],
                    "p_value": coint_result[1],
                    "critical_values": coint_result[2]
                },
                "residuals_adf": {
                    "statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "critical_values": adf_result[4]
                },
                "significance_level": significance_level
            }
            
        except Exception as e:
            self.logger.error(f"Error testing cointegration: {e}")
            return {"error": str(e)}
    
    def detect_structural_breaks(self, 
                                data: pd.Series, 
                                method: str = "cusum") -> Dict:
        """Detect structural breaks in time series"""
        try:
            if not TS_AVAILABLE:
                return {"error": "Time series libraries not available"}
            
            breaks = {}
            
            if method == "cusum" or method == "all":
                breaks["cusum"] = self._cusum_test(data)
            
            if method == "chow" or method == "all":
                breaks["chow"] = self._chow_test(data)
            
            if method == "bai_perron" or method == "all":
                breaks["bai_perron"] = self._bai_perron_test(data)
            
            return breaks
            
        except Exception as e:
            self.logger.error(f"Error detecting structural breaks: {e}")
            return {"error": str(e)}
    
    def _cusum_test(self, data: pd.Series) -> Dict:
        """CUSUM test for structural breaks"""
        try:
            # Calculate cumulative sum of standardized residuals
            mean = data.mean()
            std = data.std()
            standardized = (data - mean) / std
            
            cusum = standardized.cumsum()
            
            # Find potential break points
            threshold = 1.358 * np.sqrt(len(data))  # 95% confidence level
            break_points = np.where(np.abs(cusum) > threshold)[0]
            
            return {
                "test_statistic": cusum.max(),
                "threshold": threshold,
                "break_points": break_points.tolist(),
                "has_breaks": len(break_points) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in CUSUM test: {e}")
            return {"error": str(e)}
    
    def _chow_test(self, data: pd.Series) -> Dict:
        """Chow test for structural breaks"""
        try:
            # Simple Chow test implementation
            n = len(data)
            mid_point = n // 2
            
            # Split data into two periods
            period1 = data[:mid_point]
            period2 = data[mid_point:]
            
            # Calculate RSS for each period and combined
            rss1 = np.sum((period1 - period1.mean())**2)
            rss2 = np.sum((period2 - period2.mean())**2)
            rss_combined = np.sum((data - data.mean())**2)
            
            # Chow test statistic
            chow_stat = ((rss_combined - (rss1 + rss2)) / 2) / ((rss1 + rss2) / (n - 4))
            
            # F-distribution p-value
            p_value = 1 - stats.f.cdf(chow_stat, 2, n - 4)
            
            return {
                "test_statistic": chow_stat,
                "p_value": p_value,
                "has_breaks": p_value < 0.05,
                "break_point": mid_point
            }
            
        except Exception as e:
            self.logger.error(f"Error in Chow test: {e}")
            return {"error": str(e)}
    
    def _bai_perron_test(self, data: pd.Series) -> Dict:
        """Bai-Perron test for multiple structural breaks"""
        try:
            # Simplified Bai-Perron implementation
            # In practice, use specialized libraries like 'strucchange' in R
            
            # For now, implement a basic version
            n = len(data)
            max_breaks = min(5, n // 20)  # Maximum number of breaks
            
            # Find break points using rolling variance
            rolling_var = data.rolling(window=20).var()
            potential_breaks = []
            
            for i in range(20, n-20):
                if rolling_var.iloc[i] > rolling_var.iloc[i-1] * 1.5:
                    potential_breaks.append(i)
            
            return {
                "max_breaks": max_breaks,
                "potential_breaks": potential_breaks[:max_breaks],
                "has_breaks": len(potential_breaks) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in Bai-Perron test: {e}")
            return {"error": str(e)}
    
    def decompose_time_series(self, 
                             data: pd.Series, 
                             period: int = None,
                             model: str = "additive") -> Dict:
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            if not TS_AVAILABLE:
                return {"error": "Time series libraries not available"}
            
            # Seasonal decomposition
            if period is None:
                # Auto-detect period
                period = self._detect_seasonality(data)
            
            decomposition = seasonal_decompose(data, period=period, model=model)
            
            return {
                "trend": decomposition.trend,
                "seasonal": decomposition.seasonal,
                "residual": decomposition.resid,
                "period": period,
                "model": model
            }
            
        except Exception as e:
            self.logger.error(f"Error decomposing time series: {e}")
            return {"error": str(e)}
    
    def _detect_seasonality(self, data: pd.Series) -> int:
        """Auto-detect seasonality period"""
        try:
            # Use autocorrelation to detect seasonality
            acf = pd.Series(data).autocorr(lag=1)
            
            # Check common periods
            common_periods = [4, 7, 12, 52, 252]  # Quarterly, weekly, monthly, yearly, trading days
            
            best_period = 1
            best_correlation = 0
            
            for period in common_periods:
                if len(data) > period * 2:
                    correlation = pd.Series(data).autocorr(lag=period)
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_period = period
            
            return best_period
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonality: {e}")
            return 1
    
    def forecast(self, 
                model_key: str, 
                steps: int = 10,
                confidence_level: float = 0.95) -> Dict:
        """Generate forecasts using fitted models"""
        try:
            if model_key not in self.models:
                return {"error": f"Model {model_key} not found"}
            
            model = self.models[model_key]
            
            if "arima" in model_key:
                forecast_result = model.forecast(steps=steps)
                forecast_values = forecast_result.predicted_mean
                forecast_conf = forecast_result.conf_int(alpha=1-confidence_level)
                
            elif "garch" in model_key:
                # GARCH models don't have built-in forecasting for returns
                # Use conditional volatility for volatility forecasting
                forecast_values = model.forecast(horizon=steps)
                forecast_conf = None
                
            elif "var" in model_key:
                forecast_result = model.forecast(model.y, steps=steps)
                forecast_values = forecast_result
                forecast_conf = None
                
            else:
                return {"error": f"Forecasting not implemented for {model_key}"}
            
            # Store forecast
            self.forecasts[model_key] = {
                "values": forecast_values,
                "confidence_intervals": forecast_conf,
                "steps": steps,
                "confidence_level": confidence_level,
                "timestamp": pd.Timestamp.now()
            }
            
            return {
                "forecast": forecast_values,
                "confidence_intervals": forecast_conf,
                "steps": steps,
                "confidence_level": confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return {"error": str(e)}
    
    def evaluate_forecast(self, 
                         model_key: str, 
                         actual_values: pd.Series) -> Dict:
        """Evaluate forecast accuracy"""
        try:
            if model_key not in self.forecasts:
                return {"error": f"Forecast for {model_key} not found"}
            
            forecast = self.forecasts[model_key]
            predicted = forecast["values"]
            
            # Ensure same length
            min_length = min(len(predicted), len(actual_values))
            predicted = predicted[:min_length]
            actual = actual_values[:min_length]
            
            # Calculate error metrics
            mse = np.mean((actual - predicted)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Direction accuracy
            direction_correct = np.sum(np.sign(actual.diff()) == np.sign(predicted.diff()))
            direction_accuracy = direction_correct / (len(actual) - 1) * 100
            
            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "direction_accuracy": direction_accuracy,
                "actual": actual,
                "predicted": predicted
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating forecast: {e}")
            return {"error": str(e)}
    
    def generate_time_series_report(self, 
                                  data: pd.Series, 
                                  model_results: Dict) -> Dict:
        """Generate comprehensive time series analysis report"""
        try:
            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_summary": {
                    "length": len(data),
                    "start_date": data.index[0],
                    "end_date": data.index[-1],
                    "mean": data.mean(),
                    "std": data.std(),
                    "min": data.min(),
                    "max": data.max()
                },
                "stationarity_tests": self._test_stationarity(data),
                "seasonality_analysis": self._analyze_seasonality(data),
                "model_results": model_results,
                "recommendations": []
            }
            
            # Generate recommendations
            recommendations = self._generate_ts_recommendations(data, model_results)
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating time series report: {e}")
            return {"error": str(e)}
    
    def _test_stationarity(self, data: pd.Series) -> Dict:
        """Test for stationarity"""
        try:
            if not TS_AVAILABLE:
                return {"error": "Time series libraries not available"}
            
            # ADF test
            adf_result = adfuller(data.dropna())
            
            # KPSS test
            kpss_result = kpss(data.dropna())
            
            return {
                "adf": {
                    "statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "is_stationary": adf_result[1] < 0.05
                },
                "kpss": {
                    "statistic": kpss_result[0],
                    "p_value": kpss_result[1],
                    "is_stationary": kpss_result[1] > 0.05
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error testing stationarity: {e}")
            return {"error": str(e)}
    
    def _analyze_seasonality(self, data: pd.Series) -> Dict:
        """Analyze seasonality in time series"""
        try:
            # Decompose time series
            decomposition = self.decompose_time_series(data)
            
            if "error" in decomposition:
                return decomposition
            
            # Calculate seasonal strength
            seasonal_strength = np.var(decomposition["seasonal"]) / np.var(data)
            
            return {
                "seasonal_strength": seasonal_strength,
                "has_seasonality": seasonal_strength > 0.1,
                "decomposition": decomposition
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing seasonality: {e}")
            return {"error": str(e)}
    
    def _generate_ts_recommendations(self, data: pd.Series, model_results: Dict) -> List[str]:
        """Generate recommendations based on time series analysis"""
        try:
            recommendations = []
            
            # Check stationarity
            stationarity = self._test_stationarity(data)
            if "adf" in stationarity and not stationarity["adf"]["is_stationary"]:
                recommendations.append("Data is non-stationary - consider differencing")
            
            # Check seasonality
            seasonality = self._analyze_seasonality(data)
            if "has_seasonality" in seasonality and seasonality["has_seasonality"]:
                recommendations.append("Strong seasonality detected - consider SARIMA models")
            
            # Check model fit
            if "aic" in model_results:
                if model_results["aic"] > 1000:
                    recommendations.append("High AIC - consider different model specification")
            
            if not recommendations:
                recommendations.append("Time series analysis shows good model fit")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]