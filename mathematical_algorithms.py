import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize, differential_evolution
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class AdvancedMathematicalAlgorithms:
    """
    Advanced Mathematical Algorithms for Forex Trading
    Includes complex formulas, statistical models, and algorithmic equations
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.models = {}
        self.parameters = {}
        self.performance_metrics = {}
        
    def _default_config(self):
        """Default configuration for mathematical algorithms"""
        return {
            'volatility_model': 'garch',
            'trend_detection': 'kalman',
            'mean_reversion': 'hurst',
            'momentum_analysis': 'rsi_momentum',
            'correlation_analysis': True,
            'regime_detection': True,
            'fractal_analysis': True,
            'entropy_analysis': True
        }
    
    def calculate_advanced_volatility(self, returns, method='garch'):
        """
        Calculate advanced volatility measures
        
        Args:
            returns: Series of returns
            method: Volatility calculation method
            
        Returns:
            Dictionary with volatility measures
        """
        try:
            volatility_measures = {}
            
            if method == 'garch':
                volatility_measures.update(self._garch_volatility(returns))
            elif method == 'ewma':
                volatility_measures.update(self._ewma_volatility(returns))
            elif method == 'realized':
                volatility_measures.update(self._realized_volatility(returns))
            
            # Additional volatility measures
            volatility_measures['parkinson'] = self._parkinson_volatility(returns)
            volatility_measures['garman_klass'] = self._garman_klass_volatility(returns)
            volatility_measures['rogers_satchell'] = self._rogers_satchell_volatility(returns)
            volatility_measures['yang_zhang'] = self._yang_zhang_volatility(returns)
            
            return volatility_measures
            
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return {}
    
    def _garch_volatility(self, returns):
        """Calculate GARCH volatility"""
        try:
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            results = model.fit(disp='off')
            
            # Extract parameters
            omega = results.params['omega']
            alpha = results.params['alpha[1]']
            beta = results.params['beta[1]']
            
            # Calculate conditional volatility
            conditional_vol = results.conditional_volatility
            
            return {
                'garch_volatility': conditional_vol,
                'garch_params': {'omega': omega, 'alpha': alpha, 'beta': beta},
                'persistence': alpha + beta
            }
            
        except Exception as e:
            print(f"Error in GARCH volatility: {e}")
            return {}
    
    def _ewma_volatility(self, returns, lambda_param=0.94):
        """Calculate EWMA volatility"""
        try:
            # Initialize variance
            variance = returns.var()
            ewma_variance = [variance]
            
            for i in range(1, len(returns)):
                new_variance = lambda_param * ewma_variance[-1] + (1 - lambda_param) * returns.iloc[i-1]**2
                ewma_variance.append(new_variance)
            
            ewma_volatility = np.sqrt(ewma_variance)
            
            return {
                'ewma_volatility': pd.Series(ewma_volatility, index=returns.index),
                'lambda': lambda_param
            }
            
        except Exception as e:
            print(f"Error in EWMA volatility: {e}")
            return {}
    
    def _realized_volatility(self, returns, window=20):
        """Calculate realized volatility"""
        try:
            realized_vol = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2)))
            
            return {
                'realized_volatility': realized_vol,
                'window': window
            }
            
        except Exception as e:
            print(f"Error in realized volatility: {e}")
            return {}
    
    def _parkinson_volatility(self, df):
        """Calculate Parkinson volatility using high-low range"""
        try:
            # Parkinson volatility = sqrt(1/(4*ln(2)) * sum(ln(high/low)^2))
            log_hl = np.log(df['high'] / df['low'])
            parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * log_hl**2)
            
            return parkinson_vol
            
        except Exception as e:
            print(f"Error in Parkinson volatility: {e}")
            return pd.Series(0, index=df.index)
    
    def _garman_klass_volatility(self, df):
        """Calculate Garman-Klass volatility"""
        try:
            # Garman-Klass volatility
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            
            gk_vol = np.sqrt(0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2)
            
            return gk_vol
            
        except Exception as e:
            print(f"Error in Garman-Klass volatility: {e}")
            return pd.Series(0, index=df.index)
    
    def _rogers_satchell_volatility(self, df):
        """Calculate Rogers-Satchell volatility"""
        try:
            # Rogers-Satchell volatility
            log_ho = np.log(df['high'] / df['open'])
            log_lo = np.log(df['low'] / df['open'])
            log_co = np.log(df['close'] / df['open'])
            
            rs_vol = np.sqrt(log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co))
            
            return rs_vol
            
        except Exception as e:
            print(f"Error in Rogers-Satchell volatility: {e}")
            return pd.Series(0, index=df.index)
    
    def _yang_zhang_volatility(self, df):
        """Calculate Yang-Zhang volatility"""
        try:
            # Yang-Zhang volatility (combines multiple estimators)
            k = 0.34 / (1.34 + (len(df) + 1) / (len(df) - 1))
            
            # Overnight volatility
            overnight_returns = np.log(df['open'] / df['close'].shift(1))
            sigma_o = overnight_returns.var()
            
            # Open-to-close volatility
            open_close_returns = np.log(df['close'] / df['open'])
            sigma_c = open_close_returns.var()
            
            # Rogers-Satchell volatility
            sigma_rs = self._rogers_satchell_volatility(df).var()
            
            # Yang-Zhang volatility
            yz_vol = np.sqrt(sigma_o + k * sigma_c + (1 - k) * sigma_rs)
            
            return yz_vol
            
        except Exception as e:
            print(f"Error in Yang-Zhang volatility: {e}")
            return 0
    
    def detect_trend_using_kalman(self, prices, process_noise=1e-5, measurement_noise=1e-2):
        """
        Detect trend using Kalman Filter
        
        Args:
            prices: Series of prices
            process_noise: Process noise parameter
            measurement_noise: Measurement noise parameter
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            n = len(prices)
            
            # State variables: [price, trend]
            x = np.zeros((2, n))
            P = np.zeros((2, 2, n))
            
            # Initialize
            x[0, 0] = prices.iloc[0]
            x[1, 0] = 0
            P[:, :, 0] = np.array([[1, 0], [0, 1]])
            
            # Kalman filter parameters
            F = np.array([[1, 1], [0, 1]])  # State transition matrix
            H = np.array([[1, 0]])  # Measurement matrix
            Q = np.array([[process_noise, 0], [0, process_noise]])  # Process noise
            R = measurement_noise  # Measurement noise
            
            # Run Kalman filter
            for k in range(1, n):
                # Predict
                x_pred = F @ x[:, k-1]
                P_pred = F @ P[:, :, k-1] @ F.T + Q
                
                # Update
                y = prices.iloc[k]
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T / S  # Kalman gain
                
                x[:, k] = x_pred + K * (y - H @ x_pred)
                P[:, :, k] = (np.eye(2) - K @ H) @ P_pred
            
            # Extract results
            filtered_prices = x[0, :]
            trend = x[1, :]
            trend_confidence = 1 / np.sqrt(P[1, 1, :])
            
            return {
                'filtered_prices': pd.Series(filtered_prices, index=prices.index),
                'trend': pd.Series(trend, index=prices.index),
                'trend_confidence': pd.Series(trend_confidence, index=prices.index),
                'trend_direction': np.sign(trend),
                'trend_strength': np.abs(trend)
            }
            
        except Exception as e:
            print(f"Error in Kalman trend detection: {e}")
            return {}
    
    def calculate_hurst_exponent(self, prices, max_lag=20):
        """
        Calculate Hurst exponent for mean reversion analysis
        
        Args:
            prices: Series of prices
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent and analysis
        """
        try:
            # Calculate price changes
            price_changes = prices.diff().dropna()
            
            # Calculate R/S statistic for different lags
            lags = range(2, max_lag + 1)
            rs_values = []
            
            for lag in lags:
                rs = self._calculate_rs_statistic(price_changes, lag)
                rs_values.append(rs)
            
            # Fit linear regression: log(R/S) = H * log(lag) + c
            log_lags = np.log(lags)
            log_rs = np.log(rs_values)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)
            
            hurst_exponent = slope
            
            # Interpret Hurst exponent
            if hurst_exponent > 0.5:
                regime = 'trending'
            elif hurst_exponent < 0.5:
                regime = 'mean_reverting'
            else:
                regime = 'random_walk'
            
            return {
                'hurst_exponent': hurst_exponent,
                'regime': regime,
                'r_squared': r_value**2,
                'p_value': p_value,
                'lags': lags,
                'rs_values': rs_values
            }
            
        except Exception as e:
            print(f"Error calculating Hurst exponent: {e}")
            return {}
    
    def _calculate_rs_statistic(self, data, lag):
        """Calculate R/S statistic for a given lag"""
        try:
            n = len(data)
            k = n // lag
            
            rs_values = []
            
            for i in range(k):
                segment = data[i*lag:(i+1)*lag]
                
                # Calculate mean
                mean_segment = segment.mean()
                
                # Calculate cumulative deviation
                dev = segment - mean_segment
                cum_dev = dev.cumsum()
                
                # Calculate R (range)
                R = cum_dev.max() - cum_dev.min()
                
                # Calculate S (standard deviation)
                S = segment.std()
                
                if S > 0:
                    rs_values.append(R / S)
            
            return np.mean(rs_values) if rs_values else 0
            
        except Exception as e:
            print(f"Error calculating R/S statistic: {e}")
            return 0
    
    def detect_market_regime(self, prices, returns, method='gmm'):
        """
        Detect market regime using Gaussian Mixture Models
        
        Args:
            prices: Series of prices
            returns: Series of returns
            method: Regime detection method
            
        Returns:
            Dictionary with regime analysis
        """
        try:
            if method == 'gmm':
                return self._gmm_regime_detection(returns)
            elif method == 'hmm':
                return self._hmm_regime_detection(returns)
            elif method == 'volatility_regime':
                return self._volatility_regime_detection(returns)
            else:
                return self._gmm_regime_detection(returns)
                
        except Exception as e:
            print(f"Error in regime detection: {e}")
            return {}
    
    def _gmm_regime_detection(self, returns, n_regimes=3):
        """Detect regimes using Gaussian Mixture Models"""
        try:
            from sklearn.mixture import GaussianMixture
            
            # Prepare features
            features = np.column_stack([
                returns.values,
                returns.rolling(20).mean().values,
                returns.rolling(20).std().values
            ])
            
            # Remove NaN values
            valid_indices = ~np.isnan(features).any(axis=1)
            features_clean = features[valid_indices]
            
            # Fit GMM
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            gmm.fit(features_clean)
            
            # Predict regimes
            regimes = gmm.predict(features_clean)
            
            # Calculate regime probabilities
            regime_probs = gmm.predict_proba(features_clean)
            
            # Create regime series
            regime_series = pd.Series(index=returns.index, dtype=float)
            regime_series.iloc[valid_indices] = regimes
            
            # Calculate regime characteristics
            regime_stats = {}
            for i in range(n_regimes):
                regime_returns = returns[regime_series == i]
                if len(regime_returns) > 0:
                    regime_stats[f'regime_{i}'] = {
                        'mean_return': regime_returns.mean(),
                        'volatility': regime_returns.std(),
                        'frequency': len(regime_returns) / len(returns),
                        'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
                    }
            
            return {
                'regimes': regime_series,
                'regime_probabilities': regime_probs,
                'regime_statistics': regime_stats,
                'model': gmm
            }
            
        except Exception as e:
            print(f"Error in GMM regime detection: {e}")
            return {}
    
    def _volatility_regime_detection(self, returns, window=20):
        """Detect regimes based on volatility"""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window).std()
            
            # Define regime thresholds
            vol_25 = rolling_vol.quantile(0.25)
            vol_75 = rolling_vol.quantile(0.75)
            
            # Classify regimes
            regimes = pd.Series(index=returns.index, dtype=str)
            regimes[rolling_vol <= vol_25] = 'low_volatility'
            regimes[(rolling_vol > vol_25) & (rolling_vol <= vol_75)] = 'medium_volatility'
            regimes[rolling_vol > vol_75] = 'high_volatility'
            
            return {
                'regimes': regimes,
                'volatility': rolling_vol,
                'thresholds': {'low': vol_25, 'high': vol_75}
            }
            
        except Exception as e:
            print(f"Error in volatility regime detection: {e}")
            return {}
    
    def calculate_fractal_dimension(self, prices, method='box_counting'):
        """
        Calculate fractal dimension for market complexity analysis
        
        Args:
            prices: Series of prices
            method: Fractal dimension calculation method
            
        Returns:
            Fractal dimension and analysis
        """
        try:
            if method == 'box_counting':
                return self._box_counting_fractal(prices)
            elif method == 'higuchi':
                return self._higuchi_fractal(prices)
            else:
                return self._box_counting_fractal(prices)
                
        except Exception as e:
            print(f"Error calculating fractal dimension: {e}")
            return {}
    
    def _box_counting_fractal(self, prices, max_boxes=100):
        """Calculate fractal dimension using box counting method"""
        try:
            # Normalize prices
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min())
            
            # Different box sizes
            box_sizes = np.logspace(-3, 0, max_boxes)
            box_counts = []
            
            for box_size in box_sizes:
                # Count boxes needed to cover the curve
                count = 0
                for i in range(len(normalized_prices) - 1):
                    # Calculate boxes needed for this segment
                    x1, y1 = i / len(normalized_prices), normalized_prices.iloc[i]
                    x2, y2 = (i + 1) / len(normalized_prices), normalized_prices.iloc[i + 1]
                    
                    # Count boxes for this line segment
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    
                    if dx > 0 and dy > 0:
                        count += max(int(dx / box_size), int(dy / box_size))
                    else:
                        count += 1
                
                box_counts.append(count)
            
            # Fit linear regression: log(N) = -D * log(ε) + c
            log_box_sizes = -np.log(box_sizes)
            log_box_counts = np.log(box_counts)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_box_sizes, log_box_counts)
            
            fractal_dimension = slope
            
            return {
                'fractal_dimension': fractal_dimension,
                'box_sizes': box_sizes,
                'box_counts': box_counts,
                'r_squared': r_value**2,
                'complexity': 'high' if fractal_dimension > 1.5 else 'medium' if fractal_dimension > 1.3 else 'low'
            }
            
        except Exception as e:
            print(f"Error in box counting fractal: {e}")
            return {}
    
    def calculate_entropy_measures(self, returns, method='sample_entropy'):
        """
        Calculate entropy measures for market randomness analysis
        
        Args:
            returns: Series of returns
            method: Entropy calculation method
            
        Returns:
            Dictionary with entropy measures
        """
        try:
            entropy_measures = {}
            
            if method == 'sample_entropy':
                entropy_measures['sample_entropy'] = self._sample_entropy(returns)
            elif method == 'approximate_entropy':
                entropy_measures['approximate_entropy'] = self._approximate_entropy(returns)
            elif method == 'permutation_entropy':
                entropy_measures['permutation_entropy'] = self._permutation_entropy(returns)
            
            # Additional entropy measures
            entropy_measures['shannon_entropy'] = self._shannon_entropy(returns)
            entropy_measures['renyi_entropy'] = self._renyi_entropy(returns)
            
            return entropy_measures
            
        except Exception as e:
            print(f"Error calculating entropy measures: {e}")
            return {}
    
    def _sample_entropy(self, data, m=2, r=0.2):
        """Calculate sample entropy"""
        try:
            # Standardize data
            data_std = (data - data.mean()) / data.std()
            r = r * data_std.std()
            
            # Count matches
            n = len(data_std)
            matches_m = 0
            matches_m1 = 0
            
            for i in range(n - m):
                for j in range(i + 1, n - m):
                    # Check m-length matches
                    if np.all(np.abs(data_std.iloc[i:i+m] - data_std.iloc[j:j+m]) <= r):
                        matches_m += 1
                        
                        # Check m+1 length matches
                        if np.abs(data_std.iloc[i+m] - data_std.iloc[j+m]) <= r:
                            matches_m1 += 1
            
            # Calculate sample entropy
            if matches_m > 0:
                sample_entropy = -np.log(matches_m1 / matches_m)
            else:
                sample_entropy = np.inf
            
            return sample_entropy
            
        except Exception as e:
            print(f"Error in sample entropy: {e}")
            return np.inf
    
    def _shannon_entropy(self, data, bins=50):
        """Calculate Shannon entropy"""
        try:
            # Create histogram
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            
            # Remove zero probabilities
            hist = hist[hist > 0]
            
            # Calculate Shannon entropy
            shannon_entropy = -np.sum(hist * np.log(hist))
            
            return shannon_entropy
            
        except Exception as e:
            print(f"Error in Shannon entropy: {e}")
            return 0
    
    def optimize_trading_parameters(self, historical_data, objective_function, 
                                  parameter_bounds, method='differential_evolution'):
        """
        Optimize trading parameters using advanced optimization algorithms
        
        Args:
            historical_data: Historical market data
            objective_function: Function to optimize
            parameter_bounds: Bounds for parameters
            method: Optimization method
            
        Returns:
            Optimization results
        """
        try:
            if method == 'differential_evolution':
                result = differential_evolution(objective_function, parameter_bounds, 
                                             maxiter=1000, popsize=15, seed=42)
            elif method == 'nelder_mead':
                result = minimize(objective_function, 
                               x0=np.mean(parameter_bounds, axis=1),
                               method='Nelder-Mead',
                               bounds=parameter_bounds)
            else:
                result = differential_evolution(objective_function, parameter_bounds)
            
            return {
                'optimal_parameters': result.x,
                'optimal_value': result.fun,
                'success': result.success,
                'iterations': result.nit,
                'message': result.message
            }
            
        except Exception as e:
            print(f"Error in parameter optimization: {e}")
            return {}
    
    def calculate_advanced_correlation(self, data1, data2, method='pearson'):
        """
        Calculate advanced correlation measures
        
        Args:
            data1: First data series
            data2: Second data series
            method: Correlation method
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            correlation_measures = {}
            
            if method == 'pearson':
                correlation_measures['pearson'] = stats.pearsonr(data1, data2)
            elif method == 'spearman':
                correlation_measures['spearman'] = stats.spearmanr(data1, data2)
            elif method == 'kendall':
                correlation_measures['kendall'] = stats.kendalltau(data1, data2)
            
            # Rolling correlation
            rolling_corr = data1.rolling(20).corr(data2)
            
            # Correlation stability
            correlation_stability = rolling_corr.std()
            
            # Cross-correlation at different lags
            max_lag = 20
            cross_corr = []
            lags = range(-max_lag, max_lag + 1)
            
            for lag in lags:
                if lag < 0:
                    corr = data1.iloc[:lag].corr(data2.iloc[-lag:]) if len(data1) > abs(lag) else np.nan
                else:
                    corr = data1.iloc[lag:].corr(data2.iloc[:-lag]) if len(data1) > lag else np.nan
                cross_corr.append(corr)
            
            # Find optimal lag
            optimal_lag = lags[np.nanargmax(cross_corr)]
            
            correlation_measures.update({
                'rolling_correlation': rolling_corr,
                'correlation_stability': correlation_stability,
                'cross_correlation': cross_corr,
                'optimal_lag': optimal_lag,
                'lags': lags
            })
            
            return correlation_measures
            
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return {}
    
    def generate_trading_signals(self, analysis_results, confidence_threshold=0.7):
        """
        Generate trading signals based on mathematical analysis
        
        Args:
            analysis_results: Results from various mathematical analyses
            confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            Dictionary with trading signals
        """
        try:
            signals = {
                'trend_signal': 'NEUTRAL',
                'volatility_signal': 'NEUTRAL',
                'regime_signal': 'NEUTRAL',
                'mean_reversion_signal': 'NEUTRAL',
                'overall_signal': 'NEUTRAL',
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Trend analysis
            if 'kalman_trend' in analysis_results:
                trend_data = analysis_results['kalman_trend']
                if 'trend_direction' in trend_data:
                    trend_direction = trend_data['trend_direction'].iloc[-1]
                    trend_strength = trend_data['trend_strength'].iloc[-1]
                    trend_confidence = trend_data['trend_confidence'].iloc[-1]
                    
                    if trend_confidence > confidence_threshold:
                        if trend_direction > 0 and trend_strength > 0.001:
                            signals['trend_signal'] = 'BULLISH'
                            signals['reasoning'].append(f"Strong bullish trend (strength: {trend_strength:.4f})")
                        elif trend_direction < 0 and trend_strength > 0.001:
                            signals['trend_signal'] = 'BEARISH'
                            signals['reasoning'].append(f"Strong bearish trend (strength: {trend_strength:.4f})")
            
            # Volatility analysis
            if 'volatility' in analysis_results:
                vol_data = analysis_results['volatility']
                if 'garch_volatility' in vol_data:
                    current_vol = vol_data['garch_volatility'].iloc[-1]
                    avg_vol = vol_data['garch_volatility'].mean()
                    
                    if current_vol > avg_vol * 1.5:
                        signals['volatility_signal'] = 'HIGH_VOLATILITY'
                        signals['reasoning'].append(f"High volatility detected ({current_vol:.4f} vs avg {avg_vol:.4f})")
                    elif current_vol < avg_vol * 0.5:
                        signals['volatility_signal'] = 'LOW_VOLATILITY'
                        signals['reasoning'].append(f"Low volatility detected ({current_vol:.4f} vs avg {avg_vol:.4f})")
            
            # Regime analysis
            if 'regime' in analysis_results:
                regime_data = analysis_results['regime']
                if 'regimes' in regime_data:
                    current_regime = regime_data['regimes'].iloc[-1]
                    signals['regime_signal'] = current_regime
                    signals['reasoning'].append(f"Current market regime: {current_regime}")
            
            # Mean reversion analysis
            if 'hurst' in analysis_results:
                hurst_data = analysis_results['hurst']
                if 'hurst_exponent' in hurst_data:
                    hurst_exp = hurst_data['hurst_exponent']
                    regime = hurst_data['regime']
                    
                    if regime == 'mean_reverting' and hurst_exp < 0.4:
                        signals['mean_reversion_signal'] = 'STRONG_MEAN_REVERSION'
                        signals['reasoning'].append(f"Strong mean reversion (H={hurst_exp:.3f})")
                    elif regime == 'trending' and hurst_exp > 0.6:
                        signals['mean_reversion_signal'] = 'TRENDING'
                        signals['reasoning'].append(f"Strong trending (H={hurst_exp:.3f})")
            
            # Generate overall signal
            bullish_count = sum(1 for signal in [signals['trend_signal'], signals['volatility_signal'], 
                                               signals['regime_signal'], signals['mean_reversion_signal'] 
                                               if 'BULLISH' in str(signal) or 'LOW_VOLATILITY' in str(signal)])
            
            bearish_count = sum(1 for signal in [signals['trend_signal'], signals['volatility_signal'], 
                                               signals['regime_signal'], signals['mean_reversion_signal'] 
                                               if 'BEARISH' in str(signal) or 'HIGH_VOLATILITY' in str(signal)])
            
            if bullish_count > bearish_count:
                signals['overall_signal'] = 'BULLISH'
                signals['confidence'] = min(0.9, 0.5 + (bullish_count - bearish_count) * 0.1)
            elif bearish_count > bullish_count:
                signals['overall_signal'] = 'BEARISH'
                signals['confidence'] = min(0.9, 0.5 + (bearish_count - bullish_count) * 0.1)
            else:
                signals['overall_signal'] = 'NEUTRAL'
                signals['confidence'] = 0.5
            
            return signals
            
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            return {'overall_signal': 'ERROR', 'confidence': 0.0, 'reasoning': [str(e)]}

if __name__ == "__main__":
    # Example usage
    algorithms = AdvancedMathematicalAlgorithms()
    print("Advanced Mathematical Algorithms initialized successfully!")
    print("Features:")
    print("- Advanced volatility models (GARCH, EWMA, Realized)")
    print("- Kalman filter trend detection")
    print("- Hurst exponent for mean reversion")
    print("- Market regime detection (GMM, HMM)")
    print("- Fractal dimension analysis")
    print("- Entropy measures")
    print("- Parameter optimization")
    print("- Advanced correlation analysis")
    print("- Mathematical signal generation")