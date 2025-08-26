#!/usr/bin/env python3
"""
Advanced Mathematical Tools for Forex Trading
Features:
- Fractal Analysis and Chaos Theory
- Advanced Statistical Methods
- Mathematical Pattern Recognition
- Complex Mathematical Indicators
- Chaos Theory Applications
- Advanced Optimization Algorithms
- Mathematical Market Models
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, signal
from scipy.stats import norm, skew, kurtosis, chi2, f
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d, splrep, splev
from scipy.linalg import eig, inv, det
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

class AdvancedMathematicalTools:
    def __init__(self, config: Dict):
        """Initialize advanced mathematical tools"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fractal_cache = {}
        self.chaos_cache = {}
        self.pattern_cache = {}
        
    def calculate_fractal_dimension(self, data: pd.Series, method: str = "box_counting") -> Dict:
        """Calculate fractal dimension using multiple methods"""
        try:
            results = {}
            
            if method == "box_counting" or method == "all":
                results["box_counting"] = self._box_counting_fractal(data)
            
            if method == "higuchi" or method == "all":
                results["higuchi"] = self._higuchi_fractal(data)
            
            if method == "katz" or method == "all":
                results["katz"] = self._katz_fractal(data)
            
            if method == "petrosian" or method == "all":
                results["petrosian"] = self._petrosian_fractal(data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return {"error": str(e)}
    
    def _box_counting_fractal(self, data: pd.Series) -> Dict:
        """Box-counting fractal dimension"""
        try:
            # Normalize data
            normalized = (data - data.min()) / (data.max() - data.min())
            
            # Different box sizes
            box_sizes = np.logspace(-3, 0, 20)
            box_counts = []
            
            for size in box_sizes:
                # Count boxes needed to cover the curve
                x_bins = int(1 / size)
                y_bins = int(1 / size)
                
                # Create grid
                grid = np.zeros((x_bins, y_bins))
                
                # Fill grid based on data
                for i, (x, y) in enumerate(zip(np.linspace(0, 1, len(normalized)), normalized)):
                    x_idx = int(x * x_bins)
                    y_idx = int(y * y_bins)
                    if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                        grid[x_idx, y_idx] = 1
                
                # Count non-empty boxes
                count = np.sum(grid > 0)
                box_counts.append(count)
            
            # Linear regression on log-log plot
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)
            
            # Remove zeros and infinities
            valid_mask = np.isfinite(log_counts) & np.isfinite(log_sizes)
            if np.sum(valid_mask) < 2:
                return {"fractal_dimension": 1.0, "error": "Insufficient valid data"}
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_sizes[valid_mask], log_counts[valid_mask]
            )
            
            fractal_dimension = -slope
            
            return {
                "fractal_dimension": fractal_dimension,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_error": std_err,
                "method": "box_counting"
            }
            
        except Exception as e:
            self.logger.error(f"Error in box counting fractal: {e}")
            return {"fractal_dimension": 1.0, "error": str(e)}
    
    def _higuchi_fractal(self, data: pd.Series) -> Dict:
        """Higuchi fractal dimension method"""
        try:
            # Higuchi method for fractal dimension
            k_max = min(20, len(data) // 4)
            k_values = np.arange(1, k_max + 1)
            l_values = []
            
            for k in k_values:
                l_k = 0
                for m in range(k):
                    # Calculate L(m,k)
                    l_mk = 0
                    for i in range(1, int((len(data) - m) / k)):
                        l_mk += abs(data.iloc[m + i * k] - data.iloc[m + (i - 1) * k])
                    
                    l_mk = l_mk * (len(data) - 1) / (k**2 * int((len(data) - m) / k))
                    l_k += l_mk
                
                l_k /= k
                l_values.append(l_k)
            
            # Linear regression
            log_k = np.log(k_values)
            log_l = np.log(l_values)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_l)
            fractal_dimension = 1 - slope
            
            return {
                "fractal_dimension": fractal_dimension,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_error": std_err,
                "method": "higuchi"
            }
            
        except Exception as e:
            self.logger.error(f"Error in Higuchi fractal: {e}")
            return {"fractal_dimension": 1.0, "error": str(e)}
    
    def _katz_fractal(self, data: pd.Series) -> Dict:
        """Katz fractal dimension method"""
        try:
            # Katz method
            # Calculate total length and maximum distance
            total_length = 0
            max_distance = 0
            
            for i in range(1, len(data)):
                diff = abs(data.iloc[i] - data.iloc[i-1])
                total_length += diff
                
                distance = np.sqrt(i**2 + diff**2)
                max_distance = max(max_distance, distance)
            
            # Katz fractal dimension
            fractal_dimension = np.log(len(data)) / np.log(max_distance / total_length)
            
            return {
                "fractal_dimension": fractal_dimension,
                "total_length": total_length,
                "max_distance": max_distance,
                "method": "katz"
            }
            
        except Exception as e:
            self.logger.error(f"Error in Katz fractal: {e}")
            return {"fractal_dimension": 1.0, "error": str(e)}
    
    def _petrosian_fractal(self, data: pd.Series) -> Dict:
        """Petrosian fractal dimension method"""
        try:
            # Petrosian method
            # Calculate number of sign changes
            sign_changes = 0
            for i in range(1, len(data)):
                if (data.iloc[i] - data.iloc[i-1]) * (data.iloc[i-1] - data.iloc[i-2]) < 0:
                    sign_changes += 1
            
            # Petrosian fractal dimension
            fractal_dimension = np.log(len(data)) / (np.log(len(data)) + np.log(len(data) / (len(data) + 0.4 * sign_changes)))
            
            return {
                "fractal_dimension": fractal_dimension,
                "sign_changes": sign_changes,
                "method": "petrosian"
            }
            
        except Exception as e:
            self.logger.error(f"Error in Petrosian fractal: {e}")
            return {"fractal_dimension": 1.0, "error": str(e)}
    
    def calculate_lyapunov_exponent(self, data: pd.Series, embedding_dim: int = 3, delay: int = 1) -> Dict:
        """Calculate Lyapunov exponent for chaos detection"""
        try:
            # Phase space reconstruction
            n_points = len(data) - (embedding_dim - 1) * delay
            phase_space = np.zeros((n_points, embedding_dim))
            
            for i in range(n_points):
                for j in range(embedding_dim):
                    phase_space[i, j] = data.iloc[i + j * delay]
            
            # Calculate Lyapunov exponent
            lyap_exp = 0
            n_neighbors = min(10, n_points // 10)
            
            for i in range(n_points):
                # Find nearest neighbors
                distances = np.linalg.norm(phase_space - phase_space[i], axis=1)
                nearest_indices = np.argsort(distances)[1:n_neighbors+1]
                
                for j in nearest_indices:
                    if j < n_points - 1:
                        # Calculate separation
                        initial_sep = np.linalg.norm(phase_space[i] - phase_space[j])
                        final_sep = np.linalg.norm(phase_space[i+1] - phase_space[j+1])
                        
                        if initial_sep > 0 and final_sep > 0:
                            lyap_exp += np.log(final_sep / initial_sep)
            
            lyap_exp /= (n_points * n_neighbors)
            
            # Determine chaos
            is_chaotic = lyap_exp > 0.1
            
            return {
                "lyapunov_exponent": lyap_exp,
                "is_chaotic": is_chaotic,
                "embedding_dimension": embedding_dim,
                "delay": delay,
                "method": "phase_space_reconstruction"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Lyapunov exponent: {e}")
            return {"lyapunov_exponent": 0, "is_chaotic": False, "error": str(e)}
    
    def calculate_hurst_exponent(self, data: pd.Series, method: str = "rs") -> Dict:
        """Calculate Hurst exponent for long-term memory detection"""
        try:
            if method == "rs":
                return self._rs_hurst(data)
            elif method == "aggregate_variance":
                return self._aggregate_variance_hurst(data)
            elif method == "differenced_variance":
                return self._differenced_variance_hurst(data)
            else:
                return self._rs_hurst(data)
                
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return {"hurst_exponent": 0.5, "error": str(e)}
    
    def _rs_hurst(self, data: pd.Series) -> Dict:
        """R/S (Rescaled Range) method for Hurst exponent"""
        try:
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Different time scales
            scales = [10, 20, 50, 100, 200]
            rs_values = []
            
            for scale in scales:
                if len(returns) < scale * 2:
                    continue
                
                rs_scale = []
                for i in range(0, len(returns) - scale, scale):
                    segment = returns.iloc[i:i+scale]
                    
                    # Calculate R (range)
                    cumulative = segment.cumsum()
                    r = cumulative.max() - cumulative.min()
                    
                    # Calculate S (standard deviation)
                    s = segment.std()
                    
                    if s > 0:
                        rs_scale.append(r / s)
                
                if rs_scale:
                    rs_values.append(np.mean(rs_scale))
                else:
                    rs_values.append(np.nan)
            
            # Remove NaN values
            valid_scales = [scales[i] for i in range(len(scales)) if not np.isnan(rs_values[i])]
            valid_rs = [rs_values[i] for i in range(len(rs_values)) if not np.isnan(rs_values[i])]
            
            if len(valid_scales) < 2:
                return {"hurst_exponent": 0.5, "error": "Insufficient data"}
            
            # Linear regression
            log_scales = np.log(valid_scales)
            log_rs = np.log(valid_rs)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_rs)
            hurst_exponent = slope
            
            return {
                "hurst_exponent": hurst_exponent,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_error": std_err,
                "method": "rs",
                "scales": valid_scales,
                "rs_values": valid_rs
            }
            
        except Exception as e:
            self.logger.error(f"Error in R/S Hurst: {e}")
            return {"hurst_exponent": 0.5, "error": str(e)}
    
    def _aggregate_variance_hurst(self, data: pd.Series) -> Dict:
        """Aggregate variance method for Hurst exponent"""
        try:
            # Different aggregation levels
            scales = [2, 4, 8, 16, 32, 64]
            variances = []
            
            for scale in scales:
                if len(data) < scale * 2:
                    continue
                
                # Aggregate data
                aggregated = []
                for i in range(0, len(data), scale):
                    if i + scale <= len(data):
                        aggregated.append(data.iloc[i:i+scale].mean())
                
                if len(aggregated) > 1:
                    variances.append(np.var(aggregated))
                else:
                    variances.append(np.nan)
            
            # Remove NaN values
            valid_scales = [scales[i] for i in range(len(scales)) if not np.isnan(variances[i])]
            valid_variances = [variances[i] for i in range(len(variances)) if not np.isnan(variances[i])]
            
            if len(valid_scales) < 2:
                return {"hurst_exponent": 0.5, "error": "Insufficient data"}
            
            # Linear regression
            log_scales = np.log(valid_scales)
            log_variances = np.log(valid_variances)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_variances)
            hurst_exponent = 1 - slope / 2
            
            return {
                "hurst_exponent": hurst_exponent,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_error": std_err,
                "method": "aggregate_variance"
            }
            
        except Exception as e:
            self.logger.error(f"Error in aggregate variance Hurst: {e}")
            return {"hurst_exponent": 0.5, "error": str(e)}
    
    def calculate_entropy_measures(self, data: pd.Series) -> Dict:
        """Calculate various entropy measures"""
        try:
            results = {}
            
            # Sample entropy
            results["sample_entropy"] = self._calculate_sample_entropy(data)
            
            # Approximate entropy
            results["approximate_entropy"] = self._calculate_approximate_entropy(data)
            
            # Permutation entropy
            results["permutation_entropy"] = self._calculate_permutation_entropy(data)
            
            # Shannon entropy
            results["shannon_entropy"] = self._calculate_shannon_entropy(data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy measures: {e}")
            return {"error": str(e)}
    
    def _calculate_sample_entropy(self, data: pd.Series, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy"""
        try:
            # Normalize data
            normalized = (data - data.mean()) / data.std()
            
            # Calculate template matches
            n = len(normalized)
            a, b = 0, 0
            
            for i in range(n - m):
                for j in range(i + 1, n - m):
                    # Check if templates match
                    if np.max(np.abs(normalized.iloc[i:i+m] - normalized.iloc[j:j+m])) <= r:
                        a += 1
                        if np.abs(normalized.iloc[i+m] - normalized.iloc[j+m]) <= r:
                            b += 1
            
            if a == 0:
                return 0
            
            return -np.log(b / a)
            
        except Exception as e:
            self.logger.error(f"Error in sample entropy: {e}")
            return 0
    
    def _calculate_approximate_entropy(self, data: pd.Series, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy"""
        try:
            # Normalize data
            normalized = (data - data.mean()) / data.std()
            
            # Calculate phi(m) and phi(m+1)
            phi_m = self._calculate_phi(normalized, m, r)
            phi_m1 = self._calculate_phi(normalized, m + 1, r)
            
            return phi_m - phi_m1
            
        except Exception as e:
            self.logger.error(f"Error in approximate entropy: {e}")
            return 0
    
    def _calculate_phi(self, data: pd.Series, m: int, r: float) -> float:
        """Helper function for approximate entropy"""
        try:
            n = len(data)
            count = 0
            
            for i in range(n - m + 1):
                for j in range(i + 1, n - m + 1):
                    if np.max(np.abs(data.iloc[i:i+m] - data.iloc[j:j+m])) <= r:
                        count += 1
            
            return np.log(count / (n - m + 1))
            
        except Exception as e:
            self.logger.error(f"Error in phi calculation: {e}")
            return 0
    
    def _calculate_permutation_entropy(self, data: pd.Series, m: int = 3) -> float:
        """Calculate permutation entropy"""
        try:
            n = len(data)
            permutations = {}
            
            for i in range(n - m + 1):
                # Get permutation pattern
                segment = data.iloc[i:i+m]
                pattern = tuple(np.argsort(segment))
                
                if pattern in permutations:
                    permutations[pattern] += 1
                else:
                    permutations[pattern] = 1
            
            # Calculate entropy
            total = sum(permutations.values())
            entropy = 0
            
            for count in permutations.values():
                p = count / total
                entropy -= p * np.log(p)
            
            return entropy / np.log(np.math.factorial(m))
            
        except Exception as e:
            self.logger.error(f"Error in permutation entropy: {e}")
            return 0
    
    def _calculate_shannon_entropy(self, data: pd.Series, bins: int = 20) -> float:
        """Calculate Shannon entropy"""
        try:
            # Create histogram
            hist, _ = np.histogram(data, bins=bins, density=True)
            
            # Calculate entropy
            entropy = 0
            for p in hist:
                if p > 0:
                    entropy -= p * np.log(p)
            
            return entropy
            
        except Exception as e:
            self.logger.error(f"Error in Shannon entropy: {e}")
            return 0
    
    def calculate_advanced_statistics(self, data: pd.Series) -> Dict:
        """Calculate advanced statistical measures"""
        try:
            results = {}
            
            # Basic statistics
            results["mean"] = data.mean()
            results["std"] = data.std()
            results["skewness"] = data.skew()
            results["kurtosis"] = data.kurtosis()
            
            # Advanced moments
            results["variance"] = data.var()
            results["coefficient_of_variation"] = data.std() / abs(data.mean()) if data.mean() != 0 else 0
            
            # Percentiles
            results["percentiles"] = {
                "p1": data.quantile(0.01),
                "p5": data.quantile(0.05),
                "p25": data.quantile(0.25),
                "p50": data.quantile(0.50),
                "p75": data.quantile(0.75),
                "p95": data.quantile(0.95),
                "p99": data.quantile(0.99)
            }
            
            # Robust statistics
            results["median"] = data.median()
            results["mad"] = stats.median_abs_deviation(data)
            results["iqr"] = data.quantile(0.75) - data.quantile(0.25)
            
            # Distribution tests
            results["normality_tests"] = self._run_normality_tests(data)
            
            # Autocorrelation
            results["autocorrelation"] = self._calculate_autocorrelation(data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced statistics: {e}")
            return {"error": str(e)}
    
    def _run_normality_tests(self, data: pd.Series) -> Dict:
        """Run various normality tests"""
        try:
            tests = {}
            
            # Shapiro-Wilk test
            tests["shapiro"] = stats.shapiro(data)
            
            # Anderson-Darling test
            tests["anderson"] = stats.anderson(data)
            
            # D'Agostino K^2 test
            tests["dagostino"] = stats.normaltest(data)
            
            # Jarque-Bera test
            tests["jarque_bera"] = stats.jarque_bera(data)
            
            return tests
            
        except Exception as e:
            self.logger.error(f"Error in normality tests: {e}")
            return {"error": str(e)}
    
    def _calculate_autocorrelation(self, data: pd.Series, max_lag: int = 20) -> Dict:
        """Calculate autocorrelation function"""
        try:
            lags = range(1, min(max_lag + 1, len(data) // 4))
            autocorr = []
            
            for lag in lags:
                if lag < len(data):
                    corr = data.autocorr(lag=lag)
                    autocorr.append(corr)
                else:
                    autocorr.append(np.nan)
            
            # Remove NaN values
            valid_lags = [lags[i] for i in range(len(lags)) if not np.isnan(autocorr[i])]
            valid_autocorr = [autocorr[i] for i in range(len(autocorr)) if not np.isnan(autocorr[i])]
            
            return {
                "lags": valid_lags,
                "autocorrelation": valid_autocorr,
                "max_autocorr": max(valid_autocorr) if valid_autocorr else 0,
                "min_autocorr": min(valid_autocorr) if valid_autocorr else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in autocorrelation: {e}")
            return {"error": str(e)}
    
    def calculate_mathematical_patterns(self, data: pd.Series) -> Dict:
        """Detect mathematical patterns in data"""
        try:
            patterns = {}
            
            # Fibonacci retracements
            patterns["fibonacci"] = self._calculate_fibonacci_levels(data)
            
            # Golden ratio levels
            patterns["golden_ratio"] = self._calculate_golden_ratio_levels(data)
            
            # Harmonic patterns
            patterns["harmonics"] = self._detect_harmonic_patterns(data)
            
            # Elliott Wave structure
            patterns["elliott_wave"] = self._analyze_elliott_wave(data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting mathematical patterns: {e}")
            return {"error": str(e)}
    
    def _calculate_fibonacci_levels(self, data: pd.Series) -> Dict:
        """Calculate Fibonacci retracement levels"""
        try:
            high = data.max()
            low = data.min()
            diff = high - low
            
            levels = {
                "0.0": low,
                "0.236": low + 0.236 * diff,
                "0.382": low + 0.382 * diff,
                "0.500": low + 0.500 * diff,
                "0.618": low + 0.618 * diff,
                "0.786": low + 0.786 * diff,
                "1.0": high
            }
            
            return {
                "levels": levels,
                "high": high,
                "low": low,
                "range": diff
            }
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci levels: {e}")
            return {"error": str(e)}
    
    def _calculate_golden_ratio_levels(self, data: pd.Series) -> Dict:
        """Calculate Golden Ratio levels"""
        try:
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
            high = data.max()
            low = data.min()
            diff = high - low
            
            levels = {
                "0.0": low,
                "0.382": low + (1 - 1/phi) * diff,
                "0.500": low + 0.5 * diff,
                "0.618": low + (1/phi) * diff,
                "1.0": high
            }
            
            return {
                "levels": levels,
                "golden_ratio": phi,
                "high": high,
                "low": low
            }
            
        except Exception as e:
            self.logger.error(f"Error in Golden Ratio levels: {e}")
            return {"error": str(e)}
    
    def _detect_harmonic_patterns(self, data: pd.Series) -> Dict:
        """Detect harmonic patterns (Gartley, Butterfly, etc.)"""
        try:
            # Simplified harmonic pattern detection
            # In practice, this would be much more complex
            
            patterns = {
                "gartley": False,
                "butterfly": False,
                "bat": False,
                "crab": False
            }
            
            # Basic harmonic ratio check
            if len(data) >= 5:
                # Check for potential harmonic pattern
                ratios = []
                for i in range(1, len(data)):
                    if i > 0 and data.iloc[i-1] != 0:
                        ratio = abs(data.iloc[i] / data.iloc[i-1])
                        ratios.append(ratio)
                
                # Check for harmonic ratios
                harmonic_ratios = [0.618, 1.618, 2.618, 3.618]
                harmonic_count = 0
                
                for ratio in ratios:
                    for harmonic in harmonic_ratios:
                        if abs(ratio - harmonic) < 0.1:  # 10% tolerance
                            harmonic_count += 1
                            break
                
                if harmonic_count >= 2:
                    patterns["gartley"] = True
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in harmonic patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_elliott_wave(self, data: pd.Series) -> Dict:
        """Basic Elliott Wave analysis"""
        try:
            # Simplified Elliott Wave detection
            # In practice, this requires sophisticated pattern recognition
            
            # Find peaks and troughs
            peaks, _ = find_peaks(data.values)
            troughs, _ = find_peaks(-data.values)
            
            # Basic wave structure
            waves = {
                "wave_1": None,
                "wave_2": None,
                "wave_3": None,
                "wave_4": None,
                "wave_5": None
            }
            
            if len(peaks) >= 3 and len(troughs) >= 2:
                # Simple wave identification
                waves["wave_1"] = data.iloc[peaks[0]] if len(peaks) > 0 else None
                waves["wave_3"] = data.iloc[peaks[1]] if len(peaks) > 1 else None
                waves["wave_5"] = data.iloc[peaks[2]] if len(peaks) > 2 else None
                
                if len(troughs) > 0:
                    waves["wave_2"] = data.iloc[troughs[0]]
                if len(troughs) > 1:
                    waves["wave_4"] = data.iloc[troughs[1]]
            
            return {
                "waves": waves,
                "peaks_count": len(peaks),
                "troughs_count": len(troughs),
                "pattern_complete": all(wave is not None for wave in waves.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error in Elliott Wave analysis: {e}")
            return {"error": str(e)}
    
    def generate_mathematical_report(self, data: pd.Series) -> Dict:
        """Generate comprehensive mathematical analysis report"""
        try:
            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "fractal_analysis": self.calculate_fractal_dimension(data, "all"),
                "chaos_analysis": self.calculate_lyapunov_exponent(data),
                "hurst_analysis": self.calculate_hurst_exponent(data, "all"),
                "entropy_analysis": self.calculate_entropy_measures(data),
                "statistical_analysis": self.calculate_advanced_statistics(data),
                "pattern_analysis": self.calculate_mathematical_patterns(data),
                "recommendations": []
            }
            
            # Generate recommendations based on mathematical analysis
            recommendations = self._generate_mathematical_recommendations(report)
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating mathematical report: {e}")
            return {"error": str(e)}
    
    def _generate_mathematical_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on mathematical analysis"""
        try:
            recommendations = []
            
            # Fractal analysis recommendations
            if "fractal_analysis" in report:
                fractal_results = report["fractal_analysis"]
                if "box_counting" in fractal_results:
                    fd = fractal_results["box_counting"].get("fractal_dimension", 1.0)
                    if fd > 1.5:
                        recommendations.append("High fractal dimension detected - consider fractal-based trading strategies")
            
            # Chaos analysis recommendations
            if "chaos_analysis" in report:
                chaos_results = report["chaos_analysis"]
                if chaos_results.get("is_chaotic", False):
                    recommendations.append("Chaotic behavior detected - use chaos theory based indicators")
            
            # Hurst exponent recommendations
            if "hurst_analysis" in report:
                hurst_results = report["hurst_analysis"]
                if "rs" in hurst_results:
                    h = hurst_results["rs"].get("hurst_exponent", 0.5)
                    if h > 0.7:
                        recommendations.append("High Hurst exponent - strong trend following behavior")
                    elif h < 0.3:
                        recommendations.append("Low Hurst exponent - mean reversion behavior")
            
            # Entropy recommendations
            if "entropy_analysis" in report:
                entropy_results = report["entropy_analysis"]
                if "sample_entropy" in entropy_results:
                    se = entropy_results["sample_entropy"]
                    if se > 2.0:
                        recommendations.append("High sample entropy - complex market structure")
                    elif se < 0.5:
                        recommendations.append("Low sample entropy - predictable market patterns")
            
            if not recommendations:
                recommendations.append("Mathematical analysis shows balanced market characteristics")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]