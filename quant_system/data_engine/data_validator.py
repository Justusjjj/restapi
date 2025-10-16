"""
Data Validator
==============

Validates data quality and detects anomalies in market data.
Ensures data integrity for reliable backtesting and live trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates market data quality and detects anomalies.
    
    Validation checks:
    - Missing data detection
    - Price relationship validation (OHLC)
    - Spread validation
    - Outlier detection
    - Time gap detection
    - Volume validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataValidator.
        
        Args:
            config: Validation configuration parameters
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default validation configuration."""
        return {
            'max_spread_pct': 0.1,  # Maximum spread as % of price
            'min_volume': 0,  # Minimum volume threshold
            'max_price_change_pct': 0.05,  # Maximum single-bar price change
            'max_gap_minutes': 5,  # Maximum acceptable time gap in minutes
            'outlier_std_threshold': 3,  # Standard deviations for outlier detection
            'min_tick_count': 1,  # Minimum ticks per bar
        }
        
    def validate_ohlcv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLCV data quality.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for missing data
        missing_data = self._check_missing_data(df)
        if missing_data['count'] > 0:
            results['warnings'].append(f"Found {missing_data['count']} missing values")
            results['stats']['missing_data'] = missing_data
            
        # Validate OHLC relationships
        ohlc_errors = self._validate_ohlc_relationships(df)
        if ohlc_errors['count'] > 0:
            results['errors'].append(f"Found {ohlc_errors['count']} invalid OHLC relationships")
            results['valid'] = False
            results['stats']['ohlc_errors'] = ohlc_errors
            
        # Check for outliers
        outliers = self._detect_outliers(df)
        if outliers['count'] > 0:
            results['warnings'].append(f"Found {outliers['count']} potential outliers")
            results['stats']['outliers'] = outliers
            
        # Validate time gaps
        time_gaps = self._check_time_gaps(df)
        if time_gaps['count'] > 0:
            results['warnings'].append(f"Found {time_gaps['count']} time gaps")
            results['stats']['time_gaps'] = time_gaps
            
        # Validate volume
        volume_issues = self._validate_volume(df)
        if volume_issues['count'] > 0:
            results['warnings'].append(f"Found {volume_issues['count']} volume issues")
            results['stats']['volume_issues'] = volume_issues
            
        # Calculate basic statistics
        results['stats']['basic'] = self._calculate_basic_stats(df)
        
        logger.info(f"OHLCV validation complete: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
        
    def validate_tick_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate tick data quality.
        
        Args:
            df: Tick DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for missing data
        missing_data = self._check_missing_data(df)
        if missing_data['count'] > 0:
            results['warnings'].append(f"Found {missing_data['count']} missing values")
            results['stats']['missing_data'] = missing_data
            
        # Validate bid-ask relationships
        spread_errors = self._validate_spreads(df)
        if spread_errors['count'] > 0:
            results['errors'].append(f"Found {spread_errors['count']} invalid spreads")
            results['valid'] = False
            results['stats']['spread_errors'] = spread_errors
            
        # Check for duplicate timestamps
        duplicates = self._check_duplicate_timestamps(df)
        if duplicates['count'] > 0:
            results['warnings'].append(f"Found {duplicates['count']} duplicate timestamps")
            results['stats']['duplicates'] = duplicates
            
        # Validate tick frequency
        frequency_issues = self._validate_tick_frequency(df)
        if frequency_issues['count'] > 0:
            results['warnings'].append(f"Found {frequency_issues['count']} frequency issues")
            results['stats']['frequency_issues'] = frequency_issues
            
        # Calculate basic statistics
        results['stats']['basic'] = self._calculate_basic_stats(df)
        
        logger.info(f"Tick validation complete: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
        
    def _check_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing data in DataFrame."""
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        return {
            'count': total_missing,
            'by_column': missing.to_dict(),
            'percentage': (total_missing / (len(df) * len(df.columns))) * 100
        }
        
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate OHLC price relationships."""
        errors = []
        
        # High should be >= Low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            errors.extend(df[invalid_hl].index.tolist())
            
        # High should be >= Open
        invalid_ho = df['high'] < df['open']
        if invalid_ho.any():
            errors.extend(df[invalid_ho].index.tolist())
            
        # High should be >= Close
        invalid_hc = df['high'] < df['close']
        if invalid_hc.any():
            errors.extend(df[invalid_hc].index.tolist())
            
        # Low should be <= Open
        invalid_lo = df['low'] > df['open']
        if invalid_lo.any():
            errors.extend(df[invalid_lo].index.tolist())
            
        # Low should be <= Close
        invalid_lc = df['low'] > df['close']
        if invalid_lc.any():
            errors.extend(df[invalid_lc].index.tolist())
            
        return {
            'count': len(errors),
            'indices': errors
        }
        
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect price outliers using statistical methods."""
        outliers = []
        
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Calculate z-scores
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > self.config['outlier_std_threshold']
                
                if outlier_mask.any():
                    outliers.extend(df[outlier_mask].index.tolist())
                    
        return {
            'count': len(outliers),
            'indices': outliers
        }
        
    def _check_time_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for unexpected time gaps in data."""
        if len(df) < 2:
            return {'count': 0, 'gaps': []}
            
        # Calculate time differences
        time_diffs = df.index.to_series().diff()
        
        # Expected frequency (infer from most common difference)
        expected_freq = time_diffs.mode().iloc[0] if not time_diffs.empty else pd.Timedelta(minutes=1)
        max_gap = pd.Timedelta(minutes=self.config['max_gap_minutes'])
        
        # Find gaps larger than expected
        large_gaps = time_diffs > max_gap
        gap_indices = df.index[large_gaps].tolist()
        
        return {
            'count': len(gap_indices),
            'gaps': gap_indices,
            'expected_frequency': str(expected_freq)
        }
        
    def _validate_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate volume data."""
        issues = []
        
        if 'volume' in df.columns:
            # Check for negative volume
            negative_volume = df['volume'] < 0
            if negative_volume.any():
                issues.extend(df[negative_volume].index.tolist())
                
            # Check for zero volume (might be valid for some instruments)
            zero_volume = df['volume'] == 0
            if zero_volume.any():
                issues.extend(df[zero_volume].index.tolist())
                
        return {
            'count': len(issues),
            'indices': issues
        }
        
    def _validate_spreads(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate bid-ask spreads."""
        errors = []
        
        if 'bid' in df.columns and 'ask' in df.columns:
            # Check for negative spreads
            negative_spreads = df['ask'] < df['bid']
            if negative_spreads.any():
                errors.extend(df[negative_spreads].index.tolist())
                
            # Check for unreasonably large spreads
            spreads = df['ask'] - df['bid']
            mid_prices = (df['ask'] + df['bid']) / 2
            spread_pct = (spreads / mid_prices) * 100
            
            large_spreads = spread_pct > self.config['max_spread_pct']
            if large_spreads.any():
                errors.extend(df[large_spreads].index.tolist())
                
        return {
            'count': len(errors),
            'indices': errors
        }
        
    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate timestamps."""
        duplicates = df.index.duplicated()
        
        return {
            'count': duplicates.sum(),
            'indices': df.index[duplicates].tolist()
        }
        
    def _validate_tick_frequency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate tick frequency and detect unusual patterns."""
        if len(df) < 2:
            return {'count': 0, 'issues': []}
            
        # Calculate time differences between ticks
        time_diffs = df.index.to_series().diff().dt.total_seconds()
        
        # Detect unusually long gaps (more than 1 second for tick data)
        long_gaps = time_diffs > 1.0
        gap_indices = df.index[long_gaps].tolist()
        
        return {
            'count': len(gap_indices),
            'issues': gap_indices
        }
        
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for the data."""
        stats = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min().isoformat() if not df.empty else None,
                'end': df.index.max().isoformat() if not df.empty else None
            },
            'columns': list(df.columns)
        }
        
        # Add price statistics if available
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]
        
        if available_price_cols:
            stats['price_stats'] = {}
            for col in available_price_cols:
                stats['price_stats'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
                
        return stats