#!/usr/bin/env python3
"""
Enhanced Financial Indicators with Cash Flow Analysis
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional
import logging

warnings.filterwarnings('ignore')

class EnhancedFinancialIndicators:
    """Enhanced Financial Indicators with Cash Flow Analysis"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self._setup_logging()
        
    def _default_config(self):
        return {
            'default_period': 20,
            'risk_free_rate': 0.02,
            'volatility_window': 252
        }
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_cash_flow_indicators(self, cash_flow_data: pd.DataFrame) -> Dict:
        """Calculate cash flow indicators"""
        try:
            indicators = {}
            
            if cash_flow_data.empty:
                return {'error': 'No cash flow data provided'}
            
            # Operating Cash Flow
            if 'Operating Cash Flow' in cash_flow_data.index:
                ocf = cash_flow_data.loc['Operating Cash Flow']
                indicators['operating_cash_flow'] = {
                    'current': ocf.iloc[0] if len(ocf) > 0 else np.nan,
                    'growth_rate': self._calculate_growth_rate(ocf.iloc[0], ocf.iloc[1]) if len(ocf) > 1 else np.nan
                }
            
            # Free Cash Flow
            if 'Free Cash Flow' in cash_flow_data.index:
                fcf = cash_flow_data.loc['Free Cash Flow']
                indicators['free_cash_flow'] = {
                    'current': fcf.iloc[0] if len(fcf) > 0 else np.nan,
                    'growth_rate': self._calculate_growth_rate(fcf.iloc[0], fcf.iloc[1]) if len(fcf) > 1 else np.nan
                }
            
            # Cash Flow Quality
            indicators['cash_flow_quality'] = self._calculate_cash_flow_quality(cash_flow_data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating cash flow indicators: {e}")
            return {'error': str(e)}
    
    def _calculate_growth_rate(self, current: float, previous: float) -> float:
        """Calculate growth rate"""
        try:
            if pd.isna(current) or pd.isna(previous) or previous == 0:
                return np.nan
            return ((current - previous) / abs(previous)) * 100
        except:
            return np.nan
    
    def _calculate_cash_flow_quality(self, cash_flow_data: pd.DataFrame) -> Dict:
        """Calculate cash flow quality metrics"""
        try:
            quality_metrics = {}
            
            # OCF to Net Income ratio
            if 'Operating Cash Flow' in cash_flow_data.index and 'Net Income' in cash_flow_data.index:
                ocf = cash_flow_data.loc['Operating Cash Flow'].iloc[0]
                net_income = cash_flow_data.loc['Net Income'].iloc[0]
                
                if pd.notna(ocf) and pd.notna(net_income) and net_income != 0:
                    quality_metrics['ocf_to_net_income'] = ocf / net_income
                    
                    # Quality assessment
                    if quality_metrics['ocf_to_net_income'] > 1.0:
                        quality_metrics['quality_assessment'] = 'excellent'
                    elif quality_metrics['ocf_to_net_income'] > 0.8:
                        quality_metrics['quality_assessment'] = 'good'
                    elif quality_metrics['ocf_to_net_income'] > 0.6:
                        quality_metrics['quality_assessment'] = 'fair'
                    else:
                        quality_metrics['quality_assessment'] = 'poor'
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating cash flow quality: {e}")
            return {}
    
    def calculate_fundamental_ratios(self, financial_data: Dict) -> Dict:
        """Calculate fundamental financial ratios"""
        try:
            ratios = {}
            
            balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
            income_stmt = financial_data.get('income_statement', pd.DataFrame())
            
            # Liquidity Ratios
            ratios['liquidity'] = self._calculate_liquidity_ratios(balance_sheet)
            
            # Profitability Ratios
            ratios['profitability'] = self._calculate_profitability_ratios(income_stmt, balance_sheet)
            
            # Solvency Ratios
            ratios['solvency'] = self._calculate_solvency_ratios(balance_sheet)
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating fundamental ratios: {e}")
            return {}
    
    def _calculate_liquidity_ratios(self, balance_sheet: pd.DataFrame) -> Dict:
        """Calculate liquidity ratios"""
        try:
            ratios = {}
            
            if balance_sheet.empty:
                return ratios
            
            # Current Ratio
            if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
                
                if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities != 0:
                    ratios['current_ratio'] = current_assets / current_liabilities
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity ratios: {e}")
            return {}
    
    def _calculate_profitability_ratios(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        """Calculate profitability ratios"""
        try:
            ratios = {}
            
            if income_stmt.empty:
                return ratios
            
            # Gross Margin
            if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                gross_profit = income_stmt.loc['Gross Profit'].iloc[0]
                total_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                
                if pd.notna(gross_profit) and pd.notna(total_revenue) and total_revenue != 0:
                    ratios['gross_margin'] = gross_profit / total_revenue
            
            # Net Margin
            if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                total_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                
                if pd.notna(net_income) and pd.notna(total_revenue) and total_revenue != 0:
                    ratios['net_margin'] = net_income / total_revenue
            
            # ROE
            if 'Net Income' in income_stmt.index and 'Total Stockholder Equity' in balance_sheet.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                
                if pd.notna(net_income) and pd.notna(total_equity) and total_equity != 0:
                    ratios['roe'] = net_income / total_equity
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating profitability ratios: {e}")
            return {}
    
    def _calculate_solvency_ratios(self, balance_sheet: pd.DataFrame) -> Dict:
        """Calculate solvency ratios"""
        try:
            ratios = {}
            
            if balance_sheet.empty:
                return ratios
            
            # Debt to Equity
            if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                
                if pd.notna(total_debt) and pd.notna(total_equity) and total_equity != 0:
                    ratios['debt_to_equity'] = total_debt / total_equity
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating solvency ratios: {e}")
            return {}
    
    def calculate_risk_metrics(self, price_data: pd.DataFrame) -> Dict:
        """Calculate risk metrics"""
        try:
            risk_metrics = {}
            
            if price_data.empty or 'Close' not in price_data.columns:
                return {'error': 'No price data provided'}
            
            returns = price_data['Close'].pct_change().dropna()
            
            # Volatility
            risk_metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # VaR
            risk_metrics['var_95'] = np.percentile(returns, 5)
            
            # Max Drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            risk_metrics['max_drawdown'] = drawdown.min()
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_analysis(self, symbol: str, price_data: pd.DataFrame, financial_data: Dict) -> Dict:
        """Get comprehensive financial analysis"""
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now(),
                'cash_flow_indicators': {},
                'fundamental_ratios': {},
                'risk_metrics': {}
            }
            
            # Calculate indicators
            if 'cash_flow_statement' in financial_data:
                analysis['cash_flow_indicators'] = self.calculate_cash_flow_indicators(financial_data['cash_flow_statement'])
            
            analysis['fundamental_ratios'] = self.calculate_fundamental_ratios(financial_data)
            analysis['risk_metrics'] = self.calculate_risk_metrics(price_data)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive analysis: {e}")
            return {'error': str(e), 'symbol': symbol}

if __name__ == "__main__":
    # Example usage
    indicators = EnhancedFinancialIndicators()
    
    # Sample data
    cash_flow_data = pd.DataFrame({
        '2023': [1000000, -200000, 500000],
        '2022': [800000, -150000, 400000]
    }, index=['Operating Cash Flow', 'Investing Cash Flow', 'Free Cash Flow'])
    
    financial_data = {
        'balance_sheet': pd.DataFrame({
            '2023': [5000000, 3000000, 2000000, 1000000]
        }, index=['Total Assets', 'Total Current Assets', 'Total Debt', 'Total Stockholder Equity']),
        'income_statement': pd.DataFrame({
            '2023': [10000000, 6000000, 1000000]
        }, index=['Total Revenue', 'Cost Of Revenue', 'Net Income']),
        'cash_flow_statement': cash_flow_data
    }
    
    price_data = pd.DataFrame({
        'Close': np.random.normal(100, 5, 252)
    })
    
    # Test analysis
    analysis = indicators.get_comprehensive_analysis('AAPL', price_data, financial_data)
    print(f"Analysis completed: {len(analysis)} components")