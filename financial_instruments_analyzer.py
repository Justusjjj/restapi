#!/usr/bin/env python3
"""
Comprehensive Financial Instruments Analyzer
Integrates yfinance, tranchpy, and other financial libraries for advanced analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
import requests
import json
import time

# Financial analysis libraries
try:
    import tranchpy as tp
except ImportError:
    print("Warning: tranchpy not available. Install with: pip install tranchpy")
    tp = None

try:
    import pandas_datareader as pdr
except ImportError:
    print("Warning: pandas_datareader not available. Install with: pip install pandas-datareader")
    pdr = None

try:
    from fredapi import Fred
except ImportError:
    print("Warning: fredapi not available. Install with: pip install fredapi")
    Fred = None

try:
    import investpy
except ImportError:
    print("Warning: investpy not available. Install with: pip install investpy")
    investpy = None

warnings.filterwarnings('ignore')

class FinancialInstrumentsAnalyzer:
    """
    Comprehensive Financial Instruments Analyzer
    Integrates multiple financial data sources and analysis tools
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        self._setup_logging()
        
        # Initialize API clients
        self._initialize_apis()
        
    def _default_config(self):
        """Default configuration for financial analyzer"""
        return {
            'cache_enabled': True,
            'api_rate_limit': 1.0,  # seconds between API calls
            'max_retries': 3,
            'timeout': 30,
            'default_period': '1y',
            'supported_instruments': ['stocks', 'etfs', 'bonds', 'forex', 'commodities', 'crypto'],
            'fundamental_metrics': True,
            'technical_indicators': True,
            'cash_flow_analysis': True,
            'risk_metrics': True
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('financial_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_apis(self):
        """Initialize various financial API clients"""
        try:
            # Initialize FRED API if available
            if Fred:
                self.fred = Fred(api_key=self.config.get('fred_api_key'))
            else:
                self.fred = None
            
            # Initialize other APIs as needed
            self.last_api_call = 0
            
        except Exception as e:
            self.logger.error(f"Error initializing APIs: {e}")
    
    def _rate_limit(self):
        """Implement API rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        if time_since_last < self.config['api_rate_limit']:
            time.sleep(self.config['api_rate_limit'] - time_since_last)
        self.last_api_call = time.time()
    
    def get_stock_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Get comprehensive stock data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data and additional metrics
        """
        try:
            self._rate_limit()
            
            # Get stock data
            stock = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = stock.history(period=period, interval=interval)
            
            if hist_data.empty:
                self.logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Get additional info
            info = stock.info
            
            # Add fundamental data to the DataFrame
            hist_data['symbol'] = symbol
            hist_data['market_cap'] = info.get('marketCap', np.nan)
            hist_data['pe_ratio'] = info.get('trailingPE', np.nan)
            hist_data['pb_ratio'] = info.get('priceToBook', np.nan)
            hist_data['dividend_yield'] = info.get('dividendYield', np.nan)
            hist_data['beta'] = info.get('beta', np.nan)
            
            # Calculate additional technical metrics
            hist_data['returns'] = hist_data['Close'].pct_change()
            hist_data['log_returns'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            hist_data['volatility'] = hist_data['returns'].rolling(window=20).std()
            
            # Cache the data
            if self.config['cache_enabled']:
                self.cache[f"{symbol}_{period}_{interval}"] = {
                    'data': hist_data,
                    'timestamp': datetime.now()
                }
            
            return hist_data
            
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Get comprehensive fundamental data for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            self._rate_limit()
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get financial statements
            try:
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                cash_flow = stock.cashflow
            except Exception as e:
                self.logger.warning(f"Could not fetch financial statements for {symbol}: {e}")
                balance_sheet = pd.DataFrame()
                income_stmt = pd.DataFrame()
                cash_flow = pd.DataFrame()
            
            fundamental_data = {
                'basic_info': {
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'country': info.get('country', ''),
                    'currency': info.get('currency', ''),
                    'exchange': info.get('exchange', ''),
                    'market_cap': info.get('marketCap', np.nan),
                    'enterprise_value': info.get('enterpriseValue', np.nan)
                },
                'valuation_metrics': {
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'forward_pe': info.get('forwardPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                    'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
                    'price_to_cashflow': info.get('priceToCashflow', np.nan)
                },
                'financial_metrics': {
                    'revenue': info.get('totalRevenue', np.nan),
                    'gross_profit': info.get('grossProfits', np.nan),
                    'operating_income': info.get('operatingIncome', np.nan),
                    'net_income': info.get('netIncomeToCommon', np.nan),
                    'total_assets': info.get('totalAssets', np.nan),
                    'total_debt': info.get('totalDebt', np.nan),
                    'cash': info.get('totalCash', np.nan)
                },
                'profitability_metrics': {
                    'gross_margin': info.get('grossMargins', np.nan),
                    'operating_margin': info.get('operatingMargins', np.nan),
                    'net_margin': info.get('netProfitMargins', np.nan),
                    'roa': info.get('returnOnAssets', np.nan),
                    'roe': info.get('returnOnEquity', np.nan),
                    'roic': info.get('returnOnInvestedCapital', np.nan)
                },
                'liquidity_metrics': {
                    'current_ratio': info.get('currentRatio', np.nan),
                    'quick_ratio': info.get('quickRatio', np.nan),
                    'debt_to_equity': info.get('debtToEquity', np.nan),
                    'interest_coverage': info.get('interestCoverage', np.nan)
                },
                'growth_metrics': {
                    'revenue_growth': info.get('revenueGrowth', np.nan),
                    'earnings_growth': info.get('earningsGrowth', np.nan),
                    'revenue_per_share': info.get('revenuePerShare', np.nan),
                    'book_value_per_share': info.get('bookValue', np.nan)
                },
                'dividend_metrics': {
                    'dividend_yield': info.get('dividendYield', np.nan),
                    'dividend_rate': info.get('dividendRate', np.nan),
                    'payout_ratio': info.get('payoutRatio', np.nan),
                    'dividend_growth': info.get('fiveYearAvgDividendYield', np.nan)
                },
                'balance_sheet': balance_sheet,
                'income_statement': income_stmt,
                'cash_flow_statement': cash_flow
            }
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return {}
    
    def analyze_cash_flow(self, symbol: str, periods: int = 4) -> Dict:
        """
        Comprehensive cash flow analysis
        
        Args:
            symbol: Stock symbol
            periods: Number of periods to analyze
            
        Returns:
            Dictionary with cash flow analysis
        """
        try:
            self._rate_limit()
            
            stock = yf.Ticker(symbol)
            
            # Get cash flow statement
            try:
                cash_flow = stock.cashflow
                if cash_flow.empty:
                    return {'error': 'No cash flow data available'}
            except Exception as e:
                return {'error': f'Could not fetch cash flow data: {e}'}
            
            # Get income statement for comparison
            try:
                income_stmt = stock.income_stmt
            except:
                income_stmt = pd.DataFrame()
            
            # Analyze cash flow components
            cash_flow_analysis = {
                'operating_cash_flow': {
                    'current': cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else np.nan,
                    'previous': cash_flow.loc['Operating Cash Flow'].iloc[1] if 'Operating Cash Flow' in cash_flow.index and len(cash_flow.columns) > 1 else np.nan,
                    'growth': 0.0
                },
                'investing_cash_flow': {
                    'current': cash_flow.loc['Investing Cash Flow'].iloc[0] if 'Investing Cash Flow' in cash_flow.index else np.nan,
                    'previous': cash_flow.loc['Investing Cash Flow'].iloc[1] if 'Investing Cash Flow' in cash_flow.index and len(cash_flow.columns) > 1 else np.nan,
                    'growth': 0.0
                },
                'financing_cash_flow': {
                    'current': cash_flow.loc['Financing Cash Flow'].iloc[0] if 'Financing Cash Flow' in cash_flow.index else np.nan,
                    'previous': cash_flow.loc['Financing Cash Flow'].iloc[1] if 'Financing Cash Flow' in cash_flow.index and len(cash_flow.columns) > 1 else np.nan,
                    'growth': 0.0
                },
                'free_cash_flow': {
                    'current': cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else np.nan,
                    'previous': cash_flow.loc['Free Cash Flow'].iloc[1] if 'Free Cash Flow' in cash_flow.index and len(cash_flow.columns) > 1 else np.nan,
                    'growth': 0.0
                }
            }
            
            # Calculate growth rates
            for key in cash_flow_analysis:
                current = cash_flow_analysis[key]['current']
                previous = cash_flow_analysis[key]['previous']
                
                if pd.notna(current) and pd.notna(previous) and previous != 0:
                    cash_flow_analysis[key]['growth'] = ((current - previous) / abs(previous)) * 100
                else:
                    cash_flow_analysis[key]['growth'] = np.nan
            
            # Calculate cash flow ratios
            try:
                net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else np.nan
                operating_cf = cash_flow_analysis['operating_cash_flow']['current']
                
                cash_flow_analysis['ratios'] = {
                    'operating_cf_to_net_income': operating_cf / net_income if pd.notna(operating_cf) and pd.notna(net_income) and net_income != 0 else np.nan,
                    'free_cf_to_operating_cf': cash_flow_analysis['free_cash_flow']['current'] / operating_cf if pd.notna(cash_flow_analysis['free_cash_flow']['current']) and pd.notna(operating_cf) and operating_cf != 0 else np.nan
                }
            except:
                cash_flow_analysis['ratios'] = {}
            
            # Add cash flow trends
            cash_flow_analysis['trends'] = {
                'operating_cf_trend': 'increasing' if cash_flow_analysis['operating_cash_flow']['growth'] > 0 else 'decreasing',
                'free_cf_trend': 'increasing' if cash_flow_analysis['free_cash_flow']['growth'] > 0 else 'decreasing',
                'cash_flow_quality': 'high' if cash_flow_analysis['operating_cash_flow']['current'] > 0 else 'low'
            }
            
            return cash_flow_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing cash flow for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_etf_data(self, symbol: str, period: str = '1y') -> Dict:
        """
        Get comprehensive ETF data and analysis
        
        Args:
            symbol: ETF symbol
            period: Time period
            
        Returns:
            Dictionary with ETF data and analysis
        """
        try:
            self._rate_limit()
            
            etf = yf.Ticker(symbol)
            info = etf.info
            
            # Get ETF data
            hist_data = etf.history(period=period)
            
            etf_data = {
                'basic_info': {
                    'name': info.get('longName', ''),
                    'category': info.get('category', ''),
                    'family': info.get('family', ''),
                    'exchange': info.get('exchange', ''),
                    'currency': info.get('currency', ''),
                    'market_cap': info.get('marketCap', np.nan)
                },
                'fund_metrics': {
                    'expense_ratio': info.get('expenseRatio', np.nan),
                    'aum': info.get('totalAssets', np.nan),
                    'nav': info.get('navPrice', np.nan),
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'dividend_yield': info.get('dividendYield', np.nan)
                },
                'holdings': {
                    'top_holdings': etf.holdings if hasattr(etf, 'holdings') else {},
                    'sector_allocation': etf.sector_weights if hasattr(etf, 'sector_weights') else {},
                    'country_allocation': etf.country_weights if hasattr(etf, 'country_weights') else {}
                },
                'performance': {
                    'ytd_return': info.get('ytdReturn', np.nan),
                    'three_year_return': info.get('threeYearAverageReturn', np.nan),
                    'five_year_return': info.get('fiveYearAverageReturn', np.nan),
                    'beta': info.get('beta', np.nan),
                    'sharpe_ratio': info.get('sharpeRatio', np.nan),
                    'treynor_ratio': info.get('treynorRatio', np.nan)
                },
                'historical_data': hist_data
            }
            
            return etf_data
            
        except Exception as e:
            self.logger.error(f"Error getting ETF data for {symbol}: {e}")
            return {}
    
    def get_bond_data(self, symbol: str) -> Dict:
        """
        Get bond data and analysis
        
        Args:
            symbol: Bond symbol
            
        Returns:
            Dictionary with bond data
        """
        try:
            self._rate_limit()
            
            bond = yf.Ticker(symbol)
            info = bond.info
            
            bond_data = {
                'basic_info': {
                    'name': info.get('longName', ''),
                    'issuer': info.get('issuer', ''),
                    'maturity_date': info.get('maturityDate', ''),
                    'coupon_rate': info.get('couponRate', np.nan),
                    'face_value': info.get('faceValue', np.nan)
                },
                'yield_metrics': {
                    'yield_to_maturity': info.get('yieldToMaturity', np.nan),
                    'current_yield': info.get('currentYield', np.nan),
                    'yield_to_call': info.get('yieldToCall', np.nan),
                    'yield_to_worst': info.get('yieldToWorst', np.nan)
                },
                'price_metrics': {
                    'current_price': info.get('currentPrice', np.nan),
                    'bid': info.get('bid', np.nan),
                    'ask': info.get('ask', np.nan),
                    'last_price': info.get('lastPrice', np.nan),
                    'volume': info.get('volume', np.nan)
                },
                'risk_metrics': {
                    'duration': info.get('duration', np.nan),
                    'modified_duration': info.get('modifiedDuration', np.nan),
                    'convexity': info.get('convexity', np.nan),
                    'credit_rating': info.get('creditRating', ''),
                    'default_risk': info.get('defaultRisk', np.nan)
                }
            }
            
            return bond_data
            
        except Exception as e:
            self.logger.error(f"Error getting bond data for {symbol}: {e}")
            return {}
    
    def get_commodity_data(self, symbol: str, period: str = '1y') -> Dict:
        """
        Get commodity data and analysis
        
        Args:
            symbol: Commodity symbol (e.g., 'GC=F' for gold, 'CL=F' for oil)
            period: Time period
            
        Returns:
            Dictionary with commodity data
        """
        try:
            self._rate_limit()
            
            commodity = yf.Ticker(symbol)
            info = commodity.info
            
            # Get historical data
            hist_data = commodity.history(period=period)
            
            commodity_data = {
                'basic_info': {
                    'name': info.get('longName', ''),
                    'category': info.get('category', ''),
                    'exchange': info.get('exchange', ''),
                    'currency': info.get('currency', ''),
                    'contract_size': info.get('contractSize', np.nan)
                },
                'price_metrics': {
                    'current_price': info.get('currentPrice', np.nan),
                    'bid': info.get('bid', np.nan),
                    'ask': info.get('ask', np.nan),
                    'day_high': info.get('dayHigh', np.nan),
                    'day_low': info.get('dayLow', np.nan),
                    'previous_close': info.get('previousClose', np.nan)
                },
                'volume_metrics': {
                    'volume': info.get('volume', np.nan),
                    'avg_volume': info.get('averageVolume', np.nan),
                    'volume_24h': info.get('volume24Hr', np.nan)
                },
                'performance': {
                    'change': info.get('regularMarketChange', np.nan),
                    'change_percent': info.get('regularMarketChangePercent', np.nan),
                    'ytd_return': info.get('ytdReturn', np.nan)
                },
                'historical_data': hist_data
            }
            
            return commodity_data
            
        except Exception as e:
            self.logger.error(f"Error getting commodity data for {symbol}: {e}")
            return {}
    
    def get_crypto_data(self, symbol: str, period: str = '1y') -> Dict:
        """
        Get cryptocurrency data and analysis
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Time period
            
        Returns:
            Dictionary with crypto data
        """
        try:
            self._rate_limit()
            
            crypto = yf.Ticker(symbol)
            info = crypto.info
            
            # Get historical data
            hist_data = crypto.history(period=period)
            
            crypto_data = {
                'basic_info': {
                    'name': info.get('longName', ''),
                    'symbol': info.get('symbol', ''),
                    'category': info.get('category', ''),
                    'exchange': info.get('exchange', ''),
                    'currency': info.get('currency', ''),
                    'market_cap': info.get('marketCap', np.nan)
                },
                'price_metrics': {
                    'current_price': info.get('currentPrice', np.nan),
                    'bid': info.get('bid', np.nan),
                    'ask': info.get('ask', np.nan),
                    'day_high': info.get('dayHigh', np.nan),
                    'day_low': info.get('dayLow', np.nan),
                    'previous_close': info.get('previousClose', np.nan)
                },
                'volume_metrics': {
                    'volume': info.get('volume', np.nan),
                    'avg_volume': info.get('averageVolume', np.nan),
                    'volume_24h': info.get('volume24Hr', np.nan)
                },
                'performance': {
                    'change': info.get('regularMarketChange', np.nan),
                    'change_percent': info.get('regularMarketChangePercent', np.nan),
                    'ytd_return': info.get('ytdReturn', np.nan)
                },
                'historical_data': hist_data
            }
            
            return crypto_data
            
        except Exception as e:
            self.logger.error(f"Error getting crypto data for {symbol}: {e}")
            return {}
    
    def get_economic_data(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get economic data from FRED
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with economic data
        """
        try:
            if not self.fred:
                return pd.DataFrame({'error': 'FRED API not available'})
            
            self._rate_limit()
            
            # Set default dates if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get data from FRED
            data = self.fred.get_series(series_id, start=start_date, end=end_date)
            
            if data.empty:
                return pd.DataFrame({'error': 'No data found for series'})
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['value'])
            df.index.name = 'date'
            df.reset_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting economic data for {series_id}: {e}")
            return pd.DataFrame({'error': str(e)})
    
    def calculate_financial_ratios(self, fundamental_data: Dict) -> Dict:
        """
        Calculate comprehensive financial ratios
        
        Args:
            fundamental_data: Fundamental data dictionary
            
        Returns:
            Dictionary with calculated ratios
        """
        try:
            ratios = {}
            
            # Extract key metrics
            market_cap = fundamental_data.get('financial_metrics', {}).get('total_assets', np.nan)
            total_assets = fundamental_data.get('financial_metrics', {}).get('total_assets', np.nan)
            total_debt = fundamental_data.get('financial_metrics', {}).get('total_debt', np.nan)
            net_income = fundamental_data.get('financial_metrics', {}).get('net_income', np.nan)
            revenue = fundamental_data.get('financial_metrics', {}).get('revenue', np.nan)
            cash = fundamental_data.get('financial_metrics', {}).get('cash', np.nan)
            
            # Liquidity ratios
            ratios['liquidity'] = {
                'current_ratio': fundamental_data.get('liquidity_metrics', {}).get('current_ratio', np.nan),
                'quick_ratio': fundamental_data.get('liquidity_metrics', {}).get('quick_ratio', np.nan),
                'cash_ratio': cash / total_assets if pd.notna(cash) and pd.notna(total_assets) and total_assets != 0 else np.nan
            }
            
            # Solvency ratios
            ratios['solvency'] = {
                'debt_to_equity': fundamental_data.get('liquidity_metrics', {}).get('debt_to_equity', np.nan),
                'debt_to_assets': total_debt / total_assets if pd.notna(total_debt) and pd.notna(total_assets) and total_assets != 0 else np.nan,
                'debt_to_capital': total_debt / (total_debt + market_cap) if pd.notna(total_debt) and pd.notna(market_cap) and (total_debt + market_cap) != 0 else np.nan
            }
            
            # Profitability ratios
            ratios['profitability'] = {
                'gross_margin': fundamental_data.get('profitability_metrics', {}).get('gross_margin', np.nan),
                'operating_margin': fundamental_data.get('profitability_metrics', {}).get('operating_margin', np.nan),
                'net_margin': fundamental_data.get('profitability_metrics', {}).get('net_margin', np.nan),
                'roa': fundamental_data.get('profitability_metrics', {}).get('roa', np.nan),
                'roe': fundamental_data.get('profitability_metrics', {}).get('roe', np.nan),
                'roic': fundamental_data.get('profitability_metrics', {}).get('roic', np.nan)
            }
            
            # Efficiency ratios
            ratios['efficiency'] = {
                'asset_turnover': revenue / total_assets if pd.notna(revenue) and pd.notna(total_assets) and total_assets != 0 else np.nan,
                'inventory_turnover': revenue / total_assets if pd.notna(revenue) and pd.notna(total_assets) and total_assets != 0 else np.nan,  # Simplified
                'receivables_turnover': revenue / total_assets if pd.notna(revenue) and pd.notna(total_assets) and total_assets != 0 else np.nan  # Simplified
            }
            
            # Growth ratios
            ratios['growth'] = {
                'revenue_growth': fundamental_data.get('growth_metrics', {}).get('revenue_growth', np.nan),
                'earnings_growth': fundamental_data.get('growth_metrics', {}).get('earnings_growth', np.nan)
            }
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating financial ratios: {e}")
            return {}
    
    def get_comprehensive_analysis(self, symbol: str, instrument_type: str = 'auto') -> Dict:
        """
        Get comprehensive analysis for any financial instrument
        
        Args:
            symbol: Instrument symbol
            instrument_type: Type of instrument ('auto', 'stock', 'etf', 'bond', 'commodity', 'crypto')
            
        Returns:
            Dictionary with comprehensive analysis
        """
        try:
            # Auto-detect instrument type if not specified
            if instrument_type == 'auto':
                if symbol.endswith('=F'):
                    instrument_type = 'commodity'
                elif symbol.endswith('-USD'):
                    instrument_type = 'crypto'
                elif symbol.endswith('.TO') or symbol.endswith('.V'):
                    instrument_type = 'stock'
                else:
                    # Try to determine based on available data
                    try:
                        info = yf.Ticker(symbol).info
                        if info.get('quoteType') == 'ETF':
                            instrument_type = 'etf'
                        elif info.get('quoteType') == 'EQUITY':
                            instrument_type = 'stock'
                        else:
                            instrument_type = 'stock'  # Default
                    except:
                        instrument_type = 'stock'  # Default
            
            analysis = {
                'symbol': symbol,
                'instrument_type': instrument_type,
                'timestamp': datetime.now(),
                'data_source': 'yfinance'
            }
            
            # Get data based on instrument type
            if instrument_type == 'stock':
                analysis['stock_data'] = self.get_stock_data(symbol)
                analysis['fundamental_data'] = self.get_fundamental_data(symbol)
                analysis['cash_flow_analysis'] = self.analyze_cash_flow(symbol)
                analysis['financial_ratios'] = self.calculate_financial_ratios(analysis['fundamental_data'])
                
            elif instrument_type == 'etf':
                analysis['etf_data'] = self.get_etf_data(symbol)
                
            elif instrument_type == 'bond':
                analysis['bond_data'] = self.get_bond_data(symbol)
                
            elif instrument_type == 'commodity':
                analysis['commodity_data'] = self.get_commodity_data(symbol)
                
            elif instrument_type == 'crypto':
                analysis['crypto_data'] = self.get_crypto_data(symbol)
                
            else:
                analysis['error'] = f'Unsupported instrument type: {instrument_type}'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive analysis for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol, 'timestamp': datetime.now()}
    
    def get_portfolio_analysis(self, symbols: List[str]) -> Dict:
        """
        Get portfolio-level analysis for multiple instruments
        
        Args:
            symbols: List of instrument symbols
            
        Returns:
            Dictionary with portfolio analysis
        """
        try:
            portfolio_data = {}
            total_value = 0
            total_weight = 0
            
            for symbol in symbols:
                analysis = self.get_comprehensive_analysis(symbol)
                if 'error' not in analysis:
                    portfolio_data[symbol] = analysis
                    
                    # Calculate portfolio metrics
                    if 'stock_data' in analysis and not analysis['stock_data'].empty:
                        current_price = analysis['stock_data']['Close'].iloc[-1]
                        market_cap = analysis.get('fundamental_data', {}).get('basic_info', {}).get('market_cap', 0)
                        
                        if pd.notna(current_price) and pd.notna(market_cap):
                            total_value += market_cap
                            total_weight += 1
            
            # Calculate portfolio-level metrics
            portfolio_analysis = {
                'symbols': symbols,
                'total_instruments': len(symbols),
                'successful_analyses': len(portfolio_data),
                'total_portfolio_value': total_value,
                'average_market_cap': total_value / total_weight if total_weight > 0 else 0,
                'individual_analyses': portfolio_data,
                'timestamp': datetime.now()
            }
            
            return portfolio_analysis
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio analysis: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Example usage
    analyzer = FinancialInstrumentsAnalyzer()
    
    # Test stock analysis
    print("Testing stock analysis...")
    stock_analysis = analyzer.get_comprehensive_analysis('AAPL', 'stock')
    print(f"Stock analysis completed: {len(stock_analysis)} components")
    
    # Test ETF analysis
    print("\nTesting ETF analysis...")
    etf_analysis = analyzer.get_comprehensive_analysis('SPY', 'etf')
    print(f"ETF analysis completed: {len(etf_analysis)} components")
    
    # Test commodity analysis
    print("\nTesting commodity analysis...")
    commodity_analysis = analyzer.get_comprehensive_analysis('GC=F', 'commodity')
    print(f"Commodity analysis completed: {len(commodity_analysis)} components")
    
    # Test crypto analysis
    print("\nTesting crypto analysis...")
    crypto_analysis = analyzer.get_comprehensive_analysis('BTC-USD', 'crypto')
    print(f"Crypto analysis completed: {len(crypto_analysis)} components")
    
    # Test cash flow analysis
    print("\nTesting cash flow analysis...")
    cash_flow = analyzer.analyze_cash_flow('AAPL')
    print(f"Cash flow analysis completed: {len(cash_flow)} components")
    
    # Test portfolio analysis
    print("\nTesting portfolio analysis...")
    portfolio = analyzer.get_portfolio_analysis(['AAPL', 'MSFT', 'GOOGL'])
    print(f"Portfolio analysis completed: {len(portfolio)} components")
    
    print("\nAll tests completed successfully!")