#!/usr/bin/env python3
"""
🚀 ENHANCED FOREX TRADING SYSTEM 🚀
Advanced AI-Powered Trading Bot with Real-Time Market Analysis

Features:
- Advanced Pattern Recognition (Head & Shoulders, Triangles, Flags)
- Market Regime Detection (Trending, Ranging, Volatile)
- Multi-Asset Portfolio Optimization
- Real-Time Sentiment Analysis
- Advanced Risk Management with Dynamic Position Sizing
- Machine Learning Price Prediction Models
- News Impact Analysis
- Automated Strategy Selection
- Real-Time Performance Monitoring
"""

import os
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

# Advanced Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

# Advanced Technical Analysis
import ta
import pandas_ta as pta
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import minimize

# Advanced Mathematics and Statistics
from scipy.stats import norm, skew, kurtosis
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from arch import arch_model

# News and Sentiment Analysis
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

# Portfolio Optimization
from scipy.optimize import minimize

# Configuration
load_dotenv()

class EnhancedForexTradingSystem:
    def __init__(self, config_path: str = "bot_config.json"):
        """Initialize the enhanced forex trading system"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_exchanges()
        self.setup_advanced_ml_models()
        self.setup_portfolio_optimizer()
        
        # Enhanced trading state
        self.active_trades = {}
        self.portfolio_positions = {}
        self.performance_metrics = {}
        self.market_regime = "unknown"
        self.pattern_signals = {}
        self.sentiment_scores = {}
        
        # Advanced caches
        self.news_cache = {}
        self.sentiment_cache = {}
        self.model_cache = {}
        self.scaler_cache = {}
        self.feature_cache = {}
        self.pattern_cache = {}
        
        # Threading and async
        self.running = False
        self.trading_thread = None
        self.news_thread = None
        self.analysis_thread = None
        self.optimization_thread = None
        
        # Performance tracking
        self.trade_history = []
        self.performance_history = []
        self.risk_metrics = {}
        
        # Market data
        self.market_data = {}
        self.technical_indicators = {}
        self.pattern_recognition = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Load enhanced bot configuration"""
        default_config = {
            "exchanges": {
                "binance": {
                    "api_key": os.getenv("BINANCE_API_KEY", ""),
                    "secret": os.getenv("BINANCE_SECRET", ""),
                    "sandbox": True
                },
                "coinbase": {
                    "api_key": os.getenv("COINBASE_API_KEY", ""),
                    "secret": os.getenv("COINBASE_SECRET", ""),
                    "sandbox": True
                }
            },
            "trading": {
                "symbols": ["EUR/USDT", "GBP/USDT", "USD/JPY", "AUD/USDT", "USD/CAD", "NZD/USDT"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "strategies": {
                    "scalping": {
                        "enabled": True,
                        "min_profit": 0.0005,
                        "max_loss": 0.001,
                        "position_size": 0.01,
                        "momentum_threshold": 0.7,
                        "volatility_filter": True
                    },
                    "swing_trading": {
                        "enabled": True,
                        "min_profit": 0.002,
                        "max_loss": 0.005,
                        "position_size": 0.05,
                        "trend_confirmation": True
                    },
                    "pattern_trading": {
                        "enabled": True,
                        "patterns": ["head_shoulders", "triangles", "flags", "pennants"],
                        "confidence_threshold": 0.75
                    }
                },
                "risk": {
                    "max_daily_loss": 0.02,
                    "max_position_size": 0.1,
                    "stop_loss": 0.005,
                    "take_profit": 0.01,
                    "max_drawdown": 0.05,
                    "position_sizing": "kelly_criterion"
                }
            },
            "ml": {
                "enabled": True,
                "retrain_interval": 3600,  # 1 hour
                "prediction_threshold": 0.6,
                "features": ["rsi", "macd", "bollinger", "volume", "sentiment", "patterns"],
                "models": {
                    "price_prediction": "ensemble",
                    "volatility_prediction": "lstm",
                    "sentiment_analysis": "vader"
                }
            },
            "news": {
                "enabled": True,
                "api_key": os.getenv("NEWS_API_KEY", ""),
                "impact_threshold": 0.6,
                "update_interval": 300,
                "sources": ["reuters", "bloomberg", "cnbc", "forex_factory"]
            },
            "dashboard": {
                "enabled": True,
                "port": 8080,
                "host": "0.0.0.0",
                "update_interval": 5
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
        
        return default_config
    
    def setup_logging(self):
        """Setup enhanced logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('enhanced_trading_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Enhanced Forex Trading System Initialized")
    
    def setup_exchanges(self):
        """Setup cryptocurrency exchanges"""
        self.exchanges = {}
        
        # Binance
        if self.config["exchanges"]["binance"]["api_key"]:
            self.exchanges["binance"] = ccxt.binance({
                'apiKey': self.config["exchanges"]["binance"]["api_key"],
                'secret': self.config["exchanges"]["binance"]["secret"],
                'sandbox': self.config["exchanges"]["binance"]["sandbox"],
                'enableRateLimit': True,
            })
        
        # Coinbase Pro
        if self.config["exchanges"]["coinbase"]["api_key"]:
            self.exchanges["coinbase"] = ccxt.coinbasepro({
                'apiKey': self.config["exchanges"]["coinbase"]["api_key"],
                'secret': self.config["exchanges"]["coinbase"]["secret"],
                'sandbox': self.config["exchanges"]["coinbase"]["sandbox"],
                'enableRateLimit': True,
            })
        
        self.logger.info(f"✅ Initialized {len(self.exchanges)} exchanges")
    
    def setup_advanced_ml_models(self):
        """Setup advanced machine learning models"""
        self.ml_engine = AdvancedMLTradingEngine(self.config["ml"])
        self.logger.info("✅ Advanced ML models initialized")
    
    def setup_portfolio_optimizer(self):
        """Setup portfolio optimization"""
        self.portfolio_optimizer = PortfolioOptimizer(self.config["trading"]["risk"])
        self.logger.info("✅ Portfolio optimizer initialized")
    
    def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        """Get market data from exchanges"""
        try:
            # Try Binance first
            if "binance" in self.exchanges:
                exchange = self.exchanges["binance"]
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
            
            # Fallback to yfinance for forex data
            if "/" in symbol:
                yf_symbol = symbol.replace("/", "")
                if yf_symbol.endswith("USDT"):
                    yf_symbol = yf_symbol.replace("USDT", "=X")
                else:
                    yf_symbol += "=X"
            else:
                yf_symbol = symbol
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="1y", interval=timeframe)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        if df.empty:
            return df
        
        try:
            # Basic indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['macd_histogram'] = ta.trend.MACD(df['close']).macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Volume indicators
            df['volume_sma'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume']).volume_sma()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Advanced indicators
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Price patterns
            df['doji'] = self.detect_doji(df)
            df['hammer'] = self.detect_hammer(df)
            df['shooting_star'] = self.detect_shooting_star(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect advanced chart patterns"""
        patterns = {}
        
        if df.empty or len(df) < 50:
            return patterns
        
        try:
            # Head and Shoulders
            patterns['head_shoulders'] = self.detect_head_shoulders(df)
            
            # Triangles
            patterns['triangles'] = self.detect_triangles(df)
            
            # Flags and Pennants
            patterns['flags'] = self.detect_flags(df)
            patterns['pennants'] = self.detect_pennants(df)
            
            # Support and Resistance
            patterns['support_resistance'] = self.detect_support_resistance(df)
            
            # Trend lines
            patterns['trend_lines'] = self.detect_trend_lines(df)
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def detect_head_shoulders(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Head and Shoulders pattern"""
        if len(df) < 20:
            return {"detected": False}
        
        highs = df['high'].rolling(window=5, center=True).max()
        peaks, _ = find_peaks(highs, distance=5, prominence=df['high'].std() * 0.5)
        
        if len(peaks) >= 3:
            # Get the last 3 peaks
            last_peaks = peaks[-3:]
            peak_values = [df['high'].iloc[peak] for peak in last_peaks]
            
            # Check if middle peak is highest (head)
            if (peak_values[1] > peak_values[0] and 
                peak_values[1] > peak_values[2] and
                abs(peak_values[0] - peak_values[2]) / peak_values[1] < 0.02):  # Shoulders similar height
                
                return {
                    "detected": True,
                    "confidence": 0.8,
                    "left_shoulder": last_peaks[0],
                    "head": last_peaks[1],
                    "right_shoulder": last_peaks[2],
                    "neckline": (peak_values[0] + peak_values[2]) / 2
                }
        
        return {"detected": False}
    
    def detect_triangles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Triangle patterns"""
        if len(df) < 20:
            return {"detected": False}
        
        # Get recent highs and lows
        recent_data = df.tail(20)
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Calculate trend lines
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Ascending triangle: flat resistance, rising support
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            return {
                "detected": True,
                "type": "ascending",
                "confidence": 0.7,
                "resistance": highs.mean(),
                "support_slope": low_slope
            }
        
        # Descending triangle: flat support, falling resistance
        elif abs(low_slope) < 0.001 and high_slope < -0.001:
            return {
                "detected": True,
                "type": "descending",
                "confidence": 0.7,
                "support": lows.mean(),
                "resistance_slope": high_slope
            }
        
        # Symmetrical triangle: converging lines
        elif (high_slope < -0.001 and low_slope > 0.001 and 
              abs(high_slope) > 0.001 and abs(low_slope) > 0.001):
            return {
                "detected": True,
                "type": "symmetrical",
                "confidence": 0.6,
                "convergence": abs(high_slope) + abs(low_slope)
            }
        
        return {"detected": False}
    
    def detect_flags(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Flag patterns"""
        if len(df) < 15:
            return {"detected": False}
        
        recent_data = df.tail(15)
        
        # Look for strong move followed by consolidation
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        if abs(price_change) > 0.02:  # 2% move
            # Check for consolidation (low volatility)
            volatility = recent_data['close'].std() / recent_data['close'].mean()
            
            if volatility < 0.01:  # Low volatility consolidation
                return {
                    "detected": True,
                    "type": "bull_flag" if price_change > 0 else "bear_flag",
                    "confidence": 0.7,
                    "strength": abs(price_change),
                    "consolidation_volatility": volatility
                }
        
        return {"detected": False}
    
    def detect_pennants(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Pennant patterns"""
        if len(df) < 20:
            return {"detected": False}
        
        recent_data = df.tail(20)
        
        # Look for converging trend lines
        highs = recent_data['high']
        lows = recent_data['low']
        
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Pennant: both lines converging
        if (high_slope < -0.001 and low_slope > 0.001 and 
            abs(high_slope) > 0.001 and abs(low_slope) > 0.001):
            
            convergence = abs(high_slope) + abs(low_slope)
            return {
                "detected": True,
                "confidence": 0.6,
                "convergence": convergence,
                "high_slope": high_slope,
                "low_slope": low_slope
            }
        
        return {"detected": False}
    
    def detect_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Support and Resistance levels"""
        if len(df) < 50:
            return {"levels": [], "strength": []}
        
        recent_data = df.tail(50)
        prices = recent_data['close']
        
        # Find local maxima and minima
        highs, _ = find_peaks(prices, distance=5, prominence=prices.std() * 0.3)
        lows, _ = find_peaks(-prices, distance=5, prominence=prices.std() * 0.3)
        
        resistance_levels = [prices.iloc[i] for i in highs]
        support_levels = [prices.iloc[i] for i in lows]
        
        # Cluster similar levels
        resistance_clusters = self.cluster_levels(resistance_levels, tolerance=0.005)
        support_clusters = self.cluster_levels(support_levels, tolerance=0.005)
        
        return {
            "resistance": resistance_clusters,
            "support": support_clusters,
            "current_price": prices.iloc[-1]
        }
    
    def cluster_levels(self, levels: List[float], tolerance: float = 0.005) -> List[Dict]:
        """Cluster similar price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] <= tolerance * current_cluster[-1]:
                current_cluster.append(level)
            else:
                clusters.append({
                    "level": np.mean(current_cluster),
                    "strength": len(current_cluster),
                    "range": [min(current_cluster), max(current_cluster)]
                })
                current_cluster = [level]
        
        if current_cluster:
            clusters.append({
                "level": np.mean(current_cluster),
                "strength": len(current_cluster),
                "range": [min(current_cluster), max(current_cluster)]
            })
        
        return sorted(clusters, key=lambda x: x["strength"], reverse=True)
    
    def detect_trend_lines(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Trend Lines"""
        if len(df) < 20:
            return {"uptrend": False, "downtrend": False}
        
        recent_data = df.tail(20)
        prices = recent_data['close']
        
        # Simple trend detection using linear regression
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        trend_strength = abs(r_value)
        is_uptrend = slope > 0 and trend_strength > 0.7
        is_downtrend = slope < 0 and trend_strength > 0.7
        
        return {
            "uptrend": is_uptrend,
            "downtrend": is_downtrend,
            "slope": slope,
            "strength": trend_strength,
            "r_squared": r_value ** 2
        }
    
    def detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """Detect Doji candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        
        # Doji: body is less than 10% of total range
        doji = (body_size / total_range) < 0.1
        return doji
    
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect Hammer candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Hammer: small body, long lower shadow, short upper shadow
        hammer = (body_size < (df['high'] - df['low']) * 0.3) & \
                 (lower_shadow > body_size * 2) & \
                 (upper_shadow < body_size)
        
        return hammer
    
    def detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect Shooting Star candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Shooting Star: small body, long upper shadow, short lower shadow
        shooting_star = (body_size < (df['high'] - df['low']) * 0.3) & \
                       (upper_shadow > body_size * 2) & \
                       (lower_shadow < body_size)
        
        return shooting_star
    
    def analyze_market_regime(self, df: pd.DataFrame) -> str:
        """Analyze current market regime"""
        if df.empty or len(df) < 50:
            return "unknown"
        
        recent_data = df.tail(50)
        
        # Calculate volatility
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate trend strength
        x = np.arange(len(recent_data))
        slope, _, r_value, _, _ = stats.linregress(x, recent_data['close'])
        trend_strength = abs(r_value)
        
        # Calculate mean reversion
        price_ma = recent_data['close'].rolling(20).mean()
        price_std = recent_data['close'].rolling(20).std()
        z_score = (recent_data['close'] - price_ma) / price_std
        mean_reversion = abs(z_score).mean()
        
        # Determine regime
        if volatility > 0.2:  # High volatility
            return "volatile"
        elif trend_strength > 0.7:  # Strong trend
            return "trending"
        elif mean_reversion < 1.0:  # Mean reverting
            return "ranging"
        else:
            return "mixed"
    
    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment for a symbol"""
        try:
            if not self.config["news"]["api_key"]:
                return {"sentiment": 0, "confidence": 0}
            
            # Use cached sentiment if available
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # Fetch news
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} forex currency",
                'apiKey': self.config["news"]["api_key"],
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'language': 'en'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                
                # Analyze sentiment
                analyzer = SentimentIntensityAnalyzer()
                sentiments = []
                
                for article in articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title} {description}"
                    
                    sentiment = analyzer.polarity_scores(text)
                    sentiments.append(sentiment['compound'])
                
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    confidence = min(len(sentiments) / 10, 1.0)  # More articles = higher confidence
                    
                    result = {
                        "sentiment": avg_sentiment,
                        "confidence": confidence,
                        "article_count": len(sentiments)
                    }
                    
                    # Cache result
                    self.sentiment_cache[cache_key] = result
                    return result
            
            return {"sentiment": 0, "confidence": 0}
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {e}")
            return {"sentiment": 0, "confidence": 0}
    
    def generate_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive trading signals"""
        try:
            # Get market data
            df = self.get_market_data(symbol, "1h", 200)
            if df.empty:
                return {"signal": "HOLD", "confidence": 0, "reason": "No data"}
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Detect patterns
            patterns = self.detect_patterns(df)
            
            # Analyze market regime
            regime = self.analyze_market_regime(df)
            
            # Get news sentiment
            sentiment = self.get_news_sentiment(symbol)
            
            # Generate signals based on multiple factors
            signals = []
            confidences = []
            
            # Technical analysis signals
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # RSI signals
            if rsi < 30:
                signals.append("BUY")
                confidences.append(0.7)
            elif rsi > 70:
                signals.append("SELL")
                confidences.append(0.7)
            
            # MACD signals
            if macd > macd_signal and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
                signals.append("BUY")
                confidences.append(0.6)
            elif macd < macd_signal and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
                signals.append("SELL")
                confidences.append(0.6)
            
            # Pattern signals
            if patterns.get('head_shoulders', {}).get('detected'):
                signals.append("SELL")
                confidences.append(0.8)
            
            if patterns.get('triangles', {}).get('detected'):
                triangle_type = patterns['triangles'].get('type')
                if triangle_type == 'ascending':
                    signals.append("BUY")
                    confidences.append(0.7)
                elif triangle_type == 'descending':
                    signals.append("SELL")
                    confidences.append(0.7)
            
            # Sentiment signals
            if sentiment['confidence'] > 0.5:
                if sentiment['sentiment'] > 0.3:
                    signals.append("BUY")
                    confidences.append(sentiment['confidence'] * 0.5)
                elif sentiment['sentiment'] < -0.3:
                    signals.append("SELL")
                    confidences.append(sentiment['confidence'] * 0.5)
            
            # Combine signals
            if not signals:
                final_signal = "HOLD"
                final_confidence = 0
            else:
                # Weight signals by confidence
                buy_weight = sum(conf for sig, conf in zip(signals, confidences) if sig == "BUY")
                sell_weight = sum(conf for sig, conf in zip(signals, confidences) if sig == "SELL")
                
                if buy_weight > sell_weight and buy_weight > 0.5:
                    final_signal = "BUY"
                    final_confidence = buy_weight
                elif sell_weight > buy_weight and sell_weight > 0.5:
                    final_signal = "SELL"
                    final_confidence = sell_weight
                else:
                    final_signal = "HOLD"
                    final_confidence = 0
            
            return {
                "signal": final_signal,
                "confidence": final_confidence,
                "regime": regime,
                "patterns": patterns,
                "sentiment": sentiment,
                "technical": {
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_signal
                },
                "price": current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return {"signal": "HOLD", "confidence": 0, "reason": str(e)}
    
    def calculate_position_size(self, symbol: str, signal: str, confidence: float) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            if signal == "HOLD" or confidence < 0.5:
                return 0.0
            
            # Get recent performance for this symbol
            recent_trades = [t for t in self.trade_history if t.get('symbol') == symbol][-20:]
            
            if not recent_trades:
                # Default position size
                return self.config["trading"]["risk"]["max_position_size"] * confidence
            
            # Calculate win rate and average win/loss
            wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
            losses = [t for t in recent_trades if t.get('pnl', 0) < 0]
            
            if not wins and not losses:
                return self.config["trading"]["risk"]["max_position_size"] * confidence
            
            win_rate = len(wins) / len(recent_trades) if recent_trades else 0.5
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.01
            avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0.01
            
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds (avg_win/avg_loss), p = win_rate, q = 1 - win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                p = win_rate
                q = 1 - win_rate
                kelly_fraction = (b * p - q) / b
                
                # Apply confidence and risk limits
                position_size = max(0, min(kelly_fraction * confidence, 
                                        self.config["trading"]["risk"]["max_position_size"]))
            else:
                position_size = self.config["trading"]["risk"]["max_position_size"] * confidence
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def execute_trade(self, symbol: str, signal: str, position_size: float) -> Dict[str, Any]:
        """Execute a trade"""
        try:
            if position_size <= 0:
                return {"success": False, "reason": "Invalid position size"}
            
            # Get current price
            df = self.get_market_data(symbol, "1m", 1)
            if df.empty:
                return {"success": False, "reason": "No price data"}
            
            current_price = df['close'].iloc[-1]
            
            # Create trade record
            trade = {
                "symbol": symbol,
                "side": signal,
                "size": position_size,
                "price": current_price,
                "timestamp": datetime.now(),
                "status": "pending"
            }
            
            # In a real implementation, you would execute the trade here
            # For now, we'll simulate it
            trade["status"] = "executed"
            trade["trade_id"] = f"{symbol}_{int(time.time())}"
            
            # Add to active trades
            self.active_trades[trade["trade_id"]] = trade
            
            # Add to trade history
            self.trade_history.append(trade)
            
            self.logger.info(f"✅ Executed {signal} trade for {symbol}: {position_size} @ {current_price}")
            
            return {
                "success": True,
                "trade_id": trade["trade_id"],
                "symbol": symbol,
                "side": signal,
                "size": position_size,
                "price": current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {"success": False, "reason": str(e)}
    
    def run_trading_loop(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting enhanced trading system...")
        self.running = True
        
        while self.running:
            try:
                for symbol in self.config["trading"]["symbols"]:
                    # Generate signals
                    signals = self.generate_trading_signals(symbol)
                    
                    if signals["signal"] != "HOLD" and signals["confidence"] > 0.6:
                        # Calculate position size
                        position_size = self.calculate_position_size(
                            symbol, signals["signal"], signals["confidence"]
                        )
                        
                        if position_size > 0:
                            # Execute trade
                            trade_result = self.execute_trade(symbol, signals["signal"], position_size)
                            
                            if trade_result["success"]:
                                self.logger.info(f"📈 Trade executed: {trade_result}")
                            else:
                                self.logger.warning(f"❌ Trade failed: {trade_result}")
                
                # Update performance metrics
                self.update_performance_metrics()
                
                # Sleep before next iteration
                time.sleep(60)  # 1 minute
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Trading loop stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(10)
        
        self.running = False
        self.logger.info("🏁 Trading system stopped")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
            
            # Calculate Sharpe ratio (simplified)
            returns = [t.get('pnl', 0) for t in self.trade_history if 'pnl' in t]
            if returns:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            self.performance_metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "sharpe_ratio": sharpe_ratio,
                "active_trades": len(self.active_trades),
                "last_update": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def start(self):
        """Start the trading system"""
        self.logger.info("🚀 Starting Enhanced Forex Trading System...")
        
        # Start trading loop in separate thread
        self.trading_thread = threading.Thread(target=self.run_trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        # Start news analysis thread
        if self.config["news"]["enabled"]:
            self.news_thread = threading.Thread(target=self.run_news_analysis)
            self.news_thread.daemon = True
            self.news_thread.start()
        
        # Start pattern analysis thread
        self.analysis_thread = threading.Thread(target=self.run_pattern_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        self.logger.info("✅ All systems started successfully!")
    
    def stop(self):
        """Stop the trading system"""
        self.logger.info("🛑 Stopping trading system...")
        self.running = False
        
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        if self.news_thread:
            self.news_thread.join(timeout=5)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        
        self.logger.info("🏁 Trading system stopped")
    
    def run_news_analysis(self):
        """Run news analysis in background"""
        while self.running:
            try:
                for symbol in self.config["trading"]["symbols"]:
                    sentiment = self.get_news_sentiment(symbol)
                    self.sentiment_scores[symbol] = sentiment
                
                time.sleep(self.config["news"]["update_interval"])
            except Exception as e:
                self.logger.error(f"Error in news analysis: {e}")
                time.sleep(60)
    
    def run_pattern_analysis(self):
        """Run pattern analysis in background"""
        while self.running:
            try:
                for symbol in self.config["trading"]["symbols"]:
                    df = self.get_market_data(symbol, "1h", 100)
                    if not df.empty:
                        patterns = self.detect_patterns(df)
                        self.pattern_signals[symbol] = patterns
                
                time.sleep(300)  # 5 minutes
            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {e}")
                time.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self.running,
            "active_trades": len(self.active_trades),
            "performance": self.performance_metrics,
            "market_regime": self.market_regime,
            "patterns": self.pattern_signals,
            "sentiment": self.sentiment_scores,
            "last_update": datetime.now()
        }


class AdvancedMLTradingEngine:
    """Advanced Machine Learning Trading Engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def train_price_prediction_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train price prediction model"""
        try:
            # Prepare features
            features = self.prepare_features(data)
            target = data['close'].shift(-1).dropna()
            
            # Align features and target
            features = features.iloc[:-1]
            
            if len(features) < 100:
                return {"success": False, "reason": "Insufficient data"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            
            ensemble = VotingRegressor([
                ('rf', rf),
                ('gb', gb),
                ('xgb', xgb_model)
            ])
            
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = ensemble.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store models
            self.models['price_prediction'] = ensemble
            self.scalers['price_prediction'] = scaler
            
            return {
                "success": True,
                "mse": mse,
                "r2": r2,
                "feature_importance": self.get_feature_importance(ensemble, features.columns)
            }
            
        except Exception as e:
            return {"success": False, "reason": str(e)}
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features = pd.DataFrame()
        
        # Price features
        features['close'] = data['close']
        features['high'] = data['high']
        features['low'] = data['low']
        features['volume'] = data['volume']
        
        # Technical indicators
        if 'rsi' in data.columns:
            features['rsi'] = data['rsi']
        if 'macd' in data.columns:
            features['macd'] = data['macd']
        if 'bb_upper' in data.columns:
            features['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Price changes
        features['price_change_1h'] = data['close'].pct_change(1)
        features['price_change_4h'] = data['close'].pct_change(4)
        features['price_change_24h'] = data['close'].pct_change(24)
        
        # Volatility
        features['volatility'] = data['close'].rolling(24).std()
        
        return features.dropna()
    
    def get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'estimators_'):
                # For ensemble models
                importances = []
                for estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    return dict(zip(feature_names, avg_importance))
            
            return {}
        except Exception as e:
            return {}


class PortfolioOptimizer:
    """Portfolio Optimization Engine"""
    
    def __init__(self, risk_config: Dict):
        self.risk_config = risk_config
        self.correlation_matrix = None
        self.optimal_weights = None
    
    def optimize_portfolio(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using modern portfolio theory"""
        try:
            if returns.empty or len(returns.columns) < 2:
                return {"success": False, "reason": "Insufficient data"}
            
            # Calculate expected returns and covariance
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Calculate correlation matrix
            self.correlation_matrix = returns.corr()
            
            # Portfolio optimization using mean-variance optimization
            n_assets = len(expected_returns)
            
            # Objective function: minimize portfolio variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
            
            # Bounds: each weight between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                self.optimal_weights = dict(zip(returns.columns, result.x))
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(result.x, expected_returns)
                portfolio_volatility = np.sqrt(portfolio_variance(result.x))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return {
                    "success": True,
                    "weights": self.optimal_weights,
                    "expected_return": portfolio_return,
                    "volatility": portfolio_volatility,
                    "sharpe_ratio": sharpe_ratio
                }
            else:
                return {"success": False, "reason": "Optimization failed"}
                
        except Exception as e:
            return {"success": False, "reason": str(e)}


def main():
    """Main function to run the enhanced trading system"""
    print("🚀 Enhanced Forex Trading System")
    print("=" * 50)
    
    # Initialize system
    trading_system = EnhancedForexTradingSystem()
    
    try:
        # Start the system
        trading_system.start()
        
        # Keep running until interrupted
        while trading_system.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        trading_system.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        trading_system.stop()


if __name__ == "__main__":
    main()