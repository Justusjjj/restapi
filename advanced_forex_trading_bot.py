#!/usr/bin/env python3
"""
Advanced Forex Trading Bot with Scalping, Grid Trading, Hedging, News Analysis, and Self-Learning
Features:
- MetaTrader5 integration
- Scalping strategies
- Grid trading with dynamic levels
- Hedging mechanisms
- News sentiment analysis
- Self-learning ML models
- Risk management
- Performance analytics
"""

import os
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import joblib

# Technical Analysis
import ta
import pandas_ta as pta

# News and Sentiment
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configuration
load_dotenv()

class AdvancedForexBot:
    def __init__(self, config_path: str = "bot_config.json"):
        """Initialize the advanced forex trading bot"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_mt5()
        self.setup_ml_models()
        
        # Trading state
        self.active_trades = {}
        self.grid_levels = {}
        self.hedge_positions = {}
        self.performance_metrics = {}
        
        # News cache
        self.news_cache = {}
        self.sentiment_cache = {}
        
        # ML model cache
        self.model_cache = {}
        self.scaler_cache = {}
        
        # Threading
        self.running = False
        self.trading_thread = None
        self.news_thread = None
        self.analysis_thread = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load bot configuration"""
        default_config = {
            "mt5": {
                "login": int(os.getenv("MT5_LOGIN", "0")),
                "password": os.getenv("MT5_PASSWORD", ""),
                "server": os.getenv("MT5_SERVER", ""),
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                "timeframe": "M5"
            },
            "trading": {
                "scalping": {
                    "enabled": True,
                    "min_profit": 0.0005,
                    "max_loss": 0.001,
                    "position_size": 0.01
                },
                "grid": {
                    "enabled": True,
                    "levels": 5,
                    "spacing": 0.001,
                    "position_size": 0.01
                },
                "hedging": {
                    "enabled": True,
                    "correlation_threshold": 0.7,
                    "max_hedge_ratio": 0.5
                },
                "risk": {
                    "max_daily_loss": 0.02,
                    "max_position_size": 0.1,
                    "stop_loss": 0.005,
                    "take_profit": 0.01
                }
            },
            "news": {
                "enabled": True,
                "api_key": os.getenv("NEWS_API_KEY", ""),
                "impact_threshold": 0.6,
                "update_interval": 300
            },
            "ml": {
                "enabled": True,
                "retrain_interval": 86400,
                "prediction_threshold": 0.6,
                "features": ["rsi", "macd", "bollinger", "volume", "sentiment"]
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge user config with defaults
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
        
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_mt5(self):
        """Initialize MetaTrader5 connection"""
        if not mt5.initialize():
            self.logger.error("Failed to initialize MT5")
            return False
        
        # Login to MT5
        if not mt5.login(
            login=self.config["mt5"]["login"],
            password=self.config["mt5"]["password"],
            server=self.config["mt5"]["server"]
        ):
            self.logger.error("Failed to login to MT5")
            return False
        
        self.logger.info("Successfully connected to MetaTrader5")
        return True
    
    def setup_ml_models(self):
        """Initialize machine learning models"""
        self.models = {
            'price_prediction': RandomForestRegressor(n_estimators=100, random_state=42),
            'volatility_prediction': GradientBoostingRegressor(random_state=42),
            'sentiment_analysis': None  # Will be loaded from cache or trained
        }
        
        self.scalers = {
            'price_features': StandardScaler(),
            'volatility_features': StandardScaler()
        }
        
        # Load pre-trained models if available
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models from disk"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            return
        
        for model_name in self.models.keys():
            model_path = os.path.join(model_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load model {model_name}: {e}")
    
    def save_models(self):
        """Save trained models to disk"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    model_path = os.path.join(model_dir, f"{model_name}.joblib")
                    joblib.dump(model, model_path)
                    self.logger.info(f"Saved model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to save model {model_name}: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = "M5", bars: int = 1000) -> pd.DataFrame:
        """Get market data from MT5"""
        try:
            # Convert timeframe string to MT5 constant
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            if rates is None:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = bb.bollinger_wband()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Volume indicators
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            
            # Price patterns
            df['price_change'] = df['close'].pct_change()
            df['price_volatility'] = df['price_change'].rolling(20).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment for a currency pair"""
        try:
            # Extract base and quote currencies
            base = symbol[:3]
            quote = symbol[3:]
            
            # Check cache first
            cache_key = f"{base}_{quote}_{datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # Get news from API
            api_key = self.config["news"]["api_key"]
            if not api_key:
                return 0.0
            
            # Search for relevant news
            query = f"({base} OR {quote}) AND (forex OR currency OR economy)"
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&sortBy=publishedAt&pageSize=10"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return 0.0
            
            news_data = response.json()
            articles = news_data.get('articles', [])
            
            if not articles:
                return 0.0
            
            # Analyze sentiment
            analyzer = SentimentIntensityAnalyzer()
            sentiments = []
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                # Get sentiment scores
                scores = analyzer.polarity_scores(content)
                compound_score = scores['compound']
                sentiments.append(compound_score)
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Cache the result
            self.sentiment_cache[cache_key] = avg_sentiment
            
            return avg_sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {e}")
            return 0.0
    
    def generate_features(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Generate features for ML models"""
        try:
            features = []
            
            # Technical indicators
            tech_features = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'stoch_k', 'stoch_d',
                'atr', 'volume_ratio', 'price_volatility'
            ]
            
            for feature in tech_features:
                if feature in df.columns:
                    features.extend([
                        df[feature].iloc[-1],
                        df[feature].iloc[-1] - df[feature].iloc[-5],
                        df[feature].iloc[-1] - df[feature].iloc[-10]
                    ])
            
            # Price features
            price_features = [
                df['close'].iloc[-1],
                df['close'].pct_change().iloc[-1],
                df['close'].pct_change().rolling(5).mean().iloc[-1],
                df['close'].pct_change().rolling(10).mean().iloc[-1]
            ]
            features.extend(price_features)
            
            # Volume features
            volume_features = [
                df['tick_volume'].iloc[-1],
                df['volume_ratio'].iloc[-1]
            ]
            features.extend(volume_features)
            
            # News sentiment
            sentiment = self.get_news_sentiment(symbol)
            features.append(sentiment)
            
            # Time features
            now = datetime.now()
            features.extend([
                now.hour / 24.0,
                now.weekday() / 7.0,
                now.month / 12.0
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return np.array([])
    
    def predict_price_movement(self, symbol: str) -> Tuple[float, float]:
        """Predict price movement using ML models"""
        try:
            # Get market data
            df = self.get_market_data(symbol)
            if df.empty:
                return 0.0, 0.0
            
            # Generate features
            features = self.generate_features(df, symbol)
            if features.size == 0:
                return 0.0, 0.0
            
            # Scale features
            if 'price_features' in self.scalers:
                features_scaled = self.scalers['price_features'].transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            if self.models['price_prediction'] is not None:
                prediction = self.models['price_prediction'].predict(features_scaled)[0]
            else:
                prediction = 0.0
            
            # Calculate confidence (simplified)
            confidence = min(abs(prediction), 1.0)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement: {e}")
            return 0.0, 0.0
    
    def scalping_strategy(self, symbol: str) -> Dict:
        """Execute scalping strategy"""
        try:
            prediction, confidence = self.predict_price_movement(symbol)
            
            if confidence < self.config["ml"]["prediction_threshold"]:
                return {"action": "hold", "reason": "low_confidence"}
            
            # Get current market data
            df = self.get_market_data(symbol, bars=100)
            if df.empty:
                return {"action": "hold", "reason": "no_data"}
            
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # Scalping signals
            buy_signal = (
                prediction > 0 and
                rsi < 70 and
                macd > macd_signal and
                df['close'].iloc[-1] > df['bb_middle'].iloc[-1]
            )
            
            sell_signal = (
                prediction < 0 and
                rsi > 30 and
                macd < macd_signal and
                df['close'].iloc[-1] < df['bb_middle'].iloc[-1]
            )
            
            if buy_signal:
                return {
                    "action": "buy",
                    "symbol": symbol,
                    "type": "scalping",
                    "price": current_price,
                    "stop_loss": current_price - self.config["trading"]["risk"]["stop_loss"],
                    "take_profit": current_price + self.config["trading"]["scalping"]["min_profit"],
                    "size": self.config["trading"]["scalping"]["position_size"]
                }
            elif sell_signal:
                return {
                    "action": "sell",
                    "symbol": symbol,
                    "type": "scalping",
                    "price": current_price,
                    "stop_loss": current_price + self.config["trading"]["risk"]["stop_loss"],
                    "take_profit": current_price - self.config["trading"]["scalping"]["min_profit"],
                    "size": self.config["trading"]["scalping"]["position_size"]
                }
            
            return {"action": "hold", "reason": "no_signal"}
            
        except Exception as e:
            self.logger.error(f"Error in scalping strategy: {e}")
            return {"action": "hold", "reason": "error"}
    
    def grid_trading_strategy(self, symbol: str) -> Dict:
        """Execute grid trading strategy"""
        try:
            df = self.get_market_data(symbol, bars=100)
            if df.empty:
                return {"action": "hold", "reason": "no_data"}
            
            current_price = df['close'].iloc[-1]
            
            # Check if we need to create new grid levels
            if symbol not in self.grid_levels:
                self.create_grid_levels(symbol, current_price)
            
            # Check grid levels for entry/exit
            grid = self.grid_levels[symbol]
            
            for level in grid:
                if level['type'] == 'buy' and current_price <= level['price']:
                    if not level['filled']:
                        return {
                            "action": "buy",
                            "symbol": symbol,
                            "type": "grid",
                            "price": level['price'],
                            "stop_loss": level['price'] - self.config["trading"]["risk"]["stop_loss"],
                            "take_profit": level['price'] + self.config["trading"]["grid"]["spacing"],
                            "size": self.config["trading"]["grid"]["position_size"]
                        }
                
                elif level['type'] == 'sell' and current_price >= level['price']:
                    if not level['filled']:
                        return {
                            "action": "sell",
                            "symbol": symbol,
                            "type": "grid",
                            "symbol": symbol,
                            "price": level['price'],
                            "stop_loss": level['price'] + self.config["trading"]["risk"]["stop_loss"],
                            "take_profit": level['price'] - self.config["trading"]["grid"]["spacing"],
                            "size": self.config["trading"]["grid"]["position_size"]
                        }
            
            return {"action": "hold", "reason": "grid_waiting"}
            
        except Exception as e:
            self.logger.error(f"Error in grid trading strategy: {e}")
            return {"action": "hold", "reason": "error"}
    
    def create_grid_levels(self, symbol: str, current_price: float):
        """Create grid trading levels"""
        try:
            levels = []
            spacing = self.config["trading"]["grid"]["spacing"]
            num_levels = self.config["trading"]["grid"]["levels"]
            
            # Create buy levels below current price
            for i in range(1, num_levels + 1):
                buy_price = current_price - (i * spacing)
                levels.append({
                    "type": "buy",
                    "price": buy_price,
                    "filled": False,
                    "order_id": None
                })
            
            # Create sell levels above current price
            for i in range(1, num_levels + 1):
                sell_price = current_price + (i * spacing)
                levels.append({
                    "type": "sell",
                    "price": sell_price,
                    "filled": False,
                    "order_id": None
                })
            
            self.grid_levels[symbol] = levels
            self.logger.info(f"Created grid levels for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error creating grid levels: {e}")
    
    def hedging_strategy(self, symbol: str) -> Dict:
        """Execute hedging strategy"""
        try:
            # Check for correlated positions
            correlated_positions = self.find_correlated_positions(symbol)
            
            if not correlated_positions:
                return {"action": "hold", "reason": "no_correlation"}
            
            # Calculate hedge ratio
            total_exposure = sum(pos['size'] for pos in correlated_positions)
            max_hedge = total_exposure * self.config["trading"]["hedging"]["max_hedge_ratio"]
            
            # Check if we need to hedge
            current_hedge = self.get_current_hedge(symbol)
            
            if current_hedge < max_hedge:
                # Calculate hedge position
                hedge_size = max_hedge - current_hedge
                
                return {
                    "action": "hedge",
                    "symbol": symbol,
                    "size": hedge_size,
                    "type": "hedging",
                    "reason": "exposure_management"
                }
            
            return {"action": "hold", "reason": "hedge_sufficient"}
            
        except Exception as e:
            self.logger.error(f"Error in hedging strategy: {e}")
            return {"action": "hold", "reason": "error"}
    
    def find_correlated_positions(self, symbol: str) -> List[Dict]:
        """Find positions correlated with the given symbol"""
        # Simplified correlation check
        # In a real implementation, you would calculate actual correlations
        correlated = []
        
        for trade_id, trade in self.active_trades.items():
            if trade['symbol'] != symbol and trade['type'] in ['scalping', 'grid']:
                correlated.append(trade)
        
        return correlated
    
    def get_current_hedge(self, symbol: str) -> float:
        """Get current hedge position size for a symbol"""
        hedge_size = 0.0
        
        for trade_id, trade in self.active_trades.items():
            if trade['symbol'] == symbol and trade['type'] == 'hedging':
                hedge_size += trade['size']
        
        return hedge_size
    
    def execute_trade(self, trade_signal: Dict) -> bool:
        """Execute a trade based on the signal"""
        try:
            if trade_signal["action"] == "hold":
                return True
            
            symbol = trade_signal["symbol"]
            trade_type = trade_signal["type"]
            size = trade_signal["size"]
            
            # Check risk limits
            if not self.check_risk_limits(symbol, size):
                self.logger.warning(f"Risk limit exceeded for {symbol}")
                return False
            
            # Execute order in MT5
            if trade_signal["action"] == "buy":
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": size,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"{trade_type}_order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if specified
            if "stop_loss" in trade_signal:
                request["sl"] = trade_signal["stop_loss"]
            if "take_profit" in trade_signal:
                request["tp"] = trade_signal["take_profit"]
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode}")
                return False
            
            # Record the trade
            trade_id = result.order
            self.active_trades[trade_id] = {
                "symbol": symbol,
                "type": trade_type,
                "size": size,
                "entry_price": price,
                "entry_time": datetime.now(),
                "order_id": trade_id
            }
            
            self.logger.info(f"Executed {trade_signal['action']} order for {symbol}: {size} lots at {price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def check_risk_limits(self, symbol: str, size: float) -> bool:
        """Check if trade meets risk management criteria"""
        try:
            # Check daily loss limit
            daily_pnl = self.calculate_daily_pnl()
            if daily_pnl < -self.config["trading"]["risk"]["max_daily_loss"]:
                return False
            
            # Check position size limit
            total_exposure = self.calculate_total_exposure(symbol)
            if total_exposure + size > self.config["trading"]["risk"]["max_position_size"]:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def calculate_daily_pnl(self) -> float:
        """Calculate daily profit/loss"""
        try:
            today = datetime.now().date()
            total_pnl = 0.0
            
            # This is a simplified calculation
            # In a real implementation, you would query MT5 for actual P&L
            for trade_id, trade in self.active_trades.items():
                # Simplified P&L calculation
                pass
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def calculate_total_exposure(self, symbol: str) -> float:
        """Calculate total exposure for a symbol"""
        total_exposure = 0.0
        
        for trade_id, trade in self.active_trades.items():
            if trade['symbol'] == symbol:
                total_exposure += trade['size']
        
        return total_exposure
    
    def update_trades(self):
        """Update active trades and close completed ones"""
        try:
            # Get open positions from MT5
            positions = mt5.positions_get()
            if positions is None:
                return
            
            # Update active trades
            for position in positions:
                trade_id = position.ticket
                
                if trade_id in self.active_trades:
                    # Update trade info
                    self.active_trades[trade_id].update({
                        "current_price": position.price_current,
                        "profit": position.profit,
                        "swap": position.swap
                    })
                    
                    # Check if position should be closed
                    if self.should_close_position(position):
                        self.close_position(trade_id)
            
        except Exception as e:
            self.logger.error(f"Error updating trades: {e}")
    
    def should_close_position(self, position) -> bool:
        """Check if a position should be closed"""
        try:
            # Check stop loss or take profit
            if position.profit <= -self.config["trading"]["risk"]["stop_loss"]:
                return True
            
            if position.profit >= self.config["trading"]["risk"]["take_profit"]:
                return True
            
            # Check time-based exit for scalping
            if hasattr(position, 'time') and position.time:
                entry_time = datetime.fromtimestamp(position.time)
                current_time = datetime.now()
                
                if (current_time - entry_time).total_seconds() > 3600:  # 1 hour
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking position close: {e}")
            return False
    
    def close_position(self, trade_id: int):
        """Close a position"""
        try:
            if trade_id not in self.active_trades:
                return
            
            trade = self.active_trades[trade_id]
            
            # Close position in MT5
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": trade['symbol'],
                "volume": trade['size'],
                "type": mt5.ORDER_TYPE_SELL if trade['type'] == 'buy' else mt5.ORDER_TYPE_BUY,
                "position": trade_id,
                "price": mt5.symbol_info_tick(trade['symbol']).bid if trade['type'] == 'buy' else mt5.symbol_info_tick(trade['symbol']).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "close_order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Closed position {trade_id} for {trade['symbol']}")
                del self.active_trades[trade_id]
            else:
                self.logger.error(f"Failed to close position {trade_id}: {result.retcode}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def retrain_models(self):
        """Retrain ML models with new data"""
        try:
            self.logger.info("Starting model retraining...")
            
            # Collect training data
            training_data = self.collect_training_data()
            
            if training_data.empty:
                self.logger.warning("No training data available")
                return
            
            # Prepare features and targets
            X = training_data.drop(['target_price', 'target_volatility'], axis=1)
            y_price = training_data['target_price']
            y_volatility = training_data['target_volatility']
            
            # Split data
            X_train, X_test, y_price_train, y_price_test, y_volatility_train, y_volatility_test = train_test_split(
                X, y_price, y_volatility, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['price_features'].fit_transform(X_train)
            X_test_scaled = self.scalers['price_features'].transform(X_test)
            
            # Retrain price prediction model
            if self.models['price_prediction'] is not None:
                self.models['price_prediction'].fit(X_train_scaled, y_price_train)
                price_score = self.models['price_prediction'].score(X_test_scaled, y_price_test)
                self.logger.info(f"Price prediction model R² score: {price_score:.4f}")
            
            # Retrain volatility prediction model
            if self.models['volatility_prediction'] is not None:
                self.models['volatility_prediction'].fit(X_train_scaled, y_volatility_train)
                volatility_score = self.models['volatility_prediction'].score(X_test_scaled, y_volatility_test)
                self.logger.info(f"Volatility prediction model R² score: {volatility_score:.4f}")
            
            # Save models
            self.save_models()
            
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
    
    def collect_training_data(self) -> pd.DataFrame:
        """Collect training data for ML models"""
        try:
            training_data = []
            
            for symbol in self.config["mt5"]["symbols"]:
                # Get historical data
                df = self.get_market_data(symbol, bars=1000)
                if df.empty:
                    continue
                
                # Generate features
                for i in range(50, len(df) - 10):
                    features = self.generate_features(df.iloc[:i+1], symbol)
                    if features.size == 0:
                        continue
                    
                    # Calculate targets (future price change and volatility)
                    future_prices = df['close'].iloc[i+1:i+11]
                    price_change = (future_prices.iloc[-1] - df['close'].iloc[i]) / df['close'].iloc[i]
                    volatility = future_prices.pct_change().std()
                    
                    # Add to training data
                    row = features.flatten().tolist()
                    row.extend([price_change, volatility])
                    training_data.append(row)
            
            if not training_data:
                return pd.DataFrame()
            
            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(len(training_data[0]) - 2)]
            columns = feature_names + ['target_price', 'target_volatility']
            
            return pd.DataFrame(training_data, columns=columns)
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")
            return pd.DataFrame()
    
    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            self.logger.info("Starting trading cycle...")
            
            # Update existing trades
            self.update_trades()
            
            # Run strategies for each symbol
            for symbol in self.config["mt5"]["symbols"]:
                # Scalping strategy
                if self.config["trading"]["scalping"]["enabled"]:
                    scalping_signal = self.scalping_strategy(symbol)
                    if scalping_signal["action"] != "hold":
                        self.execute_trade(scalping_signal)
                
                # Grid trading strategy
                if self.config["trading"]["grid"]["enabled"]:
                    grid_signal = self.grid_trading_strategy(symbol)
                    if grid_signal["action"] != "hold":
                        self.execute_trade(grid_signal)
                
                # Hedging strategy
                if self.config["trading"]["hedging"]["enabled"]:
                    hedge_signal = self.hedging_strategy(symbol)
                    if hedge_signal["action"] != "hold":
                        self.execute_trade(hedge_signal)
            
            # Update performance metrics
            self.update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            total_pnl = 0.0
            total_trades = len(self.active_trades)
            
            for trade_id, trade in self.active_trades.items():
                if 'profit' in trade:
                    total_pnl += trade['profit']
            
            self.performance_metrics = {
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            self.logger.info("Starting Advanced Forex Trading Bot...")
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            # Start news monitoring thread
            if self.config["news"]["enabled"]:
                self.news_thread = threading.Thread(target=self.news_monitoring_loop)
                self.news_thread.daemon = True
                self.news_thread.start()
            
            # Start analysis thread
            if self.config["ml"]["enabled"]:
                self.analysis_thread = threading.Thread(target=self.analysis_loop)
                self.analysis_thread.daemon = True
                self.analysis_thread.start()
            
            self.logger.info("Bot started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            self.running = False
    
    def stop(self):
        """Stop the trading bot"""
        try:
            self.running = False
            self.logger.info("Stopping Advanced Forex Trading Bot...")
            
            # Wait for threads to finish
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)
            
            if self.news_thread and self.news_thread.is_alive():
                self.news_thread.join(timeout=5)
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5)
            
            # Save models
            self.save_models()
            
            # Close MT5 connection
            mt5.shutdown()
            
            self.logger.info("Bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    def trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                self.run_trading_cycle()
                time.sleep(60)  # Wait 1 minute between cycles
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def news_monitoring_loop(self):
        """News monitoring loop"""
        while self.running:
            try:
                # Update news sentiment for all symbols
                for symbol in self.config["mt5"]["symbols"]:
                    sentiment = self.get_news_sentiment(symbol)
                    self.logger.debug(f"News sentiment for {symbol}: {sentiment}")
                
                time.sleep(self.config["news"]["update_interval"])
            except Exception as e:
                self.logger.error(f"Error in news monitoring loop: {e}")
                time.sleep(300)
    
    def analysis_loop(self):
        """Analysis and model retraining loop"""
        last_retrain = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Retrain models periodically
                if current_time - last_retrain > self.config["ml"]["retrain_interval"]:
                    self.retrain_models()
                    last_retrain = current_time
                
                time.sleep(3600)  # Check every hour
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(3600)
    
    def get_status(self) -> Dict:
        """Get bot status"""
        return {
            "running": self.running,
            "active_trades": len(self.active_trades),
            "performance": self.performance_metrics,
            "grid_levels": {symbol: len(levels) for symbol, levels in self.grid_levels.items()},
            "last_update": datetime.now().isoformat()
        }

def main():
    """Main function to run the bot"""
    try:
        # Create bot instance
        bot = AdvancedForexBot()
        
        # Start the bot
        bot.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            bot.stop()
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()