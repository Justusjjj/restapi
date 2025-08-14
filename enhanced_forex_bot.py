#!/usr/bin/env python3
"""
Enhanced Forex Trading Bot with Advanced ML, ICT Analysis, and Mathematical Algorithms
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import asyncio
import json

# Import advanced modules
from ml_trading_engine import AdvancedMLTradingEngine
from ict_price_action import ICTPriceActionAnalyzer
from mathematical_algorithms import AdvancedMathematicalAlgorithms

# Load environment variables
load_dotenv()

class EnhancedForexTradingBot:
    """
    Enhanced Forex Trading Bot with Advanced Features
    - Machine Learning Trading Engine
    - ICT and Price Action Analysis
    - Advanced Mathematical Algorithms
    - Live Trading with Risk Management
    - Self-Learning Capabilities
    """
    
    def __init__(self, config=None):
        """Initialize the enhanced trading bot"""
        self.config = config or self._default_config()
        
        # Initialize exchange connection
        self.exchange = self._initialize_exchange()
        
        # Initialize advanced analysis engines
        self.ml_engine = AdvancedMLTradingEngine(self.config.get('ml_config'))
        self.ict_analyzer = ICTPriceActionAnalyzer(self.config.get('ict_config'))
        self.math_algorithms = AdvancedMathematicalAlgorithms(self.config.get('math_config'))
        
        # Trading parameters
        self.symbols = self.config.get('symbols', ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'])
        self.timeframe = self.config.get('timeframe', '1h')
        self.max_positions = self.config.get('max_positions', 3)
        self.stop_loss_pips = self.config.get('stop_loss_pips', 50)
        self.take_profit_pips = self.config.get('take_profit_pips', 100)
        
        # Risk management
        self.max_daily_loss = self.config.get('max_daily_loss', 0.02)
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Performance tracking
        self.trades = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.performance_metrics = {}
        
        # Live trading mode
        self.live_trading_mode = self.config.get('live_trading_mode', False)
        self.paper_trading = not self.live_trading_mode
        
        # Setup logging
        self._setup_logging()
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        
    def _default_config(self):
        """Default configuration for the enhanced bot"""
        return {
            'exchange_name': 'oanda',
            'api_key': os.getenv('EXCHANGE_API_KEY'),
            'secret': os.getenv('EXCHANGE_SECRET'),
            'live_trading_mode': False,
            'paper_trading': True,
            
            # Trading parameters
            'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
            'timeframe': '1h',
            'max_positions': 3,
            'stop_loss_pips': 50,
            'take_profit_pips': 100,
            
            # Risk management
            'max_daily_loss': 0.02,
            'max_portfolio_risk': 0.05,
            'risk_per_trade': 0.02,
            
            # ML configuration
            'ml_config': {
                'feature_window': 100,
                'prediction_horizon': 5,
                'retrain_frequency': 1000,
                'confidence_threshold': 0.75
            },
            
            # ICT configuration
            'ict_config': {
                'fair_value_gaps': True,
                'liquidity_levels': True,
                'order_blocks': True,
                'candlestick_patterns': True
            },
            
            # Mathematical algorithms configuration
            'math_config': {
                'volatility_model': 'garch',
                'trend_detection': 'kalman',
                'regime_detection': True
            }
        }
    
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_name = self.config.get('exchange_name', 'oanda')
            api_key = self.config.get('api_key')
            secret = self.config.get('secret')
            
            if exchange_name == 'oanda':
                exchange_class = getattr(ccxt, 'oanda')
                exchange = exchange_class({
                    'apiKey': api_key,
                    'secret': secret,
                    'sandbox': not self.live_trading_mode,
                    'enableRateLimit': True,
                })
            else:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True,
                })
            
            # Test connection
            exchange.load_markets()
            self.logger.info(f"Successfully connected to {exchange_name}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            if self.live_trading_mode:
                raise
            else:
                self.logger.warning("Running in demo mode without exchange connection")
                return None
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_forex_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_ml_models(self):
        """Initialize and train ML models"""
        try:
            self.logger.info("Initializing ML models...")
            
            # Get historical data for training
            training_data = self._get_training_data()
            
            if training_data is not None and len(training_data) > 0:
                # Extract features and train models
                X, y, feature_names = self.ml_engine.prepare_features(training_data)
                
                if X is not None and len(X) > 0:
                    self.ml_engine.train_models(X, y, feature_names)
                    self.logger.info("ML models trained successfully")
                else:
                    self.logger.warning("Insufficient data for ML model training")
            else:
                self.logger.warning("No training data available")
                
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")
    
    def _get_training_data(self, days=90):
        """Get historical data for ML model training"""
        try:
            if self.exchange is None:
                # Generate sample data for demo mode
                return self._generate_sample_training_data(days)
            
            # Get data from exchange
            all_data = {}
            for symbol in self.symbols[:1]:  # Use first symbol for training
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=days*24)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    all_data[symbol] = df
                except Exception as e:
                    self.logger.warning(f"Could not fetch data for {symbol}: {e}")
            
            if all_data:
                # Combine data from all symbols
                combined_data = pd.concat(all_data.values(), axis=1, keys=all_data.keys())
                return combined_data
            else:
                return self._generate_sample_training_data(days)
                
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return self._generate_sample_training_data(days)
    
    def _generate_sample_training_data(self, days):
        """Generate sample training data for demo mode"""
        try:
            dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                                end=datetime.now(), freq='1H')
            
            np.random.seed(42)
            base_price = 1.1000
            
            data = []
            for i, date in enumerate(dates):
                # Generate realistic price movements
                price_change = np.random.normal(0, 0.0005)
                base_price *= (1 + price_change)
                
                # Generate OHLCV
                volatility = abs(np.random.normal(0, 0.0003))
                open_price = base_price * (1 + np.random.normal(0, 0.0001))
                high_price = max(open_price, base_price) + volatility
                low_price = min(open_price, base_price) - volatility
                close_price = base_price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating sample data: {e}")
            return pd.DataFrame()
    
    def get_comprehensive_analysis(self, symbol):
        """
        Get comprehensive analysis using all engines
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with comprehensive analysis
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if market_data is None or len(market_data) == 0:
                return {}
            
            analysis_results = {}
            
            # 1. ICT and Price Action Analysis
            self.logger.info(f"Running ICT analysis for {symbol}")
            ict_analysis = self.ict_analyzer.get_comprehensive_analysis(market_data)
            analysis_results['ict_analysis'] = ict_analysis
            
            # 2. Mathematical Algorithm Analysis
            self.logger.info(f"Running mathematical analysis for {symbol}")
            returns = market_data['close'].pct_change().dropna()
            
            # Volatility analysis
            volatility_analysis = self.math_algorithms.calculate_advanced_volatility(returns)
            analysis_results['volatility'] = volatility_analysis
            
            # Trend analysis using Kalman filter
            kalman_trend = self.math_algorithms.detect_trend_using_kalman(market_data['close'])
            analysis_results['kalman_trend'] = kalman_trend
            
            # Market regime detection
            regime_analysis = self.math_algorithms.detect_market_regime(market_data['close'], returns)
            analysis_results['regime'] = regime_analysis
            
            # Hurst exponent for mean reversion
            hurst_analysis = self.math_algorithms.calculate_hurst_exponent(market_data['close'])
            analysis_results['hurst'] = hurst_analysis
            
            # Fractal analysis
            fractal_analysis = self.math_algorithms.calculate_fractal_dimension(market_data['close'])
            analysis_results['fractal'] = fractal_analysis
            
            # Entropy analysis
            entropy_analysis = self.math_algorithms.calculate_entropy_measures(returns)
            analysis_results['entropy'] = entropy_analysis
            
            # 3. ML Engine Analysis
            self.logger.info(f"Running ML analysis for {symbol}")
            ml_signal, ml_confidence = self.ml_engine.get_trading_signal(market_data)
            analysis_results['ml_signal'] = {
                'signal': ml_signal,
                'confidence': ml_confidence
            }
            
            # 4. Generate Mathematical Trading Signals
            math_signals = self.math_algorithms.generate_trading_signals(analysis_results)
            analysis_results['mathematical_signals'] = math_signals
            
            # 5. Combine all signals
            combined_signal = self._combine_all_signals(analysis_results)
            analysis_results['combined_signal'] = combined_signal
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {}
    
    def _get_market_data(self, symbol, limit=200):
        """Get market data for analysis"""
        try:
            if self.exchange is None:
                # Generate sample data for demo mode
                return self._generate_sample_market_data(symbol, limit)
            
            # Fetch real market data
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._generate_sample_market_data(symbol, limit)
    
    def _generate_sample_market_data(self, symbol, limit):
        """Generate sample market data for demo mode"""
        try:
            dates = pd.date_range(start=datetime.now() - timedelta(hours=limit), 
                                end=datetime.now(), freq='1H')
            
            np.random.seed(hash(symbol) % 1000)  # Different seed for each symbol
            base_price = 1.1000 if 'USD' in symbol else 110.0
            
            data = []
            for i, date in enumerate(dates):
                # Generate realistic price movements
                price_change = np.random.normal(0, 0.0005)
                base_price *= (1 + price_change)
                
                # Generate OHLCV
                volatility = abs(np.random.normal(0, 0.0003))
                open_price = base_price * (1 + np.random.normal(0, 0.0001))
                high_price = max(open_price, base_price) + volatility
                low_price = min(open_price, base_price) - volatility
                close_price = base_price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating sample market data: {e}")
            return pd.DataFrame()
    
    def _combine_all_signals(self, analysis_results):
        """
        Combine signals from all analysis engines
        
        Args:
            analysis_results: Results from all analysis engines
            
        Returns:
            Combined trading signal
        """
        try:
            # Initialize signal counters
            bullish_signals = 0
            bearish_signals = 0
            neutral_signals = 0
            
            total_confidence = 0
            signal_count = 0
            
            # 1. ML Engine Signal
            if 'ml_signal' in analysis_results:
                ml_signal = analysis_results['ml_signal']['signal']
                ml_confidence = analysis_results['ml_signal']['confidence']
                
                if ml_signal == 'BUY':
                    bullish_signals += 1
                    total_confidence += ml_confidence
                    signal_count += 1
                elif ml_signal == 'SELL':
                    bearish_signals += 1
                    total_confidence += ml_confidence
                    signal_count += 1
                else:
                    neutral_signals += 1
            
            # 2. Mathematical Signals
            if 'mathematical_signals' in analysis_results:
                math_signals = analysis_results['mathematical_signals']
                
                if math_signals['trend_signal'] == 'BULLISH':
                    bullish_signals += 1
                    total_confidence += math_signals['confidence']
                    signal_count += 1
                elif math_signals['trend_signal'] == 'BEARISH':
                    bearish_signals += 1
                    total_confidence += math_signals['confidence']
                    signal_count += 1
                else:
                    neutral_signals += 1
            
            # 3. ICT Analysis
            if 'ict_analysis' in analysis_results:
                ict_analysis = analysis_results['ict_analysis']
                
                # Count bullish vs bearish patterns
                pattern_dist = ict_analysis.get('candlestick_patterns', {}).get('pattern_distribution', {})
                
                bullish_patterns = sum(1 for pattern, count in pattern_dist.items() 
                                     if 'bullish' in pattern.lower() or pattern in ['hammer', 'morning_star', 'three_white_soldiers'])
                bearish_patterns = sum(1 for pattern, count in pattern_dist.items() 
                                     if 'bearish' in pattern.lower() or pattern in ['shooting_star', 'evening_star', 'three_black_crows'])
                
                if bullish_patterns > bearish_patterns:
                    bullish_signals += 1
                    total_confidence += 0.7
                    signal_count += 1
                elif bearish_patterns > bullish_patterns:
                    bearish_signals += 1
                    total_confidence += 0.7
                    signal_count += 1
                else:
                    neutral_signals += 1
            
            # 4. Market Regime
            if 'regime' in analysis_results:
                regime_analysis = analysis_results['regime']
                if 'regimes' in regime_analysis:
                    current_regime = regime_analysis['regimes'].iloc[-1] if len(regime_analysis['regimes']) > 0 else 'unknown'
                    
                    if 'low_volatility' in str(current_regime):
                        neutral_signals += 1
                    elif 'high_volatility' in str(current_regime):
                        # High volatility can be bullish or bearish depending on context
                        if bullish_signals > bearish_signals:
                            bullish_signals += 0.5
                        elif bearish_signals > bullish_signals:
                            bearish_signals += 0.5
                        else:
                            neutral_signals += 1
            
            # Determine final signal
            if signal_count > 0:
                avg_confidence = total_confidence / signal_count
            else:
                avg_confidence = 0.5
            
            # Signal strength threshold
            signal_threshold = 0.6
            
            if bullish_signals > bearish_signals and avg_confidence > signal_threshold:
                final_signal = 'BUY'
                signal_strength = min(0.9, (bullish_signals - bearish_signals) / max(bullish_signals + bearish_signals, 1))
            elif bearish_signals > bullish_signals and avg_confidence > signal_threshold:
                final_signal = 'SELL'
                signal_strength = min(0.9, (bearish_signals - bullish_signals) / max(bullish_signals + bearish_signals, 1))
            else:
                final_signal = 'HOLD'
                signal_strength = 0.5
            
            return {
                'signal': final_signal,
                'confidence': avg_confidence,
                'signal_strength': signal_strength,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'neutral_signals': neutral_signals,
                'total_signals': signal_count
            }
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'signal_strength': 0.5,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'neutral_signals': 1,
                'total_signals': 1
            }
    
    def calculate_position_size(self, signal, confidence, account_balance, market_volatility):
        """
        Calculate position size using ML insights and risk management
        
        Args:
            signal: Trading signal
            confidence: Signal confidence
            account_balance: Current account balance
            market_volatility: Current market volatility
            
        Returns:
            Position size and risk metrics
        """
        try:
            if signal == 'HOLD':
                return 0.0, 0.0, 0.0
            
            # Use ML engine for position sizing
            position_size, actual_risk, risk_percentage = self.ml_engine.calculate_risk_adjusted_position_size(
                signal, confidence, account_balance, market_volatility, self.risk_per_trade
            )
            
            # Apply additional risk checks
            if risk_percentage > self.max_portfolio_risk:
                # Reduce position size to meet risk limits
                reduction_factor = self.max_portfolio_risk / risk_percentage
                position_size *= reduction_factor
                actual_risk *= reduction_factor
                risk_percentage = self.max_portfolio_risk
            
            return position_size, actual_risk, risk_percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Fallback to simple position sizing
            risk_amount = account_balance * self.risk_per_trade
            pip_value = 1.0
            position_size = risk_amount / (self.stop_loss_pips * pip_value)
            return max(0.01, min(position_size, 1.0)), risk_amount, self.risk_per_trade
    
    def execute_trade(self, symbol, signal, confidence, position_size, stop_loss, take_profit):
        """
        Execute a trade based on analysis results
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            confidence: Signal confidence
            position_size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Trade execution result
        """
        try:
            if self.paper_trading:
                # Paper trading - simulate trade execution
                trade_result = {
                    'symbol': symbol,
                    'side': signal,
                    'amount': position_size,
                    'entry_price': self._get_current_price(symbol),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'status': 'OPEN',
                    'type': 'PAPER'
                }
                
                self.trades.append(trade_result)
                self.logger.info(f"Paper trade executed: {signal} {position_size} {symbol}")
                
                return trade_result
            else:
                # Live trading
                if self.exchange is None:
                    self.logger.error("Exchange not available for live trading")
                    return None
                
                # Place real order
                order_params = {
                    'stopLoss': stop_loss,
                    'takeProfit': take_profit
                }
                
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=signal.lower(),
                    amount=position_size,
                    params=order_params
                )
                
                # Record trade
                trade_result = {
                    'symbol': symbol,
                    'side': signal,
                    'amount': position_size,
                    'entry_price': order['price'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'status': 'OPEN',
                    'type': 'LIVE',
                    'order_id': order['id']
                }
                
                self.trades.append(trade_result)
                self.logger.info(f"Live trade executed: {signal} {position_size} {symbol}")
                
                return trade_result
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            if self.exchange is None:
                # Return sample price for demo mode
                return 1.1000 if 'USD' in symbol else 110.0
            
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 1.1000 if 'USD' in symbol else 110.0
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            self.logger.info("Starting trading cycle...")
            
            # Check if we can trade
            if len(self.trades) >= self.max_positions:
                self.logger.info("Maximum positions reached, skipping trading cycle")
                return
            
            # Get account balance
            account_balance = self._get_account_balance()
            if account_balance <= 0:
                self.logger.warning("Insufficient balance for trading")
                return
            
            # Analyze each symbol
            for symbol in self.symbols:
                try:
                    # Get comprehensive analysis
                    analysis = self.get_comprehensive_analysis(symbol)
                    if not analysis:
                        continue
                    
                    # Get combined signal
                    combined_signal = analysis.get('combined_signal', {})
                    signal = combined_signal.get('signal', 'HOLD')
                    confidence = combined_signal.get('confidence', 0.0)
                    
                    if signal == 'HOLD' or confidence < self.config['ml_config']['confidence_threshold']:
                        continue
                    
                    # Get current market data for position sizing
                    market_data = self._get_market_data(symbol, limit=50)
                    if market_data.empty:
                        continue
                    
                    # Calculate volatility for position sizing
                    returns = market_data['close'].pct_change().dropna()
                    market_volatility = returns.std()
                    
                    # Calculate position size
                    position_size, actual_risk, risk_percentage = self.calculate_position_size(
                        signal, confidence, account_balance, market_volatility
                    )
                    
                    if position_size <= 0:
                        continue
                    
                    # Calculate stop loss and take profit
                    current_price = market_data['close'].iloc[-1]
                    if signal == 'BUY':
                        stop_loss = current_price - (self.stop_loss_pips * 0.0001)
                        take_profit = current_price + (self.take_profit_pips * 0.0001)
                    else:
                        stop_loss = current_price + (self.stop_loss_pips * 0.0001)
                        take_profit = current_price - (self.take_profit_pips * 0.0001)
                    
                    # Execute trade
                    trade_result = self.execute_trade(symbol, signal, confidence, 
                                                   position_size, stop_loss, take_profit)
                    
                    if trade_result:
                        self.logger.info(f"Trade executed: {signal} {position_size} {symbol}")
                        
                        # Update ML models with new data
                        self._update_ml_models(symbol, analysis)
                    
                    # Wait between trades
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Log current status
            self._log_trading_status()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def _update_ml_models(self, symbol, analysis):
        """Update ML models with new market data"""
        try:
            # Get latest market data
            market_data = self._get_market_data(symbol, limit=100)
            
            if not market_data.empty:
                # Update ML models
                self.ml_engine.update_models(market_data)
                
        except Exception as e:
            self.logger.error(f"Error updating ML models: {e}")
    
    def _get_account_balance(self):
        """Get current account balance"""
        try:
            if self.exchange is None:
                # Return sample balance for demo mode
                return 10000.0
            
            balance = self.exchange.fetch_balance()
            return float(balance['total']['USD'])
            
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 10000.0  # Default balance for demo mode
    
    def _log_trading_status(self):
        """Log current trading status"""
        try:
            balance = self._get_account_balance()
            open_positions = len([t for t in self.trades if t['status'] == 'OPEN'])
            
            self.logger.info(f"Current balance: ${balance:.2f}")
            self.logger.info(f"Open positions: {open_positions}")
            self.logger.info(f"Total trades: {len(self.trades)}")
            
        except Exception as e:
            self.logger.error(f"Error logging trading status: {e}")
    
    def start_bot(self, interval_minutes=15):
        """Start the trading bot"""
        try:
            if self.is_running:
                self.logger.warning("Bot is already running")
                return
            
            self.is_running = True
            self.logger.info(f"Starting Enhanced Forex Trading Bot with {interval_minutes} minute intervals")
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self._run_bot_loop, args=(interval_minutes,))
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            self.is_running = False
    
    def stop_bot(self):
        """Stop the trading bot"""
        try:
            self.is_running = False
            self.logger.info("Stopping Enhanced Forex Trading Bot...")
            
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            self.logger.info("Bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    def _run_bot_loop(self, interval_minutes):
        """Main bot loop"""
        try:
            while self.is_running:
                # Run trading cycle
                self.run_trading_cycle()
                
                # Wait for next cycle
                if self.is_running:
                    self.logger.info(f"Waiting {interval_minutes} minutes until next cycle...")
                    time.sleep(interval_minutes * 60)
                    
        except Exception as e:
            self.logger.error(f"Error in bot loop: {e}")
            self.is_running = False
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        try:
            summary = {
                'total_trades': len(self.trades),
                'open_positions': len([t for t in self.trades if t['status'] == 'OPEN']),
                'closed_positions': len([t for t in self.trades if t['status'] == 'CLOSED']),
                'account_balance': self._get_account_balance(),
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'trading_mode': 'LIVE' if self.live_trading_mode else 'PAPER',
                'ml_engine_status': self.ml_engine.get_model_performance_summary(),
                'bot_status': 'RUNNING' if self.is_running else 'STOPPED'
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    bot = EnhancedForexTradingBot()
    
    # Get comprehensive analysis for a symbol
    analysis = bot.get_comprehensive_analysis('EUR/USD')
    print("Comprehensive Analysis:", json.dumps(analysis, indent=2, default=str))
    
    # Start the bot (uncomment to start live trading)
    # bot.start_bot(interval_minutes=15)
    
    # Get performance summary
    performance = bot.get_performance_summary()
    print("Performance Summary:", json.dumps(performance, indent=2, default=str))