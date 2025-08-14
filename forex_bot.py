import ccxt
import pandas as pd
import numpy as np
import ta
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ForexTradingBot:
    def __init__(self, exchange_name: str = 'oanda', api_key: str = None, secret: str = None):
        """
        Initialize the Forex Trading Bot
        
        Args:
            exchange_name: Name of the exchange (default: oanda)
            api_key: API key for the exchange
            secret: Secret key for the exchange
        """
        self.exchange_name = exchange_name
        self.api_key = api_key or os.getenv('EXCHANGE_API_KEY')
        self.secret = secret or os.getenv('EXCHANGE_SECRET')
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Trading parameters
        self.symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
        self.timeframe = '1h'
        self.position_size = 0.01  # Default lot size
        self.max_positions = 3
        self.stop_loss_pips = 50
        self.take_profit_pips = 100
        
        # Risk management
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.max_portfolio_risk = 0.05  # 5% max portfolio risk
        
        # Performance tracking
        self.trades = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Setup logging
        self._setup_logging()
        
    def _initialize_exchange(self):
        """Initialize the exchange connection"""
        try:
            if self.exchange_name == 'oanda':
                exchange_class = getattr(ccxt, 'oanda')
                exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.secret,
                    'sandbox': True,  # Use sandbox for testing
                    'enableRateLimit': True,
                })
            else:
                exchange_class = getattr(ccxt, self.exchange_name)
                exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.secret,
                    'enableRateLimit': True,
                })
            
            # Test connection
            exchange.load_markets()
            self.logger.info(f"Successfully connected to {self.exchange_name}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('forex_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_historical_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def rsi_strategy(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        RSI-based trading strategy
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            current_rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            
            # Oversold condition (RSI < 30)
            if current_rsi < 30 and prev_rsi >= 30:
                return 'BUY', 0.8
            # Overbought condition (RSI > 70)
            elif current_rsi > 70 and prev_rsi <= 70:
                return 'SELL', 0.8
            
            # Divergence signals
            elif current_rsi < 40 and df['close'].iloc[-1] > df['close'].iloc[-5]:
                return 'BUY', 0.6
            elif current_rsi > 60 and df['close'].iloc[-1] < df['close'].iloc[-5]:
                return 'SELL', 0.6
            
            return 'HOLD', 0.0
            
        except Exception as e:
            self.logger.error(f"Error in RSI strategy: {e}")
            return 'HOLD', 0.0
    
    def macd_strategy(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        MACD-based trading strategy
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            prev_macd = df['macd'].iloc[-2]
            prev_signal = df['macd_signal'].iloc[-2]
            
            # MACD crossover
            if current_macd > current_signal and prev_macd <= prev_signal:
                return 'BUY', 0.7
            elif current_macd < current_signal and prev_macd >= prev_signal:
                return 'SELL', 0.7
            
            # MACD histogram momentum
            if df['macd_histogram'].iloc[-1] > 0 and df['macd_histogram'].iloc[-2] < 0:
                return 'BUY', 0.5
            elif df['macd_histogram'].iloc[-1] < 0 and df['macd_histogram'].iloc[-2] > 0:
                return 'SELL', 0.5
            
            return 'HOLD', 0.0
            
        except Exception as e:
            self.logger.error(f"Error in MACD strategy: {e}")
            return 'HOLD', 0.0
    
    def moving_average_strategy(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Moving average crossover strategy
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            current_close = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            ema_12 = df['ema_12'].iloc[-1]
            ema_26 = df['ema_26'].iloc[-1]
            
            # SMA crossover
            if sma_20 > sma_50 and current_close > sma_20:
                return 'BUY', 0.6
            elif sma_20 < sma_50 and current_close < sma_20:
                return 'SELL', 0.6
            
            # EMA crossover
            if ema_12 > ema_26 and current_close > ema_12:
                return 'BUY', 0.5
            elif ema_12 < ema_26 and current_close < ema_12:
                return 'SELL', 0.5
            
            return 'HOLD', 0.0
            
        except Exception as e:
            self.logger.error(f"Error in Moving Average strategy: {e}")
            return 'HOLD', 0.0
    
    def bollinger_bands_strategy(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Bollinger Bands strategy
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            current_close = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1]
            
            # Price near lower band (oversold)
            if current_close <= bb_lower * 1.01:
                return 'BUY', 0.7
            # Price near upper band (overbought)
            elif current_close >= bb_upper * 0.99:
                return 'SELL', 0.7
            
            # Price crossing middle band
            elif current_close > bb_middle and df['close'].iloc[-2] <= df['bb_middle'].iloc[-2]:
                return 'BUY', 0.5
            elif current_close < bb_middle and df['close'].iloc[-2] >= df['bb_middle'].iloc[-2]:
                return 'SELL', 0.5
            
            return 'HOLD', 0.0
            
        except Exception as e:
            self.logger.error(f"Error in Bollinger Bands strategy: {e}")
            return 'HOLD', 0.0
    
    def get_combined_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Combine signals from all strategies
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            strategies = [
                self.rsi_strategy(df),
                self.macd_strategy(df),
                self.moving_average_strategy(df),
                self.bollinger_bands_strategy(df)
            ]
            
            buy_signals = [s for s in strategies if s[0] == 'BUY']
            sell_signals = [s for s in strategies if s[0] == 'SELL']
            
            if buy_signals:
                avg_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
                if avg_confidence >= 0.6:
                    return 'BUY', avg_confidence
            
            if sell_signals:
                avg_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
                if avg_confidence >= 0.6:
                    return 'SELL', avg_confidence
            
            return 'HOLD', 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting combined signal: {e}")
            return 'HOLD', 0.0
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk per trade as percentage
            
        Returns:
            Position size in lots
        """
        try:
            risk_amount = account_balance * risk_per_trade
            # Assuming 1 pip = $1 for 0.01 lot on major pairs
            pip_value = 1.0
            max_loss_pips = self.stop_loss_pips
            
            position_size = risk_amount / (max_loss_pips * pip_value)
            return max(0.01, min(position_size, 1.0))  # Between 0.01 and 1.0 lots
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    def place_order(self, symbol: str, side: str, amount: float, 
                   stop_loss: float = None, take_profit: float = None) -> Dict:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            amount: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order information
        """
        try:
            order_params = {}
            if stop_loss:
                order_params['stopLoss'] = stop_loss
            if take_profit:
                order_params['takeProfit'] = take_profit
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=amount,
                params=order_params
            )
            
            self.logger.info(f"Order placed: {side} {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {}
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['total']['USD'])
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0
    
    def get_open_positions(self) -> List[Dict]:
        """Get current open positions"""
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if p['size'] != 0]
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []
    
    def run_strategy_analysis(self, symbol: str) -> Dict:
        """
        Run complete strategy analysis for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Get historical data
            df = self.get_historical_data(symbol, self.timeframe, 100)
            if df.empty:
                return {}
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Get signals from all strategies
            rsi_signal, rsi_conf = self.rsi_strategy(df)
            macd_signal, macd_conf = self.macd_strategy(df)
            ma_signal, ma_conf = self.moving_average_strategy(df)
            bb_signal, bb_conf = self.bollinger_bands_strategy(df)
            
            # Get combined signal
            combined_signal, combined_conf = self.get_combined_signal(df)
            
            # Current market data
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'signals': {
                    'rsi': {'signal': rsi_signal, 'confidence': rsi_conf, 'value': current_rsi},
                    'macd': {'signal': macd_signal, 'confidence': macd_conf, 'value': current_macd},
                    'moving_average': {'signal': ma_signal, 'confidence': ma_conf},
                    'bollinger_bands': {'signal': bb_signal, 'confidence': bb_conf},
                    'combined': {'signal': combined_signal, 'confidence': combined_conf}
                },
                'indicators': {
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'sma_20': df['sma_20'].iloc[-1],
                    'sma_50': df['sma_50'].iloc[-1],
                    'bb_upper': df['bb_upper'].iloc[-1],
                    'bb_lower': df['bb_lower'].iloc[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in strategy analysis for {symbol}: {e}")
            return {}
    
    def execute_trades(self):
        """Execute trading decisions based on strategy analysis"""
        try:
            # Check if we can trade
            if len(self.get_open_positions()) >= self.max_positions:
                self.logger.info("Maximum positions reached, skipping trade execution")
                return
            
            # Get account balance
            balance = self.get_account_balance()
            if balance <= 0:
                self.logger.warning("Insufficient balance for trading")
                return
            
            # Analyze each symbol
            for symbol in self.symbols:
                analysis = self.run_strategy_analysis(symbol)
                if not analysis:
                    continue
                
                signal = analysis['signals']['combined']['signal']
                confidence = analysis['signals']['combined']['confidence']
                
                if signal == 'HOLD' or confidence < 0.6:
                    continue
                
                # Calculate position size
                position_size = self.calculate_position_size(balance)
                
                # Calculate stop loss and take profit
                current_price = analysis['current_price']
                if signal == 'BUY':
                    stop_loss = current_price - (self.stop_loss_pips * 0.0001)
                    take_profit = current_price + (self.take_profit_pips * 0.0001)
                else:
                    stop_loss = current_price + (self.stop_loss_pips * 0.0001)
                    take_profit = current_price - (self.take_profit_pips * 0.0001)
                
                # Place order
                order = self.place_order(symbol, signal, position_size, stop_loss, take_profit)
                
                if order:
                    self.logger.info(f"Trade executed: {signal} {position_size} {symbol}")
                    self.trades.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'side': signal,
                        'amount': position_size,
                        'price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': confidence
                    })
                
                # Wait between trades
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")
    
    def run_bot(self, interval_minutes: int = 15):
        """
        Run the trading bot continuously
        
        Args:
            interval_minutes: Interval between trading cycles
        """
        self.logger.info(f"Starting Forex Trading Bot with {interval_minutes} minute intervals")
        
        try:
            while True:
                self.logger.info("Starting trading cycle...")
                
                # Execute trades
                self.execute_trades()
                
                # Log current status
                balance = self.get_account_balance()
                positions = self.get_open_positions()
                
                self.logger.info(f"Current balance: ${balance:.2f}")
                self.logger.info(f"Open positions: {len(positions)}")
                
                # Wait for next cycle
                self.logger.info(f"Waiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    bot = ForexTradingBot()
    
    # Run single analysis
    analysis = bot.run_strategy_analysis('EUR/USD')
    print("Strategy Analysis:", analysis)
    
    # Run bot (uncomment to start live trading)
    # bot.run_bot(interval_minutes=15)