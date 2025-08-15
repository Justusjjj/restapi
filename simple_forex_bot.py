#!/usr/bin/env python3
"""
Simple Forex Trading Bot - No External Dependencies
"""

import math
import random
import time
from datetime import datetime, timedelta

class SimpleForexBot:
    """Simple Forex Bot with Basic Features"""
    
    def __init__(self):
        self.positions = {}
        self.trading_history = []
        self.account_balance = 100000
        self.max_positions = 3
        self.risk_per_trade = 0.02
        
    def generate_market_data(self, symbol):
        """Generate simple market data"""
        base_prices = {
            'EUR/USD': 1.1000,
            'GBP/USD': 1.2500,
            'USD/JPY': 150.00,
            'AUD/USD': 0.6500
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Simple price movement
        change = random.uniform(-0.0020, 0.0020)
        current_price = base_price + change
        
        return {
            'symbol': symbol,
            'price': current_price,
            'timestamp': datetime.now()
        }
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analyze_market(self, symbol):
        """Analyze market and generate signal"""
        # Generate some price history
        prices = []
        for _ in range(20):
            data = self.generate_market_data(symbol)
            prices.append(data['price'])
        
        # Calculate RSI
        rsi = self.calculate_rsi(prices)
        
        # Simple signal generation
        if rsi < 30:
            signal = 'BUY'
            confidence = 0.8
        elif rsi > 70:
            signal = 'SELL'
            confidence = 0.8
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'rsi': rsi,
            'current_price': prices[-1],
            'timestamp': datetime.now()
        }
    
    def execute_trade(self, symbol, signal, confidence, price):
        """Execute a trade"""
        if signal == 'HOLD':
            return {'status': 'no_action', 'reason': 'HOLD signal'}
        
        if len(self.positions) >= self.max_positions:
            return {'status': 'rejected', 'reason': 'Maximum positions reached'}
        
        # Calculate position size
        risk_amount = self.account_balance * self.risk_per_trade
        stop_loss_pips = 50 / 10000  # 50 pips
        
        if signal == 'BUY':
            stop_loss = price - stop_loss_pips
            take_profit = price + (stop_loss_pips * 2)
        else:  # SELL
            stop_loss = price + stop_loss_pips
            take_profit = price - (stop_loss_pips * 2)
        
        position_size = risk_amount / abs(price - stop_loss)
        
        # Create trade
        trade = {
            'id': len(self.trading_history) + 1,
            'symbol': symbol,
            'side': signal,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': datetime.now(),
            'status': 'open'
        }
        
        # Add to positions and history
        self.positions[symbol] = trade
        self.trading_history.append(trade)
        
        return {
            'status': 'executed',
            'trade': trade,
            'message': f'{signal} {symbol} at {price:.4f}'
        }
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        print("🔄 Running trading cycle...")
        
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
        
        for symbol in symbols:
            # Analyze market
            analysis = self.analyze_market(symbol)
            print(f"📊 {symbol}: RSI={analysis['rsi']:.1f}, Signal={analysis['signal']}")
            
            # Execute trade if signal is not HOLD
            if analysis['signal'] != 'HOLD':
                trade_result = self.execute_trade(
                    symbol, 
                    analysis['signal'], 
                    analysis['confidence'], 
                    analysis['current_price']
                )
                print(f"💼 Trade result: {trade_result['status']}")
            
            # Check existing positions
            if symbol in self.positions:
                self._check_position(symbol)
        
        print(f"📈 Status: {len(self.positions)} open positions, {len(self.trading_history)} total trades")
    
    def _check_position(self, symbol):
        """Check and potentially close a position"""
        position = self.positions[symbol]
        
        # Get current price
        current_data = self.generate_market_data(symbol)
        current_price = current_data['price']
        
        # Check stop loss and take profit
        if position['side'] == 'BUY':
            if current_price <= position['stop_loss']:
                self._close_position(symbol, current_price, 'stop_loss')
            elif current_price >= position['take_profit']:
                self._close_position(symbol, current_price, 'take_profit')
        else:  # SELL
            if current_price >= position['stop_loss']:
                self._close_position(symbol, current_price, 'stop_loss')
            elif current_price <= position['take_profit']:
                self._close_position(symbol, current_price, 'take_profit')
    
    def _close_position(self, symbol, exit_price, reason):
        """Close a position"""
        position = self.positions[symbol]
        
        # Calculate P&L
        if position['side'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:  # SELL
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Update trade
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['status'] = 'closed'
        position['exit_reason'] = reason
        position['pnl'] = pnl
        
        # Remove from open positions
        del self.positions[symbol]
        
        print(f"🔒 Position closed: {symbol} {position['side']} P&L: ${pnl:.2f}")
    
    def get_status(self):
        """Get bot status"""
        closed_trades = [t for t in self.trading_history if t.get('status') == 'closed']
        
        if closed_trades:
            total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            
            return {
                'open_positions': len(self.positions),
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
                'total_pnl': total_pnl
            }
        else:
            return {
                'open_positions': len(self.positions),
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0
            }

def main():
    """Main function"""
    print("🚀 SIMPLE FOREX TRADING BOT")
    print("=" * 40)
    
    # Initialize bot
    bot = SimpleForexBot()
    
    try:
        print("✅ Bot initialized successfully!")
        
        # Run trading cycles
        for cycle in range(3):
            print(f"\n🔄 Running trading cycle {cycle + 1}/3...")
            bot.run_trading_cycle()
            
            # Show status
            status = bot.get_status()
            print(f"📊 Status: {status['open_positions']} open positions, {status['total_trades']} total trades")
            
            time.sleep(2)  # Wait between cycles
        
        # Show final performance
        final_status = bot.get_status()
        print("\n📈 FINAL PERFORMANCE:")
        print(f"Total Trades: {final_status['total_trades']}")
        print(f"Win Rate: {final_status['win_rate']:.1%}")
        print(f"Total P&L: ${final_status['total_pnl']:.2f}")
        
        print("\n🎉 Bot demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Bot interrupted by user")
    except Exception as e:
        print(f"❌ Error running bot: {e}")

if __name__ == "__main__":
    main()