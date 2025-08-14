import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os

class ForexBacktester:
    def __init__(self, initial_balance: float = 10000):
        """
        Initialize the Forex Backtester
        
        Args:
            initial_balance: Starting account balance
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
    def load_data(self, data_file: str) -> pd.DataFrame:
        """
        Load historical data from CSV file
        
        Args:
            data_file: Path to CSV file with OHLCV data
            
        Returns:
            DataFrame with historical data
        """
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for backtesting
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        try:
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            return df
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with signals added
        """
        try:
            df['signal'] = 'HOLD'
            df['signal_strength'] = 0.0
            
            for i in range(1, len(df)):
                # RSI signals
                rsi_signal, rsi_conf = self._rsi_signal(df, i)
                
                # MACD signals
                macd_signal, macd_conf = self._macd_signal(df, i)
                
                # Moving Average signals
                ma_signal, ma_conf = self._ma_signal(df, i)
                
                # Bollinger Bands signals
                bb_signal, bb_conf = self._bb_signal(df, i)
                
                # Combine signals
                combined_signal, combined_conf = self._combine_signals([
                    (rsi_signal, rsi_conf),
                    (macd_signal, macd_conf),
                    (ma_signal, ma_conf),
                    (bb_signal, bb_conf)
                ])
                
                df.iloc[i, df.columns.get_loc('signal')] = combined_signal
                df.iloc[i, df.columns.get_loc('signal_strength')] = combined_conf
            
            return df
        except Exception as e:
            print(f"Error generating signals: {e}")
            return df
    
    def _rsi_signal(self, df: pd.DataFrame, index: int) -> Tuple[str, float]:
        """Generate RSI-based signals"""
        try:
            current_rsi = df.iloc[index]['rsi']
            prev_rsi = df.iloc[index-1]['rsi']
            
            if pd.isna(current_rsi) or pd.isna(prev_rsi):
                return 'HOLD', 0.0
            
            if current_rsi < 30 and prev_rsi >= 30:
                return 'BUY', 0.8
            elif current_rsi > 70 and prev_rsi <= 70:
                return 'SELL', 0.8
            elif current_rsi < 40:
                return 'BUY', 0.6
            elif current_rsi > 60:
                return 'SELL', 0.6
            
            return 'HOLD', 0.0
        except:
            return 'HOLD', 0.0
    
    def _macd_signal(self, df: pd.DataFrame, index: int) -> Tuple[str, float]:
        """Generate MACD-based signals"""
        try:
            current_macd = df.iloc[index]['macd']
            current_signal = df.iloc[index]['macd_signal']
            prev_macd = df.iloc[index-1]['macd']
            prev_signal = df.iloc[index-1]['macd_signal']
            
            if pd.isna(current_macd) or pd.isna(current_signal):
                return 'HOLD', 0.0
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                return 'BUY', 0.7
            elif current_macd < current_signal and prev_macd >= prev_signal:
                return 'SELL', 0.7
            
            return 'HOLD', 0.0
        except:
            return 'HOLD', 0.0
    
    def _ma_signal(self, df: pd.DataFrame, index: int) -> Tuple[str, float]:
        """Generate Moving Average-based signals"""
        try:
            current_close = df.iloc[index]['close']
            sma_20 = df.iloc[index]['sma_20']
            sma_50 = df.iloc[index]['sma_50']
            
            if pd.isna(sma_20) or pd.isna(sma_50):
                return 'HOLD', 0.0
            
            if sma_20 > sma_50 and current_close > sma_20:
                return 'BUY', 0.6
            elif sma_20 < sma_50 and current_close < sma_20:
                return 'SELL', 0.6
            
            return 'HOLD', 0.0
        except:
            return 'HOLD', 0.0
    
    def _bb_signal(self, df: pd.DataFrame, index: int) -> Tuple[str, float]:
        """Generate Bollinger Bands-based signals"""
        try:
            current_close = df.iloc[index]['close']
            bb_upper = df.iloc[index]['bb_upper']
            bb_lower = df.iloc[index]['bb_lower']
            
            if pd.isna(bb_upper) or pd.isna(bb_lower):
                return 'HOLD', 0.0
            
            if current_close <= bb_lower * 1.01:
                return 'BUY', 0.7
            elif current_close >= bb_upper * 0.99:
                return 'SELL', 0.7
            
            return 'HOLD', 0.0
        except:
            return 'HOLD', 0.0
    
    def _combine_signals(self, signals: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Combine multiple signals into a single decision"""
        buy_signals = [s for s in signals if s[0] == 'BUY']
        sell_signals = [s for s in signals if s[0] == 'SELL']
        
        if buy_signals:
            avg_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
            if avg_confidence >= 0.6:
                return 'BUY', avg_confidence
        
        if sell_signals:
            avg_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
            if avg_confidence >= 0.6:
                return 'SELL', avg_confidence
        
        return 'HOLD', 0.0
    
    def run_backtest(self, df: pd.DataFrame, position_size: float = 0.01, 
                    stop_loss_pips: int = 50, take_profit_pips: int = 100) -> Dict:
        """
        Run the backtest
        
        Args:
            df: DataFrame with signals
            position_size: Position size in lots
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.balance = self.initial_balance
            self.positions = []
            self.trades = []
            self.equity_curve = []
            self.max_drawdown = 0
            self.peak_balance = self.initial_balance
            
            for i in range(len(df)):
                current_row = df.iloc[i]
                current_price = current_row['close']
                signal = current_row['signal']
                timestamp = df.index[i]
                
                # Update equity curve
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'balance': self.balance,
                    'open_positions': len(self.positions)
                })
                
                # Check for stop loss or take profit on existing positions
                self._check_exit_conditions(current_price, timestamp)
                
                # Generate new signals
                if signal in ['BUY', 'SELL'] and len(self.positions) < 3:
                    self._execute_trade(signal, current_price, position_size, 
                                     stop_loss_pips, take_profit_pips, timestamp)
                
                # Update max drawdown
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                else:
                    drawdown = (self.peak_balance - self.balance) / self.peak_balance
                    if drawdown > self.max_drawdown:
                        self.max_drawdown = drawdown
            
            # Close any remaining positions
            final_price = df.iloc[-1]['close']
            for position in self.positions:
                self._close_position(position, final_price, df.index[-1])
            
            return self._calculate_results()
            
        except Exception as e:
            print(f"Error in backtest: {e}")
            return {}
    
    def _execute_trade(self, signal: str, price: float, size: float, 
                       stop_loss_pips: int, take_profit_pips: int, timestamp):
        """Execute a new trade"""
        try:
            # Calculate stop loss and take profit
            if signal == 'BUY':
                stop_loss = price - (stop_loss_pips * 0.0001)
                take_profit = price + (take_profit_pips * 0.0001)
            else:
                stop_loss = price + (stop_loss_pips * 0.0001)
                take_profit = price - (take_profit_pips * 0.0001)
            
            # Create position
            position = {
                'id': len(self.positions) + 1,
                'side': signal,
                'entry_price': price,
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': timestamp,
                'status': 'OPEN'
            }
            
            self.positions.append(position)
            
            # Record trade
            self.trades.append({
                'timestamp': timestamp,
                'action': 'OPEN',
                'side': signal,
                'price': price,
                'size': size,
                'balance': self.balance
            })
            
        except Exception as e:
            print(f"Error executing trade: {e}")
    
    def _check_exit_conditions(self, current_price: float, timestamp):
        """Check if any positions should be closed"""
        try:
            positions_to_close = []
            
            for position in self.positions:
                if position['status'] != 'OPEN':
                    continue
                
                # Check stop loss
                if (position['side'] == 'BUY' and current_price <= position['stop_loss']) or \
                   (position['side'] == 'SELL' and current_price >= position['stop_loss']):
                    position['status'] = 'STOP_LOSS'
                    position['exit_price'] = position['stop_loss']
                    position['exit_time'] = timestamp
                    positions_to_close.append(position)
                
                # Check take profit
                elif (position['side'] == 'BUY' and current_price >= position['take_profit']) or \
                     (position['side'] == 'SELL' and current_price <= position['take_profit']):
                    position['status'] = 'TAKE_PROFIT'
                    position['exit_price'] = position['take_profit']
                    position['exit_time'] = timestamp
                    positions_to_close.append(position)
            
            # Close positions
            for position in positions_to_close:
                self._close_position(position, position['exit_price'], timestamp)
                
        except Exception as e:
            print(f"Error checking exit conditions: {e}")
    
    def _close_position(self, position: Dict, exit_price: float, timestamp):
        """Close a position and calculate P&L"""
        try:
            # Calculate P&L
            if position['side'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['size'] * 100000  # Convert to pips
            else:
                pnl = (position['entry_price'] - exit_price) * position['size'] * 100000
            
            # Update balance
            self.balance += pnl
            
            # Record trade
            self.trades.append({
                'timestamp': timestamp,
                'action': 'CLOSE',
                'side': position['side'],
                'price': exit_price,
                'size': position['size'],
                'pnl': pnl,
                'balance': self.balance,
                'exit_reason': position['status']
            })
            
            # Remove from open positions
            self.positions = [p for p in self.positions if p['id'] != position['id']]
            
        except Exception as e:
            print(f"Error closing position: {e}")
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics"""
        try:
            if not self.trades:
                return {}
            
            # Calculate metrics
            total_trades = len([t for t in self.trades if t['action'] == 'CLOSE'])
            winning_trades = len([t for t in self.trades if t['action'] == 'CLOSE' and t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trades if t['action'] == 'CLOSE' and t.get('pnl', 0) < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum([t.get('pnl', 0) for t in self.trades if t['action'] == 'CLOSE'])
            avg_win = np.mean([t.get('pnl', 0) for t in self.trades if t['action'] == 'CLOSE' and t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t.get('pnl', 0) for t in self.trades if t['action'] == 'CLOSE' and t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            returns = pd.Series([t.get('pnl', 0) for t in self.trades if t['action'] == 'CLOSE'])
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_return': (self.balance - self.initial_balance) / self.initial_balance,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'trades': self.trades,
                'equity_curve': self.equity_curve
            }
            
        except Exception as e:
            print(f"Error calculating results: {e}")
            return {}
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot backtest results
        
        Args:
            results: Backtest results dictionary
            save_path: Path to save the plot
        """
        try:
            if not results:
                print("No results to plot")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Forex Trading Bot Backtest Results', fontsize=16)
            
            # Equity curve
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df.set_index('timestamp', inplace=True)
            axes[0, 0].plot(equity_df.index, equity_df['balance'])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Balance ($)')
            axes[0, 0].grid(True)
            
            # Trade P&L distribution
            trades_df = pd.DataFrame([t for t in results['trades'] if t['action'] == 'CLOSE'])
            if not trades_df.empty:
                axes[0, 1].hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Trade P&L Distribution')
                axes[0, 1].set_xlabel('P&L ($)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True)
            
            # Win rate pie chart
            win_rate = results['win_rate']
            axes[1, 0].pie([win_rate, 1-win_rate], labels=['Wins', 'Losses'], 
                           autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Win Rate')
            
            # Performance metrics table
            metrics_text = f"""
            Total Return: {results['total_return']:.2%}
            Total P&L: ${results['total_pnl']:.2f}
            Total Trades: {results['total_trades']}
            Win Rate: {results['win_rate']:.2%}
            Profit Factor: {results['profit_factor']:.2f}
            Max Drawdown: {results['max_drawdown']:.2%}
            Sharpe Ratio: {results['sharpe_ratio']:.2f}
            """
            axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='center')
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting results: {e}")
    
    def save_results(self, results: Dict, file_path: str):
        """Save backtest results to JSON file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = results.copy()
            serializable_results['trades'] = []
            
            for trade in results['trades']:
                serializable_trade = trade.copy()
                if 'timestamp' in serializable_trade:
                    serializable_trade['timestamp'] = serializable_trade['timestamp'].isoformat()
                serializable_results['trades'].append(serializable_trade)
            
            serializable_results['equity_curve'] = []
            for point in results['equity_curve']:
                serializable_point = point.copy()
                if 'timestamp' in serializable_point:
                    serializable_point['timestamp'] = serializable_point['timestamp'].isoformat()
                serializable_results['equity_curve'].append(serializable_point)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"Results saved to {file_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    # Example usage
    backtester = ForexBacktester(initial_balance=10000)
    
    # Load sample data (you would need to provide your own CSV file)
    # df = backtester.load_data('sample_data.csv')
    
    # Calculate indicators
    # df = backtester.calculate_indicators(df)
    
    # Generate signals
    # df = backtester.generate_signals(df)
    
    # Run backtest
    # results = backtester.run_backtest(df)
    
    # Plot results
    # backtester.plot_results(results)
    
    print("Backtester initialized. Load your data and run backtest() to get started.")