"""
Example Usage
=============

Example demonstrating how to use the quantitative trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from data_engine import DataLoader, DataNormalizer, DataStorage, ReplayEngine
from features import PriceIndicators, VolatilityIndicators, MomentumIndicators
from strategies import MeanReversionStrategy, MomentumStrategy
from backtest_engine import BacktestEngine
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample market data for demonstration."""
    logger.info("Creating sample market data...")
    
    # Generate sample OHLCV data
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='1min')
    n_periods = len(dates)
    
    # Generate realistic price data with trend and volatility
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, n_periods)  # 0.1% volatility per minute
    returns[::100] += np.random.normal(0, 0.005, len(returns[::100]))  # Add some jumps
    
    # Create price series
    prices = 1.1000 * np.exp(np.cumsum(returns))  # Start at 1.1000
    
    # Generate OHLCV data
    data = pd.DataFrame(index=dates)
    data['open'] = prices
    data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.0005, n_periods)))
    data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.0005, n_periods)))
    data['close'] = prices * (1 + np.random.normal(0, 0.0002, n_periods))
    data['volume'] = np.random.randint(1000, 10000, n_periods)
    
    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add bid/ask for tick data simulation
    data['bid'] = data['close'] - 0.0001
    data['ask'] = data['close'] + 0.0001
    data['mid'] = (data['bid'] + data['ask']) / 2
    data['spread'] = data['ask'] - data['bid']
    
    return data


def run_example():
    """Run example backtest."""
    logger.info("Starting example backtest...")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Initialize data components
    data_storage = DataStorage("data/storage")
    data_normalizer = DataNormalizer()
    
    # Normalize and save sample data
    normalized_data = data_normalizer.normalize(sample_data, 'ohlcv')
    data_storage.save_ohlcv(normalized_data, 'EURUSD', '1m', overwrite=True)
    
    # Initialize strategies
    mean_reversion_config = {
        'zscore_threshold': 2.0,
        'zscore_exit_threshold': 0.5,
        'vwap_window': 20,
        'zscore_window': 20,
        'atr_window': 14,
        'target_volatility': 0.02,
        'max_position_size': 0.1
    }
    
    momentum_config = {
        'fast_ema': 12,
        'slow_ema': 26,
        'signal_ema': 9,
        'rsi_window': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'atr_window': 14,
        'target_volatility': 0.02,
        'max_position_size': 0.1
    }
    
    mean_reversion_strategy = MeanReversionStrategy('MeanReversion', mean_reversion_config)
    momentum_strategy = MomentumStrategy('Momentum', momentum_config)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(initial_capital=100000.0)
    backtest_engine.add_strategy(mean_reversion_strategy)
    backtest_engine.add_strategy(momentum_strategy)
    
    # Set up replay engine
    replay_engine = ReplayEngine(data_storage)
    replay_engine.instruments = ['EURUSD']
    replay_engine.data_cache = {'EURUSD': normalized_data}
    
    # Run backtest
    results = backtest_engine.run_backtest(replay_engine)
    
    # Print results
    print("\n" + "="*60)
    print("EXAMPLE BACKTEST RESULTS")
    print("="*60)
    
    portfolio = results.get('portfolio_metrics', {})
    print(f"Total Return: {portfolio.get('total_return', 0):.2%}")
    print(f"Annualized Return: {portfolio.get('annualized_return', 0):.2%}")
    print(f"Volatility: {portfolio.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {portfolio.get('max_drawdown', 0):.2%}")
    
    trades = results.get('trade_summary', {})
    print(f"\nTotal Trades: {trades.get('total_trades', 0)}")
    print(f"Total Commission: ${trades.get('total_commission', 0):.2f}")
    
    strategies = results.get('strategy_metrics', {})
    print(f"\nStrategy Performance:")
    for name, metrics in strategies.items():
        print(f"  {name}: {metrics.get('total_positions', 0)} positions")
        
    print("="*60)
    
    # Demonstrate feature calculation
    print("\nFeature Calculation Example:")
    print("-" * 40)
    
    # Calculate some indicators
    vwap = PriceIndicators.vwap(
        normalized_data['high'], 
        normalized_data['low'], 
        normalized_data['close'], 
        normalized_data['volume']
    )
    
    rsi = MomentumIndicators.rsi(normalized_data['close'], 14)
    atr = VolatilityIndicators.atr(
        normalized_data['high'], 
        normalized_data['low'], 
        normalized_data['close'], 
        14
    )
    
    print(f"Latest VWAP: {vwap.iloc[-1]:.5f}")
    print(f"Latest RSI: {rsi.iloc[-1]:.2f}")
    print(f"Latest ATR: {atr.iloc[-1]:.5f}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    run_example()