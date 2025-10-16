"""
Main Entry Point
================

Main entry point for the quantitative trading system.
"""

import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json

from data_engine import DataLoader, DataNormalizer, DataStorage, ReplayEngine
from features import PriceIndicators, VolatilityIndicators, MomentumIndicators
from strategies import MeanReversionStrategy, MomentumStrategy, PairTradingStrategy
from backtest_engine import BacktestEngine
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quantitative Trading System')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['backtest', 'live', 'paper'],
                       default='backtest', help='Trading mode')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--instruments', nargs='+', help='Trading instruments')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    
    # Override with command line arguments
    if args.start_date:
        config.backtest.start_date = args.start_date
    if args.end_date:
        config.backtest.end_date = args.end_date
    if args.instruments:
        config.trading.instruments = args.instruments
        
    # Run system
    if args.mode == 'backtest':
        run_backtest(config)
    elif args.mode == 'live':
        run_live_trading(config)
    elif args.mode == 'paper':
        run_paper_trading(config)
    else:
        logger.error(f"Unknown mode: {args.mode}")


def run_backtest(config):
    """Run backtest."""
    logger.info("Starting backtest...")
    
    # Initialize data components
    data_loader = DataLoader(config.data.data_path)
    data_normalizer = DataNormalizer()
    data_storage = DataStorage(config.data.storage_path)
    replay_engine = ReplayEngine(data_storage, config.data.replay)
    
    # Initialize strategies
    strategies = []
    
    if config.strategies.mean_reversion.enabled:
        strategy = MeanReversionStrategy(
            'MeanReversion',
            config.strategies.mean_reversion
        )
        strategies.append(strategy)
        
    if config.strategies.momentum.enabled:
        strategy = MomentumStrategy(
            'Momentum',
            config.strategies.momentum
        )
        strategies.append(strategy)
        
    if config.strategies.pair_trading.enabled:
        strategy = PairTradingStrategy(
            'PairTrading',
            config.strategies.pair_trading
        )
        strategies.append(strategy)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        transaction_cost_config=config.backtest.transaction_costs
    )
    
    # Add strategies
    for strategy in strategies:
        backtest_engine.add_strategy(strategy)
        
    # Load market data
    instruments = config.trading.instruments
    start_date = config.backtest.start_date
    end_date = config.backtest.end_date
    
    # Load data for each instrument
    market_data = {}
    for instrument in instruments:
        try:
            df = data_loader.load_ohlcv(instrument, '1m', start_date, end_date)
            df = data_normalizer.normalize(df, 'ohlcv')
            market_data[instrument] = df
            logger.info(f"Loaded {len(df)} bars for {instrument}")
        except Exception as e:
            logger.error(f"Failed to load data for {instrument}: {e}")
            
    # Set up replay engine
    replay_engine.instruments = instruments
    replay_engine.data_cache = market_data
    
    # Run backtest
    results = backtest_engine.run_backtest(replay_engine)
    
    # Save results
    results_file = f"results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(results_file).parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info(f"Backtest completed. Results saved to {results_file}")
    
    # Print summary
    print_backtest_summary(results)


def run_live_trading(config):
    """Run live trading."""
    logger.info("Starting live trading...")
    # Implementation would go here
    logger.warning("Live trading not implemented yet")


def run_paper_trading(config):
    """Run paper trading."""
    logger.info("Starting paper trading...")
    # Implementation would go here
    logger.warning("Paper trading not implemented yet")


def print_backtest_summary(results):
    """Print backtest summary."""
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    
    # Portfolio metrics
    portfolio = results.get('portfolio_metrics', {})
    print(f"Total Return: {portfolio.get('total_return', 0):.2%}")
    print(f"Annualized Return: {portfolio.get('annualized_return', 0):.2%}")
    print(f"Volatility: {portfolio.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {portfolio.get('max_drawdown', 0):.2%}")
    
    # Trade summary
    trades = results.get('trade_summary', {})
    print(f"\nTotal Trades: {trades.get('total_trades', 0)}")
    print(f"Total Commission: ${trades.get('total_commission', 0):.2f}")
    print(f"Total Slippage: ${trades.get('total_slippage', 0):.2f}")
    
    # Strategy metrics
    strategies = results.get('strategy_metrics', {})
    print(f"\nActive Strategies: {len(strategies)}")
    for name, metrics in strategies.items():
        print(f"  {name}: {metrics.get('total_positions', 0)} positions")
        
    print("="*60)


if __name__ == "__main__":
    main()