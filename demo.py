#!/usr/bin/env python3
"""
Forex Trading Bot Demo
This script demonstrates the bot's capabilities with sample data and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from forex_bot import ForexTradingBot
from backtester import ForexBacktester

def create_sample_forex_data(symbol='EUR/USD', days=30, timeframe='1h'):
    """
    Create realistic sample forex data for demonstration
    
    Args:
        symbol: Currency pair symbol
        days: Number of days of data to generate
        timeframe: Timeframe for data (1h, 4h, 1d)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Generating sample data for {symbol} over {days} days...")
    
    # Calculate number of periods based on timeframe
    if timeframe == '1h':
        periods = days * 24
    elif timeframe == '4h':
        periods = days * 6
    elif timeframe == '1d':
        periods = days
    else:
        periods = days * 24  # Default to hourly
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # Base price for EUR/USD
    base_price = 1.1000
    
    # Generate price series with realistic volatility
    returns = np.random.normal(0, 0.0005, periods)  # 5 pips average movement per period
    
    # Add some trend and mean reversion
    trend = np.linspace(0, 0.002, periods)  # Slight upward trend
    returns += trend
    
    # Generate prices
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i in range(periods):
        price = prices[i]
        
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.0003))
        
        open_price = price * (1 + np.random.normal(0, 0.0001))
        high_price = max(open_price, price) + volatility
        low_price = min(open_price, price) - volatility
        close_price = price
        
        # Volume (not critical for forex but good for completeness)
        volume = np.random.randint(1000, 10000)
        
        # Timestamp
        if timeframe == '1h':
            timestamp = datetime.now() - timedelta(hours=periods-i)
        elif timeframe == '4h':
            timestamp = datetime.now() - timedelta(hours=(periods-i)*4)
        else:
            timestamp = datetime.now() - timedelta(days=periods-i)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"Generated {len(df)} data points")
    print(f"Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
    
    return df

def demo_strategy_analysis():
    """Demonstrate strategy analysis capabilities"""
    print("\n" + "="*50)
    print("STRATEGY ANALYSIS DEMO")
    print("="*50)
    
    # Create sample data
    df = create_sample_forex_data('EUR/USD', days=7, timeframe='1h')
    
    # Initialize bot (demo mode)
    bot = ForexTradingBot()
    
    # Run analysis
    print("\nRunning strategy analysis...")
    analysis = bot.run_strategy_analysis('EUR/USD')
    
    if analysis:
        print(f"\nAnalysis Results for EUR/USD:")
        print(f"Current Price: {analysis['current_price']:.5f}")
        print(f"Combined Signal: {analysis['signals']['combined']['signal']}")
        print(f"Signal Confidence: {analysis['signals']['combined']['confidence']:.1%}")
        
        print(f"\nIndividual Strategy Signals:")
        for strategy, data in analysis['signals'].items():
            if strategy != 'combined':
                print(f"  {strategy.upper()}: {data['signal']} (Confidence: {data['confidence']:.1%})")
        
        print(f"\nTechnical Indicators:")
        print(f"  RSI: {analysis['indicators']['rsi']:.1f}")
        print(f"  MACD: {analysis['indicators']['macd']:.6f}")
        print(f"  SMA 20: {analysis['indicators']['sma_20']:.5f}")
        print(f"  SMA 50: {analysis['indicators']['sma_50']:.5f}")
    else:
        print("Analysis failed - this is expected in demo mode without real exchange connection")

def demo_backtesting():
    """Demonstrate backtesting capabilities"""
    print("\n" + "="*50)
    print("BACKTESTING DEMO")
    print("="*50)
    
    # Create sample data
    df = create_sample_forex_data('EUR/USD', days=60, timeframe='4h')
    
    # Save sample data for backtester
    sample_file = 'sample_forex_data.csv'
    df.to_csv(sample_file)
    print(f"\nSaved sample data to {sample_file}")
    
    # Initialize backtester
    backtester = ForexBacktester(initial_balance=10000)
    
    # Load and process data
    print("\nProcessing data...")
    df = backtester.load_data(sample_file)
    df = backtester.calculate_indicators(df)
    df = backtester.generate_signals(df)
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run_backtest(
        df, 
        position_size=0.01, 
        stop_loss_pips=50, 
        take_profit_pips=100
    )
    
    if results:
        print(f"\nBacktest Results:")
        print(f"  Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"  Final Balance: ${results['final_balance']:,.2f}")
        print(f"  Total Return: {results['total_return']:.2%}")
        print(f"  Total P&L: ${results['total_pnl']:,.2f}")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        # Plot results
        print("\nGenerating performance charts...")
        try:
            backtester.plot_results(results, save_path='backtest_results.png')
            print("Charts saved to 'backtest_results.png'")
        except Exception as e:
            print(f"Chart generation failed: {e}")
        
        # Save results
        backtester.save_results(results, 'backtest_results.json')
        
    else:
        print("Backtest failed")

def demo_risk_management():
    """Demonstrate risk management features"""
    print("\n" + "="*50)
    print("RISK MANAGEMENT DEMO")
    print("="*50)
    
    # Initialize bot
    bot = ForexTradingBot()
    
    # Demonstrate position sizing
    print("\nPosition Sizing Examples:")
    account_balances = [1000, 5000, 10000, 50000]
    
    for balance in account_balances:
        position_size = bot.calculate_position_size(balance, risk_per_trade=0.02)
        max_loss = balance * 0.02
        print(f"  Account: ${balance:>6,} → Position Size: {position_size:.3f} lots (Max Loss: ${max_loss:.2f})")
    
    # Risk parameters
    print(f"\nRisk Management Parameters:")
    print(f"  Max Daily Loss: {bot.max_daily_loss:.1%}")
    print(f"  Max Portfolio Risk: {bot.max_portfolio_risk:.1%}")
    print(f"  Stop Loss: {bot.stop_loss_pips} pips")
    print(f"  Take Profit: {bot.take_profit_pips} pips")
    print(f"  Max Positions: {bot.max_positions}")

def demo_trading_strategies():
    """Demonstrate different trading strategies"""
    print("\n" + "="*50)
    print("TRADING STRATEGIES DEMO")
    print("="*50)
    
    # Create sample data
    df = create_sample_forex_data('EUR/USD', days=30, timeframe='1h')
    
    # Initialize bot
    bot = ForexTradingBot()
    
    # Calculate indicators
    df = bot.calculate_indicators(df)
    
    # Test individual strategies
    strategies = [
        ('RSI Strategy', bot.rsi_strategy),
        ('MACD Strategy', bot.macd_strategy),
        ('Moving Average Strategy', bot.moving_average_strategy),
        ('Bollinger Bands Strategy', bot.bollinger_bands_strategy)
    ]
    
    print("\nStrategy Performance Analysis:")
    print("Strategy".ljust(25) + "Signal".ljust(10) + "Confidence".ljust(15) + "Description")
    print("-" * 70)
    
    for name, strategy_func in strategies:
        signal, confidence = strategy_func(df)
        
        # Generate description
        if signal == 'BUY':
            desc = "Bullish signal - Consider long position"
        elif signal == 'SELL':
            desc = "Bearish signal - Consider short position"
        else:
            desc = "No clear signal - Wait for better opportunity"
        
        print(f"{name.ljust(25)}{signal.ljust(10)}{f'{confidence:.1%}'.ljust(15)}{desc}")
    
    # Combined signal
    combined_signal, combined_conf = bot.get_combined_signal(df)
    print(f"\nCombined Signal: {combined_signal} (Confidence: {combined_conf:.1%})")
    
    if combined_signal != 'HOLD':
        print(f"Recommendation: {combined_signal} EUR/USD with {combined_conf:.1%} confidence")
    else:
        print("Recommendation: No trade at this time")

def main():
    """Main demo function"""
    print("🚀 FOREX TRADING BOT DEMO")
    print("=" * 50)
    print("This demo showcases the bot's capabilities with sample data.")
    print("Note: This is a demonstration - no real trading will occur.")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_strategy_analysis()
        demo_backtesting()
        demo_risk_management()
        demo_trading_strategies()
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nWhat you've seen:")
        print("✅ Strategy analysis with technical indicators")
        print("✅ Backtesting with performance metrics")
        print("✅ Risk management and position sizing")
        print("✅ Multiple trading strategies")
        print("✅ Sample data generation")
        
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure your exchange API keys in .env file")
        print("3. Run the web dashboard: python dashboard.py")
        print("4. Start live trading: python forex_bot.py")
        
        print("\n⚠️  Remember: Forex trading involves substantial risk!")
        print("Always test thoroughly before live trading.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be due to missing dependencies or configuration issues.")
        print("Please check the installation instructions in the README.")

if __name__ == "__main__":
    main()