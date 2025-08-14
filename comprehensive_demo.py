#!/usr/bin/env python3
"""
Comprehensive Enhanced Forex Trading Bot Demo
Showcases all advanced features: ML, ICT, Math, Trade Evaluation, News Sentiment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from enhanced_forex_bot import EnhancedForexTradingBot
from ml_trading_engine import AdvancedMLTradingEngine
from ict_price_action import ICTPriceActionAnalyzer
from mathematical_algorithms import AdvancedMathematicalAlgorithms
from trade_evaluator import AdvancedTradeEvaluator
from advanced_indicators import AdvancedMathematicalIndicators
from news_sentiment_analyzer import NewsSentimentAnalyzer

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def demo_trade_evaluation():
    """Demonstrate advanced trade evaluation and mistake analysis"""
    print_section("Advanced Trade Evaluation & Mistake Analysis")
    
    # Initialize trade evaluator
    evaluator = AdvancedTradeEvaluator()
    
    # Create sample trades for demonstration
    sample_trades = [
        {
            'symbol': 'EUR/USD',
            'side': 'BUY',
            'entry_price': 1.1000,
            'exit_price': 1.1050,
            'entry_time': datetime.now() - timedelta(hours=4),
            'exit_time': datetime.now(),
            'amount': 10000,
            'stop_loss': 1.0950,
            'take_profit': 1.1100,
            'exit_reason': 'take_profit',
            'account_balance': 100000,
            'volume': 8000,
            'spread': 0.0002
        },
        {
            'symbol': 'GBP/USD',
            'side': 'SELL',
            'entry_price': 1.2500,
            'exit_price': 1.2450,
            'entry_time': datetime.now() - timedelta(hours=8),
            'exit_time': datetime.now() - timedelta(hours=2),
            'amount': 5000,
            'stop_loss': 1.2550,
            'take_profit': 1.2400,
            'exit_reason': 'stop_loss',
            'account_balance': 100000,
            'volume': 6000,
            'spread': 0.0003
        },
        {
            'symbol': 'USD/JPY',
            'side': 'BUY',
            'entry_price': 150.00,
            'exit_price': 149.50,
            'entry_time': datetime.now() - timedelta(hours=12),
            'exit_time': datetime.now() - timedelta(hours=1),
            'amount': 15000,
            'stop_loss': None,  # No stop loss - mistake!
            'take_profit': 151.00,
            'exit_reason': 'manual',
            'account_balance': 100000,
            'volume': 12000,
            'spread': 0.0004
        }
    ]
    
    print("Adding sample trades to evaluation system...")
    for trade in sample_trades:
        evaluator.add_trade(trade)
    
    print(f"Added {len(sample_trades)} trades for evaluation")
    
    # Evaluate each trade
    print("\nEvaluating individual trades...")
    for i, trade in enumerate(sample_trades):
        evaluation = evaluator.evaluate_trade(trade)
        print(f"\nTrade {i+1} ({trade['symbol']} {trade['side']}):")
        print(f"  Score: {evaluation['score']:.1f}/100")
        print(f"  Mistakes: {len(evaluation['mistakes'])}")
        print(f"  Recommendations: {len(evaluation['recommendations'])}")
        
        if evaluation['mistakes']:
            print("  Key Issues:")
            for mistake in evaluation['mistakes'][:2]:  # Show first 2 mistakes
                print(f"    - {mistake['description']} ({mistake['severity']})")
    
    # Analyze trading patterns
    print("\nAnalyzing trading patterns...")
    patterns = evaluator.analyze_trading_patterns()
    
    print(f"Total Trades: {patterns['total_trades']}")
    print(f"Win Rate: {patterns['win_rate']:.1%}")
    print(f"Profit Factor: {patterns['profit_factor']:.2f}")
    print(f"Average Duration: {patterns['avg_duration']:.1f} hours")
    
    if patterns['common_mistakes']:
        print("\nCommon Mistakes:")
        for mistake_type, count in patterns['common_mistakes'].items():
            print(f"  {mistake_type}: {count} occurrences")
    
    # Generate improvement plan
    print("\nGenerating improvement plan...")
    improvement_plan = evaluator.generate_improvement_plan(patterns)
    
    if improvement_plan['priority_actions']:
        print("\nPriority Actions:")
        for action in improvement_plan['priority_actions'][:3]:  # Show first 3
            print(f"  - {action['action']}: {action['description']}")
    
    # Get comprehensive report
    print("\nGenerating comprehensive report...")
    report = evaluator.get_comprehensive_report()
    print(f"Report generated successfully for {report['total_trades']} trades")

def demo_advanced_mathematical_indicators():
    """Demonstrate advanced mathematical indicators"""
    print_section("Advanced Mathematical Indicators")
    
    # Initialize indicators
    indicators = AdvancedMathematicalIndicators()
    
    # Create sample forex data
    print("Creating sample forex data...")
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.normal(1.1000, 0.005, len(dates)),
        'high': np.random.normal(1.1050, 0.005, len(dates)),
        'low': np.random.normal(1.0950, 0.005, len(dates)),
        'close': np.random.normal(1.1000, 0.005, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure high > low and close is between them
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.002, len(dates)))
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.002, len(dates)))
    
    print(f"Sample data created: {sample_data.shape}")
    
    # Calculate all adaptive indicators
    print("\nCalculating adaptive indicators...")
    result_df = indicators.get_all_adaptive_indicators(sample_data)
    
    # Display results
    adaptive_columns = [col for col in result_df.columns if 'adaptive' in col]
    print(f"\nGenerated {len(adaptive_columns)} adaptive indicators:")
    
    for col in adaptive_columns:
        print(f"  - {col}")
    
    # Show sample values for key indicators
    print("\nSample indicator values (last 5 rows):")
    key_indicators = ['adaptive_ama', 'adaptive_rsi', 'adaptive_macd', 'adaptive_bb_upper']
    available_indicators = [col for col in key_indicators if col in result_df.columns]
    
    if available_indicators:
        print(result_df[available_indicators].tail())
    
    # Demonstrate individual indicator calculations
    print("\nDemonstrating individual indicator calculations...")
    
    # Adaptive Moving Average
    ama = indicators.calculate_adaptive_moving_average(sample_data['close'])
    print(f"Adaptive MA range: {ama.min():.6f} to {ama.max():.6f}")
    
    # Adaptive RSI
    adaptive_rsi = indicators.calculate_adaptive_rsi(sample_data['close'])
    print(f"Adaptive RSI range: {adaptive_rsi.min():.1f} to {adaptive_rsi.max():.1f}")
    
    # Adaptive Bollinger Bands
    bb_result = indicators.calculate_adaptive_bollinger_bands(sample_data['close'])
    if bb_result:
        print(f"Adaptive BB bandwidth range: {bb_result['bandwidth'].min():.6f} to {bb_result['bandwidth'].max():.6f}")

def demo_news_sentiment_analysis():
    """Demonstrate news and social media sentiment analysis"""
    print_section("News & Social Media Sentiment Analysis")
    
    # Initialize analyzer
    analyzer = NewsSentimentAnalyzer()
    
    # Fetch forex news
    print("Fetching forex news from multiple sources...")
    news_data = analyzer.fetch_forex_news(['EUR', 'GBP', 'USD', 'JPY'])
    
    if not news_data.empty:
        print(f"Fetched {len(news_data)} news articles")
        print("\nSample news:")
        for _, news in news_data.head(3).iterrows():
            print(f"  {news['source']}: {news['title'][:60]}...")
    
    # Analyze news sentiment
    print("\nAnalyzing news sentiment...")
    news_sentiment = analyzer.analyze_news_sentiment(news_data)
    
    if not news_sentiment.empty:
        print(f"Analyzed sentiment for {len(news_sentiment)} articles")
        print("\nSentiment breakdown:")
        sentiment_counts = news_sentiment['sentiment_category'].value_counts()
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
    
    # Fetch social media sentiment
    print("\nFetching social media sentiment...")
    social_sentiment = analyzer.fetch_social_media_sentiment(['EUR', 'GBP', 'USD'])
    
    if not social_sentiment.empty:
        print(f"Fetched sentiment from {len(social_sentiment)} social media posts")
        print("\nSocial media sources:")
        source_counts = social_sentiment['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    
    # Calculate composite sentiment
    print("\nCalculating composite sentiment scores...")
    composite_sentiment = analyzer.calculate_composite_sentiment(news_sentiment, social_sentiment)
    
    if composite_sentiment:
        print(f"Composite sentiment calculated for {len(composite_sentiment)} currencies:")
        for currency, data in composite_sentiment.items():
            print(f"  {currency}: {data['sentiment_category']} (Score: {data['composite_score']:.3f}, Confidence: {data['confidence']:.3f})")
    
    # Generate trading signals
    print("\nGenerating trading signals based on sentiment...")
    trading_signals = analyzer.generate_trading_signals(composite_sentiment)
    
    if trading_signals:
        print(f"Generated signals for {len(trading_signals)} currencies:")
        for currency, signal in trading_signals.items():
            print(f"  {currency}: {signal['action']} - {signal['sentiment']} (Strength: {signal['strength']:.3f}, Risk: {signal['risk_level']})")
    
    # Get sentiment summary
    print("\nGenerating comprehensive sentiment summary...")
    summary = analyzer.get_sentiment_summary(['EUR', 'GBP', 'USD'])
    
    print(f"Overall Market Sentiment: {summary['overall_market_sentiment']}")
    print(f"Risk Assessment: {summary['risk_assessment']}")
    print(f"Top News Articles: {len(summary['top_news'])}")
    print(f"Trending Topics: {len(summary['trending_topics'])}")

def demo_enhanced_bot_integration():
    """Demonstrate enhanced bot with all features integrated"""
    print_section("Enhanced Bot Integration Demo")
    
    # Initialize enhanced bot
    print("Initializing enhanced forex trading bot...")
    bot_config = {
        'live_trading_mode': False,
        'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
        'timeframe': '1h',
        'max_positions': 3,
        'stop_loss_pips': 50,
        'take_profit_pips': 100,
        'max_daily_loss': 0.02,
        'max_portfolio_risk': 0.05,
        'risk_per_trade': 0.02
    }
    
    bot = EnhancedForexTradingBot(bot_config)
    
    # Get comprehensive analysis for a symbol
    print("\nGetting comprehensive analysis for EUR/USD...")
    analysis = bot.get_comprehensive_analysis('EUR/USD')
    
    if analysis:
        print("Analysis components:")
        for component, data in analysis.items():
            if isinstance(data, dict) and 'summary' in data:
                print(f"  {component}: {data['summary']}")
            else:
                print(f"  {component}: Data available")
    
    # Test position sizing with ML
    print("\nTesting ML-based position sizing...")
    try:
        position_size = bot.calculate_position_size(
            signal='BUY',
            confidence=0.8,
            account_balance=100000,
            market_volatility=0.02
        )
        print(f"ML-calculated position size: ${position_size:,.2f}")
    except Exception as e:
        print(f"Position sizing test: {e}")
    
    # Get performance summary
    print("\nGetting bot performance summary...")
    performance = bot.get_performance_summary()
    
    if performance:
        print("Performance metrics:")
        for metric, value in performance.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value}")
            else:
                print(f"  {metric}: {type(value).__name__}")

def demo_mathematical_algorithms():
    """Demonstrate advanced mathematical algorithms"""
    print_section("Advanced Mathematical Algorithms")
    
    # Initialize algorithms
    algorithms = AdvancedMathematicalAlgorithms()
    
    # Create sample data
    print("Creating sample price data...")
    np.random.seed(42)
    n_points = 1000
    prices = np.cumsum(np.random.normal(0, 0.01, n_points)) + 100
    returns = np.diff(prices) / prices[:-1]
    
    print(f"Generated {n_points} price points")
    
    # Advanced volatility calculations
    print("\nCalculating advanced volatility measures...")
    
    # GARCH volatility
    try:
        garch_vol = algorithms.calculate_advanced_volatility(returns, method='garch')
        print(f"GARCH Volatility: {garch_vol:.6f}")
    except Exception as e:
        print(f"GARCH calculation: {e}")
    
    # EWMA volatility
    try:
        ewma_vol = algorithms.calculate_advanced_volatility(returns, method='ewma')
        print(f"EWMA Volatility: {ewma_vol:.6f}")
    except Exception as e:
        print(f"EWMA calculation: {e}")
    
    # Realized volatility
    try:
        realized_vol = algorithms.calculate_advanced_volatility(returns, method='realized')
        print(f"Realized Volatility: {realized_vol:.6f}")
    except Exception as e:
        print(f"Realized volatility calculation: {e}")
    
    # Trend detection using Kalman filter
    print("\nDetecting trends using Kalman filter...")
    try:
        trend_data = algorithms.detect_trend_using_kalman(prices)
        print(f"Kalman trend detection completed: {len(trend_data)} trend points")
    except Exception as e:
        print(f"Kalman trend detection: {e}")
    
    # Market regime detection
    print("\nDetecting market regimes...")
    try:
        regime_data = algorithms.detect_market_regime(prices, returns, method='gmm')
        print(f"Market regime detection completed: {len(regime_data)} regimes identified")
    except Exception as e:
        print(f"Market regime detection: {e}")
    
    # Hurst exponent for mean reversion
    print("\nCalculating Hurst exponent...")
    try:
        hurst_exp = algorithms.calculate_hurst_exponent(prices)
        print(f"Hurst Exponent: {hurst_exp:.4f}")
        if hurst_exp < 0.5:
            print("  Interpretation: Mean-reverting market")
        elif hurst_exp > 0.5:
            print("  Interpretation: Trending market")
        else:
            print("  Interpretation: Random walk")
    except Exception as e:
        print(f"Hurst exponent calculation: {e}")
    
    # Fractal dimension
    print("\nCalculating fractal dimension...")
    try:
        fractal_dim = algorithms.calculate_fractal_dimension(prices, method='box_counting')
        print(f"Fractal Dimension: {fractal_dim:.4f}")
    except Exception as e:
        print(f"Fractal dimension calculation: {e}")
    
    # Entropy measures
    print("\nCalculating entropy measures...")
    try:
        entropy_data = algorithms.calculate_entropy_measures(returns, method='sample_entropy')
        print(f"Sample Entropy: {entropy_data:.4f}")
    except Exception as e:
        print(f"Entropy calculation: {e}")

def demo_ict_price_action():
    """Demonstrate ICT and price action analysis"""
    print_section("ICT & Price Action Analysis")
    
    # Initialize ICT analyzer
    ict_analyzer = ICTPriceActionAnalyzer()
    
    # Create sample data with more realistic patterns
    print("Creating sample price data with ICT patterns...")
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    np.random.seed(42)
    
    # Generate more realistic price movements
    n_points = len(dates)
    base_price = 1.1000
    
    # Create trending periods with pullbacks
    trend_periods = []
    current_price = base_price
    
    for i in range(0, n_points, 100):  # New trend every 100 hours
        trend_length = np.random.randint(50, 150)
        trend_direction = np.random.choice([1, -1])
        trend_strength = np.random.uniform(0.001, 0.003)
        
        for j in range(min(trend_length, n_points - i)):
            if i + j < n_points:
                # Add some noise and pullbacks
                noise = np.random.normal(0, 0.0002)
                pullback = np.random.choice([0, 1], p=[0.8, 0.2]) * np.random.uniform(-0.0005, 0.0005)
                
                current_price += trend_direction * trend_strength + noise + pullback
                trend_periods.append(current_price)
    
    # Ensure we have enough data points
    while len(trend_periods) < n_points:
        trend_periods.extend(trend_periods[:n_points - len(trend_periods)])
    
    trend_periods = trend_periods[:n_points]
    
    # Create OHLC data
    sample_data = pd.DataFrame({
        'open': trend_periods,
        'high': [p + abs(np.random.normal(0, 0.0003)) for p in trend_periods],
        'low': [p - abs(np.random.normal(0, 0.0003)) for p in trend_periods],
        'close': trend_periods,
        'volume': np.random.randint(1000, 10000, n_points)
    }, index=dates)
    
    # Ensure high > low
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.0002))
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.0002))
    
    print(f"Sample data created: {sample_data.shape}")
    
    # Analyze ICT levels
    print("\nAnalyzing ICT levels...")
    ict_analysis = ict_analyzer.analyze_ict_levels(sample_data)
    
    if ict_analysis:
        print("ICT Analysis Results:")
        for level_type, levels in ict_analysis.items():
            if isinstance(levels, list) and levels:
                print(f"  {level_type}: {len(levels)} levels identified")
            elif isinstance(levels, dict):
                print(f"  {level_type}: Analysis completed")
            else:
                print(f"  {level_type}: {levels}")
    
    # Analyze candlestick patterns
    print("\nAnalyzing candlestick patterns...")
    pattern_analysis = ict_analyzer.analyze_candlestick_patterns(sample_data)
    
    if pattern_analysis:
        print("Candlestick Pattern Analysis:")
        for pattern_type, patterns in pattern_analysis.items():
            if isinstance(patterns, list) and patterns:
                print(f"  {pattern_type}: {len(patterns)} patterns found")
            elif isinstance(patterns, dict):
                print(f"  {pattern_type}: Analysis completed")
            else:
                print(f"  {pattern_type}: {patterns}")
    
    # Analyze order flow
    print("\nAnalyzing order flow...")
    order_flow_analysis = ict_analyzer.analyze_order_flow(sample_data)
    
    if order_flow_analysis:
        print("Order Flow Analysis:")
        for flow_type, data in order_flow_analysis.items():
            if isinstance(data, (int, float)):
                print(f"  {flow_type}: {data:.4f}")
            else:
                print(f"  {flow_type}: {type(data).__name__}")
    
    # Get comprehensive analysis
    print("\nGetting comprehensive ICT analysis...")
    comprehensive_analysis = ict_analyzer.get_comprehensive_analysis(sample_data)
    
    if comprehensive_analysis:
        print("Comprehensive Analysis Summary:")
        for component, data in comprehensive_analysis.items():
            if isinstance(data, dict):
                summary_keys = list(data.keys())[:3]  # Show first 3 keys
                print(f"  {component}: {summary_keys}")
            else:
                print(f"  {component}: {type(data).__name__}")

def main():
    """Main demonstration function"""
    print_header("COMPREHENSIVE ENHANCED FOREX TRADING BOT DEMO")
    
    print("This demo showcases all the advanced features of the enhanced forex trading bot:")
    print("• Advanced Trade Evaluation & Mistake Analysis")
    print("• Advanced Mathematical Indicators")
    print("• News & Social Media Sentiment Analysis")
    print("• ICT & Price Action Analysis")
    print("• Advanced Mathematical Algorithms")
    print("• Enhanced Bot Integration")
    
    try:
        # Run all demonstrations
        demo_trade_evaluation()
        demo_advanced_mathematical_indicators()
        demo_news_sentiment_analysis()
        demo_mathematical_algorithms()
        demo_ict_price_action()
        demo_enhanced_bot_integration()
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("All advanced features have been demonstrated successfully!")
        
        print("\nNext Steps:")
        print("1. Review the generated logs and analysis results")
        print("2. Customize configurations for your trading strategy")
        print("3. Integrate with live exchange APIs for real trading")
        print("4. Set up automated trading with proper risk management")
        print("5. Monitor performance and continuously improve the system")
        
        print("\nImportant Notes:")
        print("• This demo uses simulated data - replace with real APIs for live trading")
        print("• Always test thoroughly in paper trading mode first")
        print("• Implement proper risk management and position sizing")
        print("• Monitor system performance and adjust parameters as needed")
        print("• Keep backups of configurations and trading data")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()