#!/usr/bin/env python3
"""
Enhanced Forex Trading Bot Demo
Showcases advanced ML, ICT analysis, mathematical algorithms, and live trading capabilities
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

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def print_section(title):
    """Print formatted section"""
    print(f"\n📊 {title}")
    print("-" * 60)

def demo_ml_trading_engine():
    """Demonstrate ML Trading Engine capabilities"""
    print_section("Machine Learning Trading Engine")
    
    try:
        # Initialize ML engine
        ml_engine = AdvancedMLTradingEngine()
        print("✓ ML Engine initialized successfully")
        
        # Generate sample data
        print("\nGenerating sample training data...")
        sample_data = generate_sample_forex_data('EUR/USD', days=30, timeframe='1h')
        
        # Extract advanced features
        print("Extracting advanced features...")
        df_with_features = ml_engine.extract_advanced_features(sample_data)
        print(f"✓ Features extracted: {len([col for col in df_with_features.columns if 'feature' in col.lower() or 'indicator' in col.lower()])} technical indicators")
        
        # Prepare features for ML
        print("Preparing features for ML models...")
        X, y, feature_names = ml_engine.prepare_features(df_with_features)
        
        if X is not None and len(X) > 0:
            print(f"✓ Feature matrix prepared: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Train models
            print("\nTraining ML models...")
            ml_engine.train_models(X, y, feature_names)
            
            # Get trading signal
            print("\nGetting ML trading signal...")
            signal, confidence = ml_engine.get_trading_signal(sample_data)
            print(f"✓ ML Signal: {signal} (Confidence: {confidence:.1%})")
            
            # Calculate position size
            print("\nCalculating risk-adjusted position size...")
            position_size, actual_risk, risk_percentage = ml_engine.calculate_risk_adjusted_position_size(
                signal, confidence, 10000, 0.001, 0.02
            )
            print(f"✓ Position Size: {position_size:.3f} lots")
            print(f"✓ Actual Risk: ${actual_risk:.2f}")
            print(f"✓ Risk Percentage: {risk_percentage:.1%}")
            
            # Get model performance
            performance = ml_engine.get_model_performance_summary()
            print(f"\n✓ Model Performance Summary:")
            for model_name, metrics in performance.get('models', {}).items():
                print(f"  {model_name}: CV Score = {metrics.get('cv_score', 0):.3f} ± {metrics.get('cv_std', 0):.3f}")
            
        else:
            print("⚠️ Insufficient data for ML analysis")
            
    except Exception as e:
        print(f"❌ Error in ML demo: {e}")

def demo_ict_price_action():
    """Demonstrate ICT and Price Action Analysis"""
    print_section("ICT and Price Action Analysis")
    
    try:
        # Initialize ICT analyzer
        ict_analyzer = ICTPriceActionAnalyzer()
        print("✓ ICT Analyzer initialized successfully")
        
        # Generate sample data
        print("\nGenerating sample market data...")
        sample_data = generate_sample_forex_data('EUR/USD', days=60, timeframe='4h')
        
        # Run comprehensive ICT analysis
        print("Running comprehensive ICT analysis...")
        ict_analysis = ict_analyzer.get_comprehensive_analysis(sample_data)
        
        if ict_analysis:
            print("✓ ICT Analysis completed successfully")
            
            # Display ICT levels
            ict_levels = ict_analysis.get('ict_levels', {})
            print(f"\n📈 ICT Levels Found:")
            for level_type, count in ict_levels.items():
                if isinstance(count, dict):
                    for sub_type, sub_count in count.items():
                        if sub_count > 0:
                            print(f"  {level_type} - {sub_type}: {sub_count}")
                elif count > 0:
                    print(f"  {level_type}: {count}")
            
            # Display candlestick patterns
            patterns = ict_analysis.get('candlestick_patterns', {})
            print(f"\n🕯️ Candlestick Patterns:")
            print(f"  Total Patterns: {patterns.get('total_patterns', 0):.1f}")
            
            pattern_dist = patterns.get('pattern_distribution', {})
            for pattern, count in pattern_dist.items():
                if count > 0:
                    print(f"  {pattern}: {count}")
            
            # Display order flow analysis
            order_flow = ict_analysis.get('order_flow', {})
            print(f"\n📊 Order Flow Analysis:")
            print(f"  Volume Trend: {order_flow.get('volume_trend', 0):.2f}")
            print(f"  Order Flow Imbalance: {order_flow.get('order_flow_imbalance', 0):.6f}")
            print(f"  Market Efficiency: {order_flow.get('market_efficiency', 0):.3f}")
            
            # Display time analysis
            time_analysis = ict_analysis.get('time_analysis', {})
            print(f"\n⏰ Time Analysis:")
            for session, count in time_analysis.items():
                if count > 0:
                    print(f"  {session}: {count} periods")
            
        else:
            print("⚠️ ICT analysis failed")
            
    except Exception as e:
        print(f"❌ Error in ICT demo: {e}")

def demo_mathematical_algorithms():
    """Demonstrate Advanced Mathematical Algorithms"""
    print_section("Advanced Mathematical Algorithms")
    
    try:
        # Initialize math algorithms
        math_algo = AdvancedMathematicalAlgorithms()
        print("✓ Mathematical Algorithms initialized successfully")
        
        # Generate sample data
        print("\nGenerating sample market data...")
        sample_data = generate_sample_forex_data('EUR/USD', days=90, timeframe='1d')
        returns = sample_data['close'].pct_change().dropna()
        
        # 1. Advanced Volatility Analysis
        print("\n📊 Advanced Volatility Analysis:")
        volatility_analysis = math_algo.calculate_advanced_volatility(returns, method='garch')
        
        if 'garch_volatility' in volatility_analysis:
            garch_vol = volatility_analysis['garch_volatility']
            print(f"  GARCH Volatility: {garch_vol.iloc[-1]:.6f}")
            print(f"  Volatility Range: {garch_vol.min():.6f} - {garch_vol.max():.6f}")
            
            if 'garch_params' in volatility_analysis:
                params = volatility_analysis['garch_params']
                print(f"  GARCH Parameters: ω={params.get('omega', 0):.8f}, α={params.get('alpha', 0):.3f}, β={params.get('beta', 0):.3f}")
                print(f"  Persistence: {volatility_analysis.get('persistence', 0):.3f}")
        
        # 2. Kalman Filter Trend Detection
        print("\n🎯 Kalman Filter Trend Detection:")
        kalman_trend = math_algo.detect_trend_using_kalman(sample_data['close'])
        
        if 'trend_direction' in kalman_trend:
            trend_dir = kalman_trend['trend_direction'].iloc[-1]
            trend_strength = kalman_trend['trend_strength'].iloc[-1]
            trend_confidence = kalman_trend['trend_confidence'].iloc[-1]
            
            trend_name = "BULLISH" if trend_dir > 0 else "BEARISH" if trend_dir < 0 else "NEUTRAL"
            print(f"  Trend Direction: {trend_name}")
            print(f"  Trend Strength: {trend_strength:.6f}")
            print(f"  Trend Confidence: {trend_confidence:.3f}")
        
        # 3. Market Regime Detection
        print("\n🔄 Market Regime Detection:")
        regime_analysis = math_algo.detect_market_regime(sample_data['close'], returns)
        
        if 'regimes' in regime_analysis:
            regimes = regime_analysis['regimes']
            current_regime = regimes.iloc[-1] if len(regimes) > 0 else 'unknown'
            print(f"  Current Regime: {current_regime}")
            
            regime_stats = regime_analysis.get('regime_statistics', {})
            for regime_name, stats in regime_stats.items():
                print(f"  {regime_name}: Return={stats.get('mean_return', 0):.4f}, Vol={stats.get('volatility', 0):.4f}, Sharpe={stats.get('sharpe_ratio', 0):.3f}")
        
        # 4. Hurst Exponent Analysis
        print("\n📏 Hurst Exponent Analysis:")
        hurst_analysis = math_algo.calculate_hurst_exponent(sample_data['close'])
        
        if 'hurst_exponent' in hurst_analysis:
            hurst_exp = hurst_analysis['hurst_exponent']
            regime = hurst_analysis['regime']
            print(f"  Hurst Exponent: {hurst_exp:.3f}")
            print(f"  Market Behavior: {regime.upper()}")
            print(f"  R²: {hurst_analysis.get('r_squared', 0):.3f}")
        
        # 5. Fractal Dimension Analysis
        print("\n🔷 Fractal Dimension Analysis:")
        fractal_analysis = math_algo.calculate_fractal_dimension(sample_data['close'])
        
        if 'fractal_dimension' in fractal_analysis:
            fractal_dim = fractal_analysis['fractal_dimension']
            complexity = fractal_analysis['complexity']
            print(f"  Fractal Dimension: {fractal_dim:.3f}")
            print(f"  Market Complexity: {complexity.upper()}")
            print(f"  R²: {fractal_analysis.get('r_squared', 0):.3f}")
        
        # 6. Entropy Analysis
        print("\n🎲 Entropy Analysis:")
        entropy_analysis = math_algo.calculate_entropy_measures(returns)
        
        for entropy_type, value in entropy_analysis.items():
            if isinstance(value, (int, float)) and not np.isinf(value):
                print(f"  {entropy_type}: {value:.3f}")
        
        # 7. Generate Trading Signals
        print("\n🚦 Mathematical Trading Signals:")
        analysis_results = {
            'kalman_trend': kalman_trend,
            'volatility': volatility_analysis,
            'regime': regime_analysis,
            'hurst': hurst_analysis
        }
        
        math_signals = math_algo.generate_trading_signals(analysis_results)
        
        for signal_type, value in math_signals.items():
            if signal_type != 'reasoning':
                print(f"  {signal_type}: {value}")
        
        if 'reasoning' in math_signals:
            print(f"  Reasoning: {', '.join(math_signals['reasoning'])}")
        
    except Exception as e:
        print(f"❌ Error in mathematical algorithms demo: {e}")

def demo_enhanced_bot():
    """Demonstrate the Enhanced Forex Trading Bot"""
    print_section("Enhanced Forex Trading Bot Integration")
    
    try:
        # Initialize enhanced bot
        print("Initializing Enhanced Forex Trading Bot...")
        bot = EnhancedForexTradingBot()
        print("✓ Enhanced Bot initialized successfully")
        
        # Get comprehensive analysis
        print("\nRunning comprehensive analysis for EUR/USD...")
        analysis = bot.get_comprehensive_analysis('EUR/USD')
        
        if analysis:
            print("✓ Comprehensive analysis completed")
            
            # Display combined signal
            combined_signal = analysis.get('combined_signal', {})
            print(f"\n🎯 Combined Trading Signal:")
            print(f"  Signal: {combined_signal.get('signal', 'N/A')}")
            print(f"  Confidence: {combined_signal.get('confidence', 0):.1%}")
            print(f"  Signal Strength: {combined_signal.get('signal_strength', 0):.1%}")
            print(f"  Bullish Signals: {combined_signal.get('bullish_signals', 0)}")
            print(f"  Bearish Signals: {combined_signal.get('bearish_signals', 0)}")
            print(f"  Neutral Signals: {combined_signal.get('neutral_signals', 0)}")
            print(f"  Total Signals: {combined_signal.get('total_signals', 0)}")
            
            # Display ML signal
            ml_signal = analysis.get('ml_signal', {})
            print(f"\n🤖 ML Engine Signal:")
            print(f"  Signal: {ml_signal.get('signal', 'N/A')}")
            print(f"  Confidence: {ml_signal.get('confidence', 0):.1%}")
            
            # Display mathematical signals
            math_signals = analysis.get('mathematical_signals', {})
            print(f"\n🧮 Mathematical Signals:")
            for signal_type, value in math_signals.items():
                if signal_type != 'reasoning':
                    print(f"  {signal_type}: {value}")
            
            # Display ICT analysis summary
            ict_analysis = analysis.get('ict_analysis', {})
            if ict_analysis:
                print(f"\n📊 ICT Analysis Summary:")
                ict_levels = ict_analysis.get('ict_levels', {})
                for level_type, count in ict_levels.items():
                    if isinstance(count, dict):
                        total_count = sum(count.values())
                        if total_count > 0:
                            print(f"  {level_type}: {total_count} levels")
                    elif count > 0:
                        print(f"  {level_type}: {count} levels")
            
            # Display volatility analysis
            volatility = analysis.get('volatility', {})
            if 'garch_volatility' in volatility:
                garch_vol = volatility['garch_volatility']
                print(f"\n📈 Volatility Analysis:")
                print(f"  Current GARCH Volatility: {garch_vol.iloc[-1]:.6f}")
                print(f"  Average Volatility: {garch_vol.mean():.6f}")
            
            # Display market regime
            regime = analysis.get('regime', {})
            if 'regimes' in regime:
                regimes = regime['regimes']
                current_regime = regimes.iloc[-1] if len(regimes) > 0 else 'unknown'
                print(f"\n🔄 Market Regime:")
                print(f"  Current Regime: {current_regime}")
            
            # Display performance summary
            print(f"\n📊 Performance Summary:")
            performance = bot.get_performance_summary()
            for key, value in performance.items():
                if key != 'ml_engine_status':
                    print(f"  {key}: {value}")
            
        else:
            print("⚠️ Comprehensive analysis failed")
        
        # Test position sizing
        print(f"\n💰 Position Sizing Test:")
        account_balance = 10000
        market_volatility = 0.001
        signal = combined_signal.get('signal', 'HOLD')
        confidence = combined_signal.get('confidence', 0.5)
        
        position_size, actual_risk, risk_percentage = bot.calculate_position_size(
            signal, confidence, account_balance, market_volatility
        )
        
        print(f"  Signal: {signal}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Position Size: {position_size:.3f} lots")
        print(f"  Actual Risk: ${actual_risk:.2f}")
        print(f"  Risk Percentage: {risk_percentage:.1%}")
        
    except Exception as e:
        print(f"❌ Error in enhanced bot demo: {e}")

def demo_live_trading_simulation():
    """Demonstrate live trading simulation"""
    print_section("Live Trading Simulation")
    
    try:
        # Initialize bot with live trading configuration
        config = {
            'live_trading_mode': False,  # Paper trading for demo
            'paper_trading': True,
            'symbols': ['EUR/USD', 'GBP/USD'],
            'timeframe': '1h',
            'max_positions': 2,
            'stop_loss_pips': 50,
            'take_profit_pips': 100
        }
        
        print("Initializing bot for live trading simulation...")
        bot = EnhancedForexTradingBot(config)
        print("✓ Bot initialized for simulation")
        
        # Run trading cycle
        print("\nRunning trading cycle...")
        bot.run_trading_cycle()
        
        # Display results
        print(f"\n📊 Trading Cycle Results:")
        performance = bot.get_performance_summary()
        for key, value in performance.items():
            if key != 'ml_engine_status':
                print(f"  {key}: {value}")
        
        # Display trades
        if bot.trades:
            print(f"\n💼 Executed Trades:")
            for i, trade in enumerate(bot.trades[-3:], 1):  # Show last 3 trades
                print(f"  Trade {i}: {trade['side']} {trade['amount']} {trade['symbol']} @ {trade['entry_price']:.5f}")
                print(f"    Stop Loss: {trade['stop_loss']:.5f}, Take Profit: {trade['take_profit']:.5f}")
                print(f"    Confidence: {trade['confidence']:.1%}, Type: {trade['type']}")
        
        # Start bot for continuous trading
        print(f"\n🚀 Starting continuous trading simulation...")
        bot.start_bot(interval_minutes=1)  # 1 minute intervals for demo
        
        # Let it run for a few cycles
        time.sleep(10)
        
        # Stop bot
        print("Stopping trading simulation...")
        bot.stop_bot()
        
        # Final performance summary
        print(f"\n📈 Final Performance Summary:")
        final_performance = bot.get_performance_summary()
        for key, value in final_performance.items():
            if key != 'ml_engine_status':
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error in live trading simulation: {e}")

def generate_sample_forex_data(symbol, days=30, timeframe='1h'):
    """Generate realistic sample forex data"""
    try:
        # Calculate number of periods
        if timeframe == '1h':
            periods = days * 24
        elif timeframe == '4h':
            periods = days * 6
        elif timeframe == '1d':
            periods = days
        else:
            periods = days * 24
        
        # Generate dates
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq=timeframe)
        
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 1000)
        base_price = 1.1000 if 'USD' in symbol else 110.0
        
        data = []
        for i in range(periods):
            # Generate price changes with trend and mean reversion
            trend = 0.0001 * np.sin(i / 10)  # Cyclical trend
            random_change = np.random.normal(0, 0.0005)  # Random component
            price_change = trend + random_change
            
            base_price *= (1 + price_change)
            
            # Generate OHLCV
            volatility = abs(np.random.normal(0, 0.0003))
            open_price = base_price * (1 + np.random.normal(0, 0.0001))
            high_price = max(open_price, base_price) + volatility
            low_price = min(open_price, base_price) - volatility
            close_price = base_price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': dates[i] if i < len(dates) else dates[-1],
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
        print(f"Error generating sample data: {e}")
        return pd.DataFrame()

def main():
    """Main demo function"""
    print_header("ENHANCED FOREX TRADING BOT DEMO")
    print("This demo showcases the bot's advanced capabilities with:")
    print("✅ Machine Learning Trading Engine")
    print("✅ ICT and Price Action Analysis")
    print("✅ Advanced Mathematical Algorithms")
    print("✅ Live Trading Simulation")
    print("✅ Risk Management and Position Sizing")
    print("✅ Self-Learning Capabilities")
    
    try:
        # Run all demos
        demo_ml_trading_engine()
        demo_ict_price_action()
        demo_mathematical_algorithms()
        demo_enhanced_bot()
        demo_live_trading_simulation()
        
        print_header("DEMO COMPLETED SUCCESSFULLY!")
        print("\n🎉 What you've experienced:")
        print("✅ Advanced ML models with ensemble learning")
        print("✅ ICT analysis with Fair Value Gaps, Order Blocks, and Liquidity Levels")
        print("✅ Advanced candlestick pattern recognition")
        print("✅ Mathematical algorithms including GARCH, Kalman filters, and Hurst exponent")
        print("✅ Market regime detection and fractal analysis")
        print("✅ Comprehensive signal combination and risk management")
        print("✅ Live trading simulation with paper trading")
        
        print("\n🚀 Next steps:")
        print("1. Install all dependencies: pip install -r requirements.txt")
        print("2. Configure your exchange API keys in .env file")
        print("3. Test with paper trading: python enhanced_forex_bot.py")
        print("4. Enable live trading by setting live_trading_mode=True")
        print("5. Monitor performance and adjust parameters")
        
        print("\n⚠️  Important Notes:")
        print("- This is a sophisticated trading system - test thoroughly before live trading")
        print("- Start with small position sizes and gradually increase")
        print("- Monitor ML model performance and retrain as needed")
        print("- Always use proper risk management")
        print("- Consider market conditions and adjust strategies accordingly")
        
        print("\n🔧 Customization Options:")
        print("- Modify ML model parameters in ml_config")
        print("- Adjust ICT analysis settings in ict_config")
        print("- Fine-tune mathematical algorithms in math_config")
        print("- Add custom indicators and strategies")
        print("- Implement additional risk management rules")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be due to missing dependencies or configuration issues.")
        print("Please check the installation instructions in the README.")

if __name__ == "__main__":
    main()