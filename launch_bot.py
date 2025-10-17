#!/usr/bin/env python3
"""
🚀 ENHANCED LAUNCHER SCRIPT FOR ULTRA-ADVANCED FOREX TRADING BOT 🚀
Quantitative Analysis, Mathematical Pattern Recognition & AI-Powered Trading

Features:
- Advanced Quantitative Analysis with Mathematical Formulas
- Fourier Transform Pattern Recognition
- Wavelet Analysis for Market Microstructure
- Kalman Filter Price Prediction
- GARCH Volatility Modeling
- Monte Carlo Risk Simulation
- Advanced Statistical Arbitrage
- Machine Learning Ensemble Models
- Real-time Market Regime Detection
- Advanced Portfolio Optimization
"""

import os
import sys
import time
import subprocess
import threading
import json
from pathlib import Path
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'MetaTrader5', 'pandas', 'numpy', 'scikit-learn', 
        'tensorflow', 'ta', 'flask', 'flask-socketio', 'plotly',
        'scipy', 'statsmodels', 'arch', 'pywt', 'ccxt'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.lower().replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements_compatible.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_config():
    """Check if configuration files exist"""
    config_file = Path("bot_config.json")
    env_file = Path(".env")
    
    if not config_file.exists():
        print("❌ Configuration file 'bot_config.json' not found")
        return False
    
    if not env_file.exists():
        print("⚠️  Environment file '.env' not found")
        print("Creating default .env file...")
        
        # Create default .env file
        env_content = """# MetaTrader5 Configuration
MT5_LOGIN=your_login_here
MT5_PASSWORD=your_password_here
MT5_SERVER=your_server_here

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# Trading Bot Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Created default .env file")
        print("⚠️  Please update .env file with your actual credentials")
    
    print("✅ Configuration files are ready")
    return True

def start_dashboard():
    """Start the web dashboard"""
    try:
        print("🚀 Starting Trading Dashboard...")
        subprocess.run([sys.executable, "trading_dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def start_bot():
    """Start the trading bot"""
    try:
        print("🤖 Starting Trading Bot...")
        subprocess.run([sys.executable, "advanced_forex_trading_bot.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")

def show_quantitative_features():
    """Display quantitative analysis features"""
    print("\n" + "=" * 80)
    print("🧮 QUANTITATIVE ANALYSIS FEATURES")
    print("=" * 80)
    print("📊 Mathematical Pattern Recognition:")
    print("   • Fibonacci Retracements & Extensions")
    print("   • Elliott Wave Analysis")
    print("   • Harmonic Trading Patterns (Gartley, Butterfly)")
    print("   • Support & Resistance Detection")
    print("   • Trend Line Analysis")
    
    print("\n🔬 Advanced Mathematical Analysis:")
    print("   • Fourier Transform for Cyclical Patterns")
    print("   • Wavelet Analysis for Multi-Resolution")
    print("   • Kalman Filter for Price Prediction")
    print("   • GARCH Volatility Modeling")
    print("   • Monte Carlo Risk Simulation")
    print("   • Statistical Arbitrage Analysis")
    
    print("\n📈 Technical Indicators:")
    print("   • Hurst Exponent for Trend Persistence")
    print("   • Fractal Dimension Analysis")
    print("   • Jarque-Bera Normality Tests")
    print("   • Skewness & Kurtosis Analysis")
    print("   • Advanced Volatility Measures")
    
    print("\n🤖 Machine Learning Models:")
    print("   • Ensemble Learning (Random Forest, XGBoost, LightGBM)")
    print("   • Deep Learning (LSTM, Transformer)")
    print("   • Reinforcement Learning")
    print("   • Sentiment Analysis with NLP")
    print("   • Real-time Model Retraining")
    
    print("\n⚡ High-Frequency Trading:")
    print("   • Scalping Strategies")
    print("   • Grid Trading with Dynamic Levels")
    print("   • Statistical Arbitrage")
    print("   • Market Making Algorithms")
    print("   • Order Flow Analysis")
    
    print("=" * 80)

def main():
    """Main launcher function"""
    print("=" * 80)
    print("🚀 ULTRA-ADVANCED FOREX TRADING BOT LAUNCHER 🚀")
    print("Quantitative Analysis, Mathematical Pattern Recognition & AI-Powered Trading")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    if not check_config():
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("🎯 Choose an option:")
    print("1. Start Trading Bot (Console Mode)")
    print("2. Start Web Dashboard")
    print("3. Start Both (Bot + Dashboard)")
    print("4. Show Quantitative Features")
    print("5. Run Quantitative Analysis Demo")
    print("6. Exit")
    print("=" * 80)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n🤖 Starting Ultra-Advanced Trading Bot in console mode...")
                print("📊 Quantitative Analysis: ENABLED")
                print("🧮 Mathematical Patterns: ENABLED")
                print("🤖 AI Models: ENABLED")
                start_bot()
                break
                
            elif choice == "2":
                print("\n🌐 Starting Enhanced Web Dashboard...")
                print("📱 Open your browser and go to: http://localhost:8080")
                print("📊 Real-time Quantitative Analysis Dashboard")
                start_dashboard()
                break
                
            elif choice == "3":
                print("\n🚀 Starting both Trading Bot and Enhanced Dashboard...")
                print("📱 Open your browser and go to: http://localhost:8080")
                print("📊 Full Quantitative Analysis Suite")
                
                # Start dashboard in separate thread
                dashboard_thread = threading.Thread(target=start_dashboard)
                dashboard_thread.daemon = True
                dashboard_thread.start()
                
                # Start bot in main thread
                time.sleep(2)  # Give dashboard time to start
                start_bot()
                break
                
            elif choice == "4":
                show_quantitative_features()
                continue
                
            elif choice == "5":
                print("\n🧮 Running Quantitative Analysis Demo...")
                run_quantitative_demo()
                break
                
            elif choice == "6":
                print("\n👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Launcher stopped by user")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def run_quantitative_demo():
    """Run a demonstration of quantitative analysis features"""
    try:
        print("\n🧮 QUANTITATIVE ANALYSIS DEMONSTRATION")
        print("=" * 50)
        
        # Import the trading bot
        from advanced_forex_trading_bot import AdvancedForexBot
        
        # Initialize bot
        print("🔧 Initializing Quantitative Analysis Engine...")
        bot = AdvancedForexBot()
        
        # Demo symbols
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        for symbol in symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            # Get market data
            df = bot.get_market_data(symbol, "H1", 100)
            if df.empty:
                print(f"   ❌ No data available for {symbol}")
                continue
            
            # Add technical indicators with quantitative analysis
            df = bot.add_technical_indicators(df)
            
            # Detect mathematical patterns
            patterns = bot.detect_mathematical_patterns(symbol)
            if 'error' not in patterns:
                print(f"   ✅ Patterns detected: {len(patterns)} types")
                for pattern_type, pattern_data in patterns.items():
                    if isinstance(pattern_data, dict) and 'error' not in pattern_data:
                        print(f"      • {pattern_type}: {pattern_data.get('detected', False)}")
            
            # Quantitative risk analysis
            risk_analysis = bot.quantitative_risk_analysis(symbol)
            if 'error' not in risk_analysis:
                print(f"   📈 Risk Metrics:")
                print(f"      • Current Volatility: {risk_analysis.get('current_volatility', 0):.4f}")
                print(f"      • Sharpe Ratio: {risk_analysis.get('sharpe_ratio', 0):.4f}")
                print(f"      • VaR 95%: {risk_analysis.get('var_95', 0):.4f}")
                print(f"      • Max Drawdown: {risk_analysis.get('max_drawdown', 0):.4f}")
            
            # Show quantitative indicators
            if not df.empty:
                latest = df.iloc[-1]
                print(f"   🔬 Latest Quantitative Indicators:")
                if 'trend_strength' in latest and not pd.isna(latest['trend_strength']):
                    print(f"      • Trend Strength: {latest['trend_strength']:.4f}")
                if 'hurst_exponent' in latest and not pd.isna(latest['hurst_exponent']):
                    print(f"      • Hurst Exponent: {latest['hurst_exponent']:.4f}")
                if 'fractal_dimension' in latest and not pd.isna(latest['fractal_dimension']):
                    print(f"      • Fractal Dimension: {latest['fractal_dimension']:.4f}")
        
        print("\n✅ Quantitative Analysis Demo Completed!")
        print("🚀 Ready to start full trading system with advanced quantitative analysis!")
        
    except Exception as e:
        print(f"❌ Error in quantitative demo: {e}")
        print("💡 Make sure all dependencies are installed correctly")

if __name__ == "__main__":
    main()