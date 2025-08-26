#!/usr/bin/env python3
"""
Launcher script for Advanced Forex Trading Bot
This script provides an easy way to start the bot and dashboard
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'MetaTrader5', 'pandas', 'numpy', 'scikit-learn', 
        'tensorflow', 'ta', 'flask', 'flask-socketio', 'plotly'
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
        print("pip install -r requirements.txt")
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

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🚀 Advanced Forex Trading Bot Launcher")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    if not check_config():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎯 Choose an option:")
    print("1. Start Trading Bot (Console Mode)")
    print("2. Start Web Dashboard")
    print("3. Start Both (Bot + Dashboard)")
    print("4. Exit")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🤖 Starting Trading Bot in console mode...")
                start_bot()
                break
                
            elif choice == "2":
                print("\n🌐 Starting Web Dashboard...")
                print("📱 Open your browser and go to: http://localhost:8080")
                start_dashboard()
                break
                
            elif choice == "3":
                print("\n🚀 Starting both Trading Bot and Dashboard...")
                print("📱 Open your browser and go to: http://localhost:8080")
                
                # Start dashboard in separate thread
                dashboard_thread = threading.Thread(target=start_dashboard)
                dashboard_thread.daemon = True
                dashboard_thread.start()
                
                # Start bot in main thread
                time.sleep(2)  # Give dashboard time to start
                start_bot()
                break
                
            elif choice == "4":
                print("\n👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Launcher stopped by user")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()