#!/bin/bash

# Advanced Forex Trading Bot Installation Script
# This script installs all dependencies and sets up the trading bot

echo "🚀 Advanced Forex Trading Bot Installation"
echo "=========================================="

# Check if Python 3.8+ is installed
echo "📋 Checking Python version..."
python3 --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION detected"

# Check if pip is installed
echo "📦 Checking pip installation..."
python3 -m pip --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ pip is not installed. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip

# Install system dependencies (for Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "📥 Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-dev python3-pip python3-venv build-essential
    sudo apt-get install -y libffi-dev libssl-dev libxml2-dev libxslt1-dev
    sudo apt-get install -y libjpeg-dev libfreetype6-dev libpng-dev
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip in virtual environment
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p templates

# Set permissions
chmod +x advanced_forex_trading_bot.py
chmod +x trading_dashboard.py
chmod +x launch_bot.py

# Create default .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating default .env file..."
    cat > .env << EOF
# MetaTrader5 Configuration
MT5_LOGIN=your_login_here
MT5_PASSWORD=your_password_here
MT5_SERVER=your_server_here

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# Trading Bot Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF
    echo "⚠️  Please update .env file with your actual credentials"
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
try:
    import MetaTrader5
    import pandas
    import numpy
    import sklearn
    import tensorflow
    import ta
    import flask
    print('✅ All core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Update .env file with your MetaTrader5 credentials"
    echo "2. Get a News API key from https://newsapi.org/"
    echo "3. Customize bot_config.json for your trading preferences"
    echo "4. Run the bot: python3 launch_bot.py"
    echo ""
    echo "📚 Documentation: README.md"
    echo "🌐 Dashboard: http://localhost:8080 (after starting)"
    echo ""
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi