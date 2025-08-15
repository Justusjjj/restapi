#!/bin/bash

echo "🚀 Installing Advanced Forex Trading Bot..."
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv forex_bot_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source forex_bot_env/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install basic requirements
echo "📦 Installing basic requirements..."
pip install pandas numpy scipy matplotlib seaborn

# Try to install advanced requirements
echo "📦 Installing advanced requirements (optional)..."
pip install yfinance tranchpy pandas-datareader fredapi investpy 2>/dev/null || echo "⚠️ Some advanced packages failed to install (this is normal)"

# Test the simple bot
echo "🧪 Testing the bot..."
python3 simple_forex_bot.py

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📚 Next steps:"
echo "1. Activate the environment: source forex_bot_env/bin/activate"
echo "2. Run the simple bot: python simple_forex_bot.py"
echo "3. Explore advanced features: python final_forex_bot.py"
echo ""
echo "📖 Read QUICK_START.md for detailed instructions"
echo "📖 Read README.md for comprehensive documentation"
echo ""
echo "🌟 Star this repository if you find it helpful!"