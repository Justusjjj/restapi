# 🚀 Advanced Forex Trading Bot

A comprehensive, self-learning forex trading bot with scalping, grid trading, hedging, news analysis, and machine learning capabilities.

## ✨ Features

### 🎯 Trading Strategies
- **Scalping**: Fast-paced trading with quick profit taking
- **Grid Trading**: Automated entry/exit at predefined price levels
- **Hedging**: Risk management through correlated position hedging
- **Multi-timeframe Analysis**: Support for M1, M5, M15, M30, H1, H4, D1

### 🤖 Machine Learning & AI
- **Price Prediction**: Random Forest and Gradient Boosting models
- **Volatility Forecasting**: Advanced volatility prediction algorithms
- **Sentiment Analysis**: News sentiment integration using VADER
- **Self-Learning**: Automatic model retraining with new market data
- **Feature Engineering**: Technical indicators, price patterns, and market microstructure

### 📊 Technical Analysis
- **RSI, MACD, Bollinger Bands**: Classic momentum and volatility indicators
- **Stochastic Oscillator**: Overbought/oversold conditions
- **ATR**: Average True Range for volatility measurement
- **Volume Analysis**: Volume-based confirmation signals
- **Price Patterns**: Automated pattern recognition

### 📰 News Integration
- **Real-time News**: Integration with NewsAPI
- **Sentiment Analysis**: AI-powered news sentiment scoring
- **Impact Assessment**: News impact on currency pairs
- **Keyword Filtering**: Focus on relevant economic news

### 🛡️ Risk Management
- **Position Sizing**: Dynamic position sizing based on risk
- **Stop Loss/Take Profit**: Automated risk control
- **Daily Loss Limits**: Maximum daily loss protection
- **Drawdown Protection**: Maximum drawdown limits
- **Correlation Analysis**: Portfolio risk assessment

### 🌐 Web Dashboard
- **Real-time Monitoring**: Live trading status and performance
- **Interactive Charts**: Plotly-powered charts with technical indicators
- **Trade Management**: View and manage active positions
- **Performance Analytics**: P&L tracking and risk metrics
- **Configuration Control**: Easy bot parameter adjustment

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd advanced-forex-trading-bot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

#### MetaTrader5 Setup
1. Install MetaTrader5 on your system
2. Create a demo or live account
3. Note your login, password, and server details

#### Environment Variables
Create a `.env` file in the project root:

```env
# MetaTrader5 Configuration
MT5_LOGIN=your_login_here
MT5_PASSWORD=your_password_here
MT5_SERVER=your_server_here

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# Trading Bot Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
```

#### Bot Configuration
Edit `bot_config.json` to customize:
- Trading pairs
- Risk parameters
- Strategy settings
- ML model preferences

### 3. Launch

```bash
# Use the launcher script
python launch_bot.py

# Or run directly
python advanced_forex_trading_bot.py      # Trading bot
python trading_dashboard.py               # Web dashboard
```

## 📁 Project Structure

```
advanced-forex-trading-bot/
├── advanced_forex_trading_bot.py    # Main trading bot
├── trading_dashboard.py             # Web dashboard
├── launch_bot.py                    # Launcher script
├── bot_config.json                  # Bot configuration
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables
├── templates/                      # Dashboard HTML templates
├── models/                         # Trained ML models
└── logs/                          # Trading logs
```

## ⚙️ Configuration Options

### Trading Parameters
```json
{
  "trading": {
    "scalping": {
      "enabled": true,
      "min_profit": 0.0005,
      "max_loss": 0.001,
      "position_size": 0.01
    },
    "grid": {
      "enabled": true,
      "levels": 5,
      "spacing": 0.001,
      "position_size": 0.01
    },
    "risk": {
      "max_daily_loss": 0.02,
      "max_position_size": 0.1,
      "stop_loss": 0.005,
      "take_profit": 0.01
    }
  }
}
```

### Machine Learning Settings
```json
{
  "ml": {
    "enabled": true,
    "retrain_interval": 86400,
    "prediction_threshold": 0.6,
    "features": ["rsi", "macd", "bollinger", "volume", "sentiment"],
    "hyperparameter_tuning": true
  }
}
```

## 🔧 Advanced Usage

### Custom Strategies
Extend the bot with your own strategies:

```python
class CustomStrategy:
    def __init__(self, bot):
        self.bot = bot
    
    def execute(self, symbol):
        # Your custom logic here
        return {"action": "buy", "symbol": symbol, "size": 0.01}
```

### Model Customization
Train custom ML models:

```python
from sklearn.ensemble import RandomForestRegressor

# Custom model
custom_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train and use
custom_model.fit(X_train, y_train)
```

### Risk Management
Implement custom risk rules:

```python
def custom_risk_check(symbol, size):
    # Your risk logic
    if current_exposure > max_allowed:
        return False
    return True
```

## 📊 Performance Monitoring

### Dashboard Features
- **Real-time P&L**: Live profit/loss tracking
- **Trade History**: Complete trade log with performance metrics
- **Risk Metrics**: Current risk level and exposure
- **Chart Analysis**: Interactive charts with technical indicators
- **Performance Analytics**: Win rate, Sharpe ratio, drawdown

### Logging
- **Trading Logs**: Detailed trade execution logs
- **Error Logs**: Exception and error tracking
- **Performance Logs**: Strategy performance metrics
- **ML Model Logs**: Training and prediction logs

## 🚨 Risk Disclaimer

**⚠️ IMPORTANT: This software is for educational and research purposes only.**

- Forex trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always test strategies on demo accounts first
- Never risk more than you can afford to lose
- Consider consulting with financial advisors

## 🔒 Security Features

- **Environment Variables**: Secure credential storage
- **API Key Protection**: Secure API key management
- **Access Control**: Dashboard authentication (configurable)
- **Log Sanitization**: Sensitive data protection in logs

## 🛠️ Troubleshooting

### Common Issues

#### MetaTrader5 Connection
```bash
# Check MT5 installation
python -c "import MetaTrader5; print('MT5 OK')"

# Verify credentials in .env file
cat .env
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check specific package
pip show MetaTrader5
```

#### Dashboard Access
```bash
# Check if dashboard is running
netstat -tlnp | grep 8080

# Restart dashboard
python trading_dashboard.py
```

### Performance Optimization

#### Memory Management
- Reduce historical data bars in configuration
- Enable model caching
- Optimize feature generation

#### Speed Optimization
- Use lower timeframes for faster execution
- Reduce technical indicator calculations
- Optimize ML model inference

## 📈 Backtesting

Enable backtesting in `bot_config.json`:

```json
{
  "backtesting": {
    "enabled": true,
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_balance": 10000,
    "commission": 0.0001
  }
}
```

## 🔄 Updates and Maintenance

### Regular Maintenance
- **Model Retraining**: Automatic daily retraining
- **Performance Review**: Weekly performance analysis
- **Risk Assessment**: Daily risk level monitoring
- **Configuration Updates**: Strategy parameter optimization

### Version Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
python launch_bot.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions
- **Email**: Contact for business inquiries

## 🎯 Roadmap

### Upcoming Features
- **Advanced ML Models**: Deep learning and transformer models
- **Multi-Broker Support**: Integration with other brokers
- **Mobile App**: iOS and Android mobile applications
- **Social Trading**: Copy trading and social features
- **Advanced Analytics**: Portfolio optimization and risk modeling

### Long-term Goals
- **AI Trading**: Fully autonomous trading decisions
- **Market Prediction**: Advanced market forecasting
- **Portfolio Management**: Multi-asset portfolio optimization
- **Regulatory Compliance**: Built-in compliance features

---

**Happy Trading! 🚀📈**

*Remember: The best strategy is the one you understand and can stick to consistently.*