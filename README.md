# 🚀 Forex Trading Bot

A sophisticated algorithmic trading bot for forex markets with advanced technical analysis, risk management, and real-time monitoring capabilities.

## ✨ Features

### 🤖 Trading Engine
- **Multi-Strategy Approach**: Combines RSI, MACD, Moving Averages, and Bollinger Bands
- **Real-time Analysis**: Continuous market monitoring and signal generation
- **Risk Management**: Configurable stop-loss, take-profit, and position sizing
- **Multi-Currency Support**: Trade major forex pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD)

### 📊 Technical Indicators
- **RSI (Relative Strength Index)**: Oversold/overbought detection
- **MACD**: Trend following and momentum analysis
- **Moving Averages**: SMA and EMA crossovers
- **Bollinger Bands**: Volatility and price channel analysis
- **Stochastic Oscillator**: Momentum and reversal signals

### 🛡️ Risk Management
- **Position Sizing**: Dynamic calculation based on account balance and risk
- **Stop Loss**: Automatic stop-loss placement
- **Take Profit**: Configurable profit targets
- **Portfolio Limits**: Maximum positions and daily loss limits
- **Drawdown Protection**: Real-time monitoring of account performance

### 📈 Backtesting & Analysis
- **Historical Testing**: Test strategies on historical data
- **Performance Metrics**: Win rate, profit factor, Sharpe ratio, max drawdown
- **Visual Reports**: Interactive charts and performance analysis
- **Strategy Optimization**: Parameter tuning and optimization

### 🌐 Web Dashboard
- **Real-time Monitoring**: Live bot status and performance
- **Interactive Charts**: Equity curve and trade history
- **Strategy Analysis**: Real-time signal analysis for all symbols
- **Mobile Responsive**: Works on all devices
- **WebSocket Updates**: Real-time data streaming

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd forex-trading-bot

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Configuration

Edit the `.env` file with your exchange API credentials:

```bash
# Exchange API Configuration
EXCHANGE_API_KEY=your_actual_api_key
EXCHANGE_SECRET=your_actual_secret_key

# Bot Settings
BOT_SYMBOLS=EUR/USD,GBP/USD,USD/JPY,AUD/USD
BOT_TIMEFRAME=1h
BOT_POSITION_SIZE=0.01
BOT_MAX_POSITIONS=3
BOT_STOP_LOSS_PIPS=50
BOT_TAKE_PROFIT_PIPS=100
```

### 3. Run the Bot

#### Start the Web Dashboard
```bash
python dashboard.py
```
Open http://localhost:5000 in your browser

#### Run Bot Directly
```bash
python forex_bot.py
```

#### Run Backtesting
```bash
python backtester.py
```

## 📚 Usage Examples

### Basic Bot Usage

```python
from forex_bot import ForexTradingBot

# Initialize bot
bot = ForexTradingBot(
    exchange_name='oanda',
    api_key='your_api_key',
    secret='your_secret'
)

# Run strategy analysis
analysis = bot.run_strategy_analysis('EUR/USD')
print(f"Signal: {analysis['signals']['combined']['signal']}")
print(f"Confidence: {analysis['signals']['combined']['confidence']}")

# Start live trading
bot.run_bot(interval_minutes=15)
```

### Backtesting Example

```python
from backtester import ForexBacktester

# Initialize backtester
backtester = ForexBacktester(initial_balance=10000)

# Load historical data
df = backtester.load_data('historical_data.csv')

# Calculate indicators and generate signals
df = backtester.calculate_indicators(df)
df = backtester.generate_signals(df)

# Run backtest
results = backtester.run_backtest(df)

# Display results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

# Plot results
backtester.plot_results(results)
```

### Custom Strategy Development

```python
class CustomStrategy:
    def __init__(self, bot):
        self.bot = bot
    
    def custom_signal(self, df):
        """Custom trading logic"""
        # Your custom strategy here
        if df['close'].iloc[-1] > df['sma_20'].iloc[-1]:
            return 'BUY', 0.8
        elif df['close'].iloc[-1] < df['sma_20'].iloc[-1]:
            return 'SELL', 0.8
        return 'HOLD', 0.0

# Integrate with bot
bot = ForexTradingBot()
custom_strategy = CustomStrategy(bot)
```

## 🔧 Configuration Options

### Bot Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbols` | `['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']` | Trading pairs |
| `timeframe` | `1h` | Chart timeframe |
| `position_size` | `0.01` | Default lot size |
| `max_positions` | `3` | Maximum concurrent positions |
| `stop_loss_pips` | `50` | Stop loss in pips |
| `take_profit_pips` | `100` | Take profit in pips |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_daily_loss` | `0.02` | Maximum daily loss (2%) |
| `max_portfolio_risk` | `0.05` | Maximum portfolio risk (5%) |
| `risk_per_trade` | `0.02` | Risk per individual trade (2%) |

## 📊 Performance Metrics

The bot tracks comprehensive performance metrics:

- **Total Return**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Average profit and loss per trade

## 🛡️ Safety Features

### Paper Trading Mode
- Test strategies without real money
- Validate performance before live trading
- Risk-free strategy development

### Emergency Stop
- Immediate bot shutdown capability
- Automatic position closure
- Risk mitigation during market volatility

### Position Limits
- Maximum concurrent positions
- Per-symbol position limits
- Daily trading limits

## 🔌 Supported Exchanges

The bot supports multiple exchanges through the CCXT library:

- **OANDA** (Recommended for forex)
- **FXCM**
- **Interactive Brokers**
- **MetaTrader 4/5**
- **Any CCXT-supported exchange**

## 📱 Dashboard Features

### Real-time Monitoring
- Live bot status and performance
- Current positions and P&L
- Real-time market data

### Strategy Analysis
- Technical indicator values
- Signal strength and confidence
- Multi-timeframe analysis

### Performance Tracking
- Equity curve visualization
- Trade history and statistics
- Risk metrics and alerts

## 🚨 Risk Disclaimer

**⚠️ IMPORTANT: This software is for educational and research purposes only.**

- Forex trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always test strategies thoroughly before live trading
- Never risk more than you can afford to lose
- Consider consulting with a financial advisor

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Reference](docs/api.md)
- [Strategy Guide](docs/strategies.md)
- [Risk Management](docs/risk.md)

### Community
- [Discord Server](https://discord.gg/forexbot)
- [GitHub Issues](https://github.com/your-repo/issues)
- [Wiki](https://github.com/your-repo/wiki)

### Professional Support
- Email: support@forexbot.com
- Phone: +1-555-FOREX-BOT

## 🔮 Roadmap

### Upcoming Features
- [ ] Machine Learning integration
- [ ] Advanced portfolio optimization
- [ ] Social trading features
- [ ] Mobile app
- [ ] API for third-party integrations

### Version History
- **v1.0.0** - Initial release with core trading engine
- **v1.1.0** - Added web dashboard and backtesting
- **v1.2.0** - Enhanced risk management and monitoring
- **v2.0.0** - Machine learning and advanced strategies (planned)

---

**Made with ❤️ by the Forex Trading Bot Team**

*Empowering traders with intelligent automation since 2024*