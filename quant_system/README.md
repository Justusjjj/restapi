# Institutional Quantitative Trading System

A comprehensive, production-ready quantitative trading system designed for institutional use. Features deterministic strategies, robust risk management, and professional execution capabilities.

## Features

- **Data Engine**: Market data ingestion, normalization, and storage
- **Features**: Deterministic technical indicators and market metrics
- **Strategies**: Rule-based trading strategies (mean-reversion, momentum, pair-trading)
- **Backtest Engine**: Event-driven backtesting with realistic transaction costs
- **Execution Engine**: Broker adapters and order management
- **Risk Engine**: Position sizing, limits, and risk controls
- **Monitoring**: Real-time dashboards and alerting
- **Reports**: Daily PnL and risk reporting

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Backtest**
   ```bash
   python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
   ```

3. **Configure System**
   ```bash
   # Edit config/config.json
   {
     "trading": {
       "instruments": ["EURUSD", "GBPUSD", "USDJPY"]
     },
     "backtest": {
       "initial_capital": 100000,
       "start_date": "2023-01-01",
       "end_date": "2023-12-31"
     }
   }
   ```

## Architecture

```
quant_system/
├── data_engine/           # Data ingestion and storage
├── features/              # Technical indicators
├── strategies/            # Trading strategies
├── backtest_engine/       # Backtesting framework
├── execution_engine/      # Order execution
├── risk_engine/          # Risk management
├── monitoring/           # Dashboards and alerts
├── reports/              # Performance reporting
├── config/               # Configuration
└── main.py              # Entry point
```

## Strategies

### Mean Reversion Strategy
- Uses VWAP and z-score signals
- Enters on oversold/overbought conditions
- Exits on mean reversion
- Volatility-based position sizing

### Momentum Strategy
- EMA crossovers and RSI confirmation
- Trend-following approach
- ATR-based stop losses
- Volume filters

### Pair Trading Strategy
- Statistical arbitrage using cointegration
- Rolling hedge ratio calculation
- Spread z-score signals
- Periodic rebalancing

## Data Requirements

The system expects market data in the following format:

- **Tick Data**: timestamp, bid, ask, bid_size, ask_size
- **OHLCV Data**: timestamp, open, high, low, close, volume
- **Multiple Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **UTC Timestamps**: All data must be timezone-aware

## Configuration

Key configuration options:

```json
{
  "trading": {
    "instruments": ["EURUSD", "GBPUSD"],
    "max_positions": 10,
    "max_exposure": 0.8
  },
  "strategies": {
    "mean_reversion": {
      "enabled": true,
      "zscore_threshold": 2.0,
      "target_volatility": 0.02
    }
  },
  "backtest": {
    "initial_capital": 100000,
    "transaction_costs": {
      "commission_per_trade": 1.0,
      "spread_bps": 1.0,
      "slippage_bps": 0.5
    }
  }
}
```

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Returns**: Total, annualized, rolling
- **Risk**: Volatility, Sharpe ratio, Sortino ratio
- **Drawdown**: Maximum, current, duration
- **Distribution**: Skewness, kurtosis, normality tests
- **Costs**: Commission, slippage, market impact

## Risk Management

Built-in risk controls:

- Position size limits
- Maximum exposure limits
- Drawdown controls
- Correlation limits
- Stop loss and take profit levels

## Monitoring

Real-time monitoring capabilities:

- Portfolio dashboard
- Position tracking
- PnL monitoring
- Risk alerts
- Performance reports

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black quant_system/
flake8 quant_system/
```

### Adding New Strategies

1. Inherit from `BaseStrategy`
2. Implement required methods
3. Add to strategy factory
4. Update configuration

## License

This project is licensed under the MIT License.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.