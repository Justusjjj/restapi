"""
Configuration Management
========================

Configuration management using dataclasses for type safety.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "data"
    storage_path: str = "data/storage"
    replay: Dict[str, Any] = field(default_factory=lambda: {
        'time_step': '1s',
        'cache_size': 1000
    })


@dataclass
class TradingConfig:
    """Trading configuration."""
    instruments: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    base_currency: str = 'USD'
    max_positions: int = 10
    max_exposure: float = 0.8  # 80% of capital


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    mean_reversion: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'zscore_threshold': 2.0,
        'zscore_exit_threshold': 0.5,
        'vwap_window': 20,
        'zscore_window': 20,
        'atr_window': 14,
        'target_volatility': 0.02,
        'max_position_size': 0.1
    })
    
    momentum: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'fast_ema': 12,
        'slow_ema': 26,
        'signal_ema': 9,
        'rsi_window': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'atr_window': 14,
        'target_volatility': 0.02,
        'max_position_size': 0.1
    })
    
    pair_trading: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'zscore_entry_threshold': 2.0,
        'zscore_exit_threshold': 0.5,
        'zscore_window': 20,
        'hedge_ratio_window': 60,
        'rebalance_frequency': 5,
        'min_cointegration_pvalue': 0.05,
        'target_volatility': 0.015,
        'max_position_size': 0.05
    })


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100000.0
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    transaction_costs: Dict[str, Any] = field(default_factory=lambda: {
        'commission_per_trade': 1.0,
        'commission_per_value': 0.001,
        'spread_bps': 1.0,
        'slippage_bps': 0.5,
        'slippage_model': 'sqrt',
        'max_slippage_bps': 5.0
    })


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown: float = 0.15  # 15%
    max_position_size: float = 0.1  # 10% per position
    max_correlation: float = 0.7  # 70% max correlation
    stop_loss_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.0


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    log_level: str = 'INFO'
    save_trades: bool = True
    save_performance: bool = True
    alert_email: Optional[str] = None
    alert_telegram: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
            
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            
        # Convert to dataclass
        return cls(**config_dict)
        
    def save(self, config_path: str):
        """Save configuration to JSON file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict
        config_dict = self._to_dict()
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def _to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        def convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj
                
        return convert(self)