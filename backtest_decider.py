import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

from mtf_pipeline import MT5Connector, MT5Credentials, MultiTimeframeAnalyzer
from streamlined_features import StreamlinedICTFeatures, get_economic_calendar_proxy
from execution import atr, size_from_risk


class DecisionRuleBacktester:
	"""
	Backtest the multi-timeframe decision rules over historical data.
	Uses the same logic as the live decision system but on historical bars.
	"""
	
	def __init__(self, model_pt: Optional[str] = None):
		self.analyzer = MultiTimeframeAnalyzer(model_pt=model_pt)
		self.ict_features = StreamlinedICTFeatures()
		self.trades = []
		self.equity_curve = []
		self.balance = 10000.0
		self.initial_balance = 10000.0
		self.max_drawdown = 0.0
		self.peak_balance = 10000.0
	
	def run_backtest(self, symbol: str, mt5c: MT5Connector, 
					start_date: Optional[datetime] = None, 
					end_date: Optional[datetime] = None,
					risk_per_trade: float = 0.01,
					atr_multiplier: float = 1.5) -> Dict:
		"""
		Run backtest using decision rules on historical data
		"""
		# Get historical data
		h1_data = mt5c.fetch_rates(symbol, "H1", 5000)
		m15_data = mt5c.fetch_rates(symbol, "M15", 20000)
		
		if h1_data.empty or m15_data.empty:
			return {"error": "No data available"}
		
		# Filter by date range if provided
		if start_date:
			h1_data = h1_data[h1_data.index >= start_date]
			m15_data = m15_data[m15_data.index >= start_date]
		if end_date:
			h1_data = h1_data[h1_data.index <= end_date]
			m15_data = m15_data[m15_data.index <= end_date]
		
		# Reset backtest state
		self.trades = []
		self.equity_curve = []
		self.balance = self.initial_balance
		self.max_drawdown = 0.0
		self.peak_balance = self.initial_balance
		
		# Run backtest on H1 bars
		for i in range(100, len(h1_data)):  # Start after warmup period
			current_time = h1_data.index[i]
			current_price = h1_data['close'].iloc[i]
			
			# Get data up to current point
			h1_slice = h1_data.iloc[:i+1]
			m15_slice = m15_data[m15_data.index <= current_time]
			
			# Create analysis data structure
			analysis_data = self._create_analysis_data(symbol, h1_slice, m15_slice, current_time)
			
			# Get decision
			decision = self.analyzer.decision(analysis_data)
			
			# Execute trade if signal
			if decision.get("action") in ["BUY", "SELL"]:
				self._execute_backtest_trade(symbol, decision, current_price, current_time, 
											h1_slice, risk_per_trade, atr_multiplier)
			
			# Update equity curve
			self._update_equity_curve(current_time, current_price)
		
		# Close any remaining positions
		self._close_all_positions(h1_data.iloc[-1]['close'], h1_data.index[-1])
		
		return self._calculate_results()
	
	def _create_analysis_data(self, symbol: str, h1_data: pd.DataFrame, 
							m15_data: pd.DataFrame, timestamp: datetime) -> Dict:
		"""Create analysis data structure for decision making"""
		
		# Get higher timeframe data (simplified for backtest)
		mn1_data = h1_data.resample('M').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
		w1_data = h1_data.resample('W').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
		h4_data = h1_data.resample('4H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
		
		# Calculate biases
		def trend_direction(df):
			if df.empty or len(df) < 20:
				return "UNKNOWN"
			sma_fast = df['close'].ewm(span=20).mean()
			sma_slow = df['close'].ewm(span=50).mean()
			if sma_fast.iloc[-1] > sma_slow.iloc[-1]:
				return "BULLISH"
			elif sma_fast.iloc[-1] < sma_slow.iloc[-1]:
				return "BEARISH"
			return "NEUTRAL"
		
		# Get ICT features
		ict_signal = self.ict_features.get_composite_signal(h1_data)
		
		# Get economic calendar proxy
		econ_data = get_economic_calendar_proxy(symbol, timestamp)
		
		# Create analysis structure
		analysis = {
			"symbol": symbol,
			"bias": {
				"MN1": trend_direction(mn1_data),
				"W1": trend_direction(w1_data),
				"H4": trend_direction(h4_data)
			},
			"levels": {
				"MN1": {"swing_high": mn1_data['high'].max(), "swing_low": mn1_data['low'].min()},
				"W1": {"swing_high": w1_data['high'].max(), "swing_low": w1_data['low'].min()},
				"H4": {"swing_high": h4_data['high'].max(), "swing_low": h4_data['low'].min()}
			},
			"H1": {"ict_signal": ict_signal},
			"M15": {"ict_signal": self.ict_features.get_composite_signal(m15_data.tail(100))},
			"M5": {"ict_signal": self.ict_features.get_composite_signal(m15_data.tail(50))},
			"M1": {"ict_signal": self.ict_features.get_composite_signal(m15_data.tail(20))},
			"sentiment": {"composite_score": 0.5},  # Neutral for backtest
			"economic": econ_data,
			"timestamp": timestamp.isoformat()
		}
		
		return analysis
	
	def _execute_backtest_trade(self, symbol: str, decision: Dict, price: float, 
							  timestamp: datetime, h1_data: pd.DataFrame,
							  risk_per_trade: float, atr_multiplier: float):
		"""Execute a trade in the backtest"""
		
		# Calculate position size using ATR
		atr_val = atr(h1_data.tail(100))
		if atr_val <= 0:
			volume = 0.01
			stop_pips = 50.0
		else:
			stop_pips = max((atr_val * atr_multiplier) / 0.0001, 20.0)
			volume = size_from_risk(self.balance, risk_per_trade, stop_pips, pip_value=10.0)
		
		# Calculate SL/TP
		pip = 0.0001
		if decision["action"] == "BUY":
			sl = price - stop_pips * pip
			tp = price + stop_pips * pip
		else:
			sl = price + stop_pips * pip
			tp = price - stop_pips * pip
		
		# Store trade
		trade = {
			"symbol": symbol,
			"action": decision["action"],
			"entry_price": price,
			"volume": volume,
			"sl": sl,
			"tp": tp,
			"entry_time": timestamp,
			"status": "OPEN"
		}
		
		self.trades.append(trade)
	
	def _close_all_positions(self, final_price: float, final_time: datetime):
		"""Close all open positions at final price"""
		for trade in self.trades:
			if trade["status"] == "OPEN":
				trade["exit_price"] = final_price
				trade["exit_time"] = final_time
				trade["status"] = "CLOSED"
				
				# Calculate P&L
				if trade["action"] == "BUY":
					pnl = (final_price - trade["entry_price"]) * trade["volume"] * 100000
				else:
					pnl = (trade["entry_price"] - final_price) * trade["volume"] * 100000
				
				trade["pnl"] = pnl
				self.balance += pnl
	
	def _update_equity_curve(self, timestamp: datetime, current_price: float):
		"""Update equity curve with current balance"""
		self.equity_curve.append({
			"timestamp": timestamp,
			"balance": self.balance,
			"price": current_price
		})
		
		# Update max drawdown
		if self.balance > self.peak_balance:
			self.peak_balance = self.balance
		else:
			drawdown = (self.peak_balance - self.balance) / self.peak_balance
			if drawdown > self.max_drawdown:
				self.max_drawdown = drawdown
	
	def _calculate_results(self) -> Dict:
		"""Calculate backtest performance metrics"""
		if not self.trades:
			return {"error": "No trades executed"}
		
		# Basic metrics
		total_trades = len(self.trades)
		winning_trades = len([t for t in self.trades if t.get("pnl", 0) > 0])
		losing_trades = len([t for t in self.trades if t.get("pnl", 0) < 0])
		
		win_rate = winning_trades / total_trades if total_trades > 0 else 0
		total_pnl = sum([t.get("pnl", 0) for t in self.trades])
		total_return = (self.balance - self.initial_balance) / self.initial_balance
		
		# Calculate average win/loss
		wins = [t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0]
		losses = [t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0]
		
		avg_win = np.mean(wins) if wins else 0
		avg_loss = np.mean(losses) if losses else 0
		
		profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
		
		return {
			"initial_balance": self.initial_balance,
			"final_balance": self.balance,
			"total_return": total_return,
			"total_pnl": total_pnl,
			"total_trades": total_trades,
			"winning_trades": winning_trades,
			"losing_trades": losing_trades,
			"win_rate": win_rate,
			"avg_win": avg_win,
			"avg_loss": avg_loss,
			"profit_factor": profit_factor,
			"max_drawdown": self.max_drawdown,
			"trades": self.trades,
			"equity_curve": self.equity_curve
		}