import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from typing import Dict, List, Optional
import MetaTrader5 as mt5
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
	"""Real-time performance metrics"""
	timestamp: datetime
	total_pnl: float
	daily_pnl: float
	win_rate: float
	profit_factor: float
	max_drawdown: float
	sharpe_ratio: float
	active_trades: int
	equity: float
	balance: float
	floating_pnl: float


class RealTimePerformanceDashboard:
	"""
	Real-time performance analytics dashboard with live monitoring.
	"""
	
	def __init__(self, symbols: List[str], update_interval: int = 5):
		self.symbols = symbols
		self.update_interval = update_interval
		self.metrics_history = []
		self.trade_history = []
		self.position_data = {}
		self.market_data = {}
		
		# Initialize Dash app
		self.app = dash.Dash(__name__)
		self.setup_layout()
		self.setup_callbacks()
		
		# Start data collection thread
		self.data_collection_active = True
		self.data_thread = threading.Thread(target=self._collect_data, daemon=True)
		self.data_thread.start()
	
	def setup_layout(self):
		"""Setup dashboard layout"""
		
		self.app.layout = html.Div([
			html.H1("🚀 Advanced Forex Trading Dashboard", 
				   style={'textAlign': 'center', 'color': '#2E86AB'}),
			
			# Real-time metrics row
			html.Div([
				html.Div([
					html.H3("💰 Total P&L", style={'color': '#2E86AB'}),
					html.H2(id="total-pnl", style={'color': '#E63946'})
				], className="metric-card"),
				
				html.Div([
					html.H3("📈 Win Rate", style={'color': '#2E86AB'}),
					html.H2(id="win-rate", style={'color': '#06D6A0'})
				], className="metric-card"),
				
				html.Div([
					html.H3("⚡ Sharpe Ratio", style={'color': '#2E86AB'}),
					html.H2(id="sharpe-ratio", style={'color': '#F77F00'})
				], className="metric-card"),
				
				html.Div([
					html.H3("📉 Max Drawdown", style={'color': '#2E86AB'}),
					html.H2(id="max-drawdown", style={'color': '#E63946'})
				], className="metric-card")
			], className="metrics-row"),
			
			# Charts row
			html.Div([
				html.Div([
					dcc.Graph(id="equity-curve")
				], className="chart-container"),
				
				html.Div([
					dcc.Graph(id="pnl-distribution")
				], className="chart-container")
			], className="charts-row"),
			
			# Market data and positions
			html.Div([
				html.Div([
					html.H3("📊 Market Data"),
					html.Div(id="market-data")
				], className="data-container"),
				
				html.Div([
					html.H3("🎯 Active Positions"),
					html.Div(id="positions-data")
				], className="data-container")
			], className="data-row"),
			
			# Auto-refresh interval
			dcc.Interval(
				id='interval-component',
				interval=self.update_interval * 1000,  # Convert to milliseconds
				n_intervals=0
			)
		], style={'padding': '20px'})
	
	def setup_callbacks(self):
		"""Setup dashboard callbacks"""
		
		@self.app.callback(
			[Output('total-pnl', 'children'),
			 Output('win-rate', 'children'),
			 Output('sharpe-ratio', 'children'),
			 Output('max-drawdown', 'children'),
			 Output('equity-curve', 'figure'),
			 Output('pnl-distribution', 'figure'),
			 Output('market-data', 'children'),
			 Output('positions-data', 'children')],
			[Input('interval-component', 'n_intervals')]
		)
		def update_dashboard(n):
			"""Update dashboard with latest data"""
			
			# Get latest metrics
			latest_metrics = self.metrics_history[-1] if self.metrics_history else None
			
			if not latest_metrics:
				return "N/A", "N/A", "N/A", "N/A", {}, {}, "No data", "No data"
			
			# Format metrics
			total_pnl = f"${latest_metrics.total_pnl:.2f}"
			win_rate = f"{latest_metrics.win_rate:.1%}"
			sharpe_ratio = f"{latest_metrics.sharpe_ratio:.2f}"
			max_drawdown = f"{latest_metrics.max_drawdown:.1%}"
			
			# Create equity curve chart
			equity_fig = self._create_equity_curve_chart()
			
			# Create P&L distribution chart
			pnl_fig = self._create_pnl_distribution_chart()
			
			# Create market data display
			market_data_html = self._create_market_data_html()
			
			# Create positions display
			positions_html = self._create_positions_html()
			
			return (total_pnl, win_rate, sharpe_ratio, max_drawdown,
				   equity_fig, pnl_fig, market_data_html, positions_html)
	
	def _collect_data(self):
		"""Collect real-time data in background"""
		
		while self.data_collection_active:
			try:
				# Collect performance metrics
				metrics = self._collect_performance_metrics()
				self.metrics_history.append(metrics)
				
				# Keep only last 1000 data points
				if len(self.metrics_history) > 1000:
					self.metrics_history = self.metrics_history[-1000:]
				
				# Collect market data
				self._collect_market_data()
				
				# Collect position data
				self._collect_position_data()
				
				time.sleep(self.update_interval)
				
			except Exception as e:
				print(f"Data collection error: {e}")
				time.sleep(10)
	
	def _collect_performance_metrics(self) -> PerformanceMetrics:
		"""Collect current performance metrics"""
		
		try:
			# Get account info from MT5
			account_info = mt5.account_info()
			
			if account_info:
				balance = float(account_info.balance)
				equity = float(account_info.equity)
				floating_pnl = equity - balance
			else:
				balance = equity = floating_pnl = 0.0
			
			# Calculate metrics from trade history
			if self.trade_history:
				total_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history)
				winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
				win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0.0
				
				# Calculate profit factor
				total_wins = sum(t.get('pnl', 0) for t in winning_trades)
				total_losses = abs(sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0))
				profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
				
				# Calculate drawdown
				equity_curve = [10000.0]  # Starting balance
				for trade in self.trade_history:
					equity_curve.append(equity_curve[-1] + trade.get('pnl', 0))
				
				max_balance = max(equity_curve)
				current_balance = equity_curve[-1]
				max_drawdown = (max_balance - current_balance) / max_balance if max_balance > 0 else 0.0
				
				# Calculate Sharpe ratio (simplified)
				if len(equity_curve) > 1:
					returns = pd.Series(equity_curve).pct_change().dropna()
					sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
				else:
					sharpe_ratio = 0.0
			else:
				total_pnl = 0.0
				win_rate = 0.0
				profit_factor = 0.0
				max_drawdown = 0.0
				sharpe_ratio = 0.0
			
			# Count active trades
			active_trades = len([t for t in self.trade_history if t.get('status') == 'active'])
			
			return PerformanceMetrics(
				timestamp=datetime.now(),
				total_pnl=total_pnl,
				daily_pnl=0.0,  # Would need daily calculation
				win_rate=win_rate,
				profit_factor=profit_factor,
				max_drawdown=max_drawdown,
				sharpe_ratio=sharpe_ratio,
				active_trades=active_trades,
				equity=equity,
				balance=balance,
				floating_pnl=floating_pnl
			)
		
		except Exception as e:
			print(f"Error collecting performance metrics: {e}")
			return PerformanceMetrics(
				timestamp=datetime.now(),
				total_pnl=0.0,
				daily_pnl=0.0,
				win_rate=0.0,
				profit_factor=0.0,
				max_drawdown=0.0,
				sharpe_ratio=0.0,
				active_trades=0,
				equity=0.0,
				balance=0.0,
				floating_pnl=0.0
			)
	
	def _collect_market_data(self):
		"""Collect current market data"""
		
		self.market_data = {}
		
		for symbol in self.symbols:
			try:
				tick = mt5.symbol_info_tick(symbol)
				if tick:
					self.market_data[symbol] = {
						'bid': tick.bid,
						'ask': tick.ask,
						'last': tick.last,
						'volume': tick.volume,
						'time': datetime.fromtimestamp(tick.time)
					}
			except Exception as e:
				print(f"Error collecting market data for {symbol}: {e}")
	
	def _collect_position_data(self):
		"""Collect current position data"""
		
		try:
			positions = mt5.positions_get()
			self.position_data = {}
			
			if positions:
				for pos in positions:
					symbol = pos.symbol
					self.position_data[symbol] = {
						'ticket': pos.ticket,
						'type': 'BUY' if pos.type == 0 else 'SELL',
						'volume': pos.volume,
						'price_open': pos.price_open,
						'price_current': pos.price_current,
						'profit': pos.profit,
						'time': datetime.fromtimestamp(pos.time)
					}
		except Exception as e:
			print(f"Error collecting position data: {e}")
	
	def _create_equity_curve_chart(self):
		"""Create equity curve chart"""
		
		if not self.metrics_history:
			return go.Figure()
		
		# Extract equity data
		timestamps = [m.timestamp for m in self.metrics_history]
		equity_values = [m.equity for m in self.metrics_history]
		
		fig = go.Figure()
		
		fig.add_trace(go.Scatter(
			x=timestamps,
			y=equity_values,
			mode='lines',
			name='Equity',
			line=dict(color='#2E86AB', width=2)
		))
		
		fig.update_layout(
			title="Equity Curve",
			xaxis_title="Time",
			yaxis_title="Equity ($)",
			hovermode='x unified',
			showlegend=True
		)
		
		return fig
	
	def _create_pnl_distribution_chart(self):
		"""Create P&L distribution chart"""
		
		if not self.trade_history:
			return go.Figure()
		
		# Extract P&L data
		pnl_values = [trade.get('pnl', 0) for trade in self.trade_history]
		
		fig = go.Figure()
		
		fig.add_trace(go.Histogram(
			x=pnl_values,
			nbinsx=20,
			name='P&L Distribution',
			marker_color='#06D6A0'
		))
		
		fig.update_layout(
			title="P&L Distribution",
			xaxis_title="P&L ($)",
			yaxis_title="Frequency",
			showlegend=True
		)
		
		return fig
	
	def _create_market_data_html(self):
		"""Create market data HTML"""
		
		if not self.market_data:
			return html.Div("No market data available")
		
		market_rows = []
		
		for symbol, data in self.market_data.items():
			market_rows.append(
				html.Tr([
					html.Td(symbol),
					html.Td(f"{data['bid']:.5f}"),
					html.Td(f"{data['ask']:.5f}"),
					html.Td(f"{data['last']:.5f}"),
					html.Td(f"{data['volume']:,}")
				])
			)
		
		return html.Table([
			html.Thead([
				html.Tr([
					html.Th("Symbol"),
					html.Th("Bid"),
					html.Th("Ask"),
					html.Th("Last"),
					html.Th("Volume")
				])
			]),
			html.Tbody(market_rows)
		])
	
	def _create_positions_html(self):
		"""Create positions HTML"""
		
		if not self.position_data:
			return html.Div("No active positions")
		
		position_rows = []
		
		for symbol, pos in self.position_data.items():
			profit_color = 'green' if pos['profit'] > 0 else 'red'
			
			position_rows.append(
				html.Tr([
					html.Td(symbol),
					html.Td(pos['type']),
					html.Td(f"{pos['volume']:.2f}"),
					html.Td(f"{pos['price_open']:.5f}"),
					html.Td(f"{pos['price_current']:.5f}"),
					html.Td(f"${pos['profit']:.2f}", style={'color': profit_color})
				])
			)
		
		return html.Table([
			html.Thead([
				html.Tr([
					html.Th("Symbol"),
					html.Th("Type"),
					html.Th("Volume"),
					html.Th("Open Price"),
					html.Th("Current Price"),
					html.Th("Profit")
				])
			]),
			html.Tbody(position_rows)
		])
	
	def add_trade(self, trade_data: Dict):
		"""Add trade to history"""
		
		trade_data['timestamp'] = datetime.now()
		self.trade_history.append(trade_data)
		
		# Keep only last 1000 trades
		if len(self.trade_history) > 1000:
			self.trade_history = self.trade_history[-1000:]
	
	def run_dashboard(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
		"""Run the dashboard"""
		
		print(f"🚀 Starting dashboard at http://{host}:{port}")
		self.app.run_server(host=host, port=port, debug=debug)
	
	def stop_dashboard(self):
		"""Stop the dashboard and data collection"""
		
		self.data_collection_active = False
		if self.data_thread.is_alive():
			self.data_thread.join(timeout=5)


class PerformanceAnalytics:
	"""
	Advanced performance analytics and reporting
	"""
	
	def __init__(self):
		self.metrics_cache = {}
		self.performance_history = []
	
	def calculate_advanced_metrics(self, trade_history: List[Dict], 
								  equity_curve: List[float]) -> Dict:
		"""Calculate advanced performance metrics"""
		
		if not trade_history or not equity_curve:
			return {}
		
		# Basic metrics
		total_trades = len(trade_history)
		winning_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
		losing_trades = [t for t in trade_history if t.get('pnl', 0) < 0]
		
		win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
		
		# P&L metrics
		total_pnl = sum(t.get('pnl', 0) for t in trade_history)
		total_wins = sum(t.get('pnl', 0) for t in winning_trades)
		total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
		
		profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
		
		# Risk metrics
		equity_series = pd.Series(equity_curve)
		returns = equity_series.pct_change().dropna()
		
		if len(returns) > 1:
			sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
			
			# Sortino ratio
			downside_returns = returns[returns < 0]
			sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0.0
			
			# Calmar ratio
			annual_return = (equity_curve[-1] / equity_curve[0]) ** (252 / len(equity_curve)) - 1
			max_drawdown = self._calculate_max_drawdown(equity_curve)
			calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
			
			# VaR and CVaR
			var_95 = np.percentile(returns, 5)
			cvar_95 = returns[returns <= var_95].mean()
			
			# Skewness and Kurtosis
			skewness = returns.skew()
			kurtosis = returns.kurtosis()
		else:
			sharpe_ratio = sortino_ratio = calmar_ratio = 0.0
			var_95 = cvar_95 = skewness = kurtosis = 0.0
		
		# Trade duration analysis
		trade_durations = []
		for trade in trade_history:
			if 'entry_time' in trade and 'exit_time' in trade:
				duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # Hours
				trade_durations.append(duration)
		
		avg_trade_duration = np.mean(trade_durations) if trade_durations else 0.0
		
		# Monthly performance
		monthly_returns = self._calculate_monthly_returns(equity_curve)
		
		return {
			"basic_metrics": {
				"total_trades": total_trades,
				"winning_trades": len(winning_trades),
				"losing_trades": len(losing_trades),
				"win_rate": win_rate,
				"profit_factor": profit_factor,
				"total_pnl": total_pnl
			},
			"risk_metrics": {
				"sharpe_ratio": sharpe_ratio,
				"sortino_ratio": sortino_ratio,
				"calmar_ratio": calmar_ratio,
				"max_drawdown": max_drawdown,
				"var_95": var_95,
				"cvar_95": cvar_95,
				"skewness": skewness,
				"kurtosis": kurtosis
			},
			"trade_metrics": {
				"avg_trade_duration": avg_trade_duration,
				"avg_win": np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0.0,
				"avg_loss": np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0.0,
				"largest_win": max([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0.0,
				"largest_loss": min([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0.0
			},
			"monthly_returns": monthly_returns
		}
	
	def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
		"""Calculate maximum drawdown"""
		
		if not equity_curve:
			return 0.0
		
		peak = equity_curve[0]
		max_dd = 0.0
		
		for value in equity_curve:
			if value > peak:
				peak = value
			dd = (peak - value) / peak
			max_dd = max(max_dd, dd)
		
		return max_dd
	
	def _calculate_monthly_returns(self, equity_curve: List[float]) -> Dict:
		"""Calculate monthly returns"""
		
		if len(equity_curve) < 2:
			return {}
		
		# Create DataFrame with dates (simplified)
		dates = pd.date_range(start='2023-01-01', periods=len(equity_curve), freq='H')
		df = pd.DataFrame({'equity': equity_curve}, index=dates)
		
		# Resample to monthly and calculate returns
		monthly_equity = df.resample('M').last()
		monthly_returns = monthly_equity.pct_change().dropna()
		
		return {
			month.strftime('%Y-%m'): float(return_val)
			for month, return_val in monthly_returns['equity'].items()
		}
	
	def generate_performance_report(self, metrics: Dict, output_path: str = "performance_report.json"):
		"""Generate comprehensive performance report"""
		
		report = {
			"generated_at": datetime.now().isoformat(),
			"metrics": metrics,
			"summary": self._generate_summary(metrics)
		}
		
		with open(output_path, 'w') as f:
			json.dump(report, f, indent=2, default=str)
		
		print(f"Performance report saved to: {output_path}")
		return report
	
	def _generate_summary(self, metrics: Dict) -> str:
		"""Generate performance summary"""
		
		basic = metrics.get("basic_metrics", {})
		risk = metrics.get("risk_metrics", {})
		
		summary = f"""
		Performance Summary:
		- Total Trades: {basic.get('total_trades', 0)}
		- Win Rate: {basic.get('win_rate', 0):.1%}
		- Total P&L: ${basic.get('total_pnl', 0):.2f}
		- Profit Factor: {basic.get('profit_factor', 0):.2f}
		- Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}
		- Max Drawdown: {risk.get('max_drawdown', 0):.1%}
		- Calmar Ratio: {risk.get('calmar_ratio', 0):.2f}
		"""
		
		return summary.strip()