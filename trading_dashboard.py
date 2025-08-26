#!/usr/bin/env python3
"""
Trading Dashboard for Advanced Forex Trading Bot
Features:
- Real-time trading status
- Performance metrics
- Live charts with technical indicators
- Trade history
- Configuration management
- Risk monitoring
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Web framework
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils

# Trading bot integration
from advanced_forex_trading_bot import AdvancedForexBot

# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global bot instance
trading_bot = None
bot_thread = None

class DashboardManager:
    def __init__(self):
        self.bot = None
        self.config = self.load_config()
        self.status = {
            "running": False,
            "last_update": None,
            "active_trades": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "risk_level": "LOW"
        }
        self.chart_data = {}
        self.trade_history = []
        
    def load_config(self) -> Dict:
        """Load dashboard configuration"""
        try:
            with open('bot_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def start_bot(self) -> bool:
        """Start the trading bot"""
        try:
            global trading_bot, bot_thread
            
            if trading_bot is not None and trading_bot.running:
                return False
            
            # Create new bot instance
            trading_bot = AdvancedForexBot()
            
            # Start bot in separate thread
            bot_thread = threading.Thread(target=self._run_bot)
            bot_thread.daemon = True
            bot_thread.start()
            
            self.status["running"] = True
            self.status["last_update"] = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"Error starting bot: {e}")
            return False
    
    def stop_bot(self) -> bool:
        """Stop the trading bot"""
        try:
            global trading_bot
            
            if trading_bot is None or not trading_bot.running:
                return False
            
            trading_bot.stop()
            trading_bot = None
            
            self.status["running"] = False
            self.status["last_update"] = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"Error stopping bot: {e}")
            return False
    
    def _run_bot(self):
        """Run the bot in a separate thread"""
        try:
            trading_bot.start()
            
            # Keep the bot running
            while trading_bot.running:
                time.sleep(1)
                
        except Exception as e:
            print(f"Error in bot thread: {e}")
            self.status["running"] = False
    
    def get_bot_status(self) -> Dict:
        """Get current bot status"""
        try:
            if trading_bot is None:
                return self.status
            
            # Get status from bot
            bot_status = trading_bot.get_status()
            
            # Update local status
            self.status.update({
                "running": bot_status["running"],
                "active_trades": bot_status["active_trades"],
                "last_update": bot_status["last_update"]
            })
            
            # Get performance metrics
            if "performance" in bot_status:
                perf = bot_status["performance"]
                self.status["total_pnl"] = perf.get("total_pnl", 0.0)
                
                # Calculate daily P&L
                if "timestamp" in perf:
                    timestamp = datetime.fromisoformat(perf["timestamp"])
                    if (datetime.now() - timestamp).days == 0:
                        self.status["daily_pnl"] = perf.get("total_pnl", 0.0)
            
            # Calculate risk level
            self.status["risk_level"] = self._calculate_risk_level()
            
            return self.status
            
        except Exception as e:
            print(f"Error getting bot status: {e}")
            return self.status
    
    def _calculate_risk_level(self) -> str:
        """Calculate current risk level"""
        try:
            daily_pnl = self.status["daily_pnl"]
            max_daily_loss = self.config.get("trading", {}).get("risk", {}).get("max_daily_loss", 0.02)
            
            if daily_pnl <= -max_daily_loss:
                return "CRITICAL"
            elif daily_pnl <= -max_daily_loss * 0.5:
                return "HIGH"
            elif daily_pnl <= 0:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "UNKNOWN"
    
    def get_chart_data(self, symbol: str, timeframe: str = "1H", bars: int = 100) -> Dict:
        """Get chart data for a symbol"""
        try:
            if trading_bot is None:
                return {}
            
            # Get market data from bot
            df = trading_bot.get_market_data(symbol, timeframe, bars)
            
            if df.empty:
                return {}
            
            # Create candlestick chart
            candlestick = go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            )
            
            # Add technical indicators
            traces = [candlestick]
            
            # RSI
            if 'rsi' in df.columns:
                rsi_trace = go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    name='RSI',
                    yaxis='y2'
                )
                traces.append(rsi_trace)
            
            # MACD
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_trace = go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    name='MACD',
                    line=dict(color='blue')
                )
                macd_signal_trace = go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    name='MACD Signal',
                    line=dict(color='red')
                )
                traces.append(macd_trace)
                traces.append(macd_signal_trace)
            
            # Bollinger Bands
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_upper_trace = go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    name='BB Upper',
                    line=dict(color='gray', dash='dash')
                )
                bb_lower_trace = go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    name='BB Lower',
                    line=dict(color='gray', dash='dash')
                )
                traces.append(bb_upper_trace)
                traces.append(bb_lower_trace)
            
            # Layout
            layout = go.Layout(
                title=f'{symbol} - {timeframe}',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Price'),
                yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 100]),
                height=600
            )
            
            # Create figure
            fig = go.Figure(data=traces, layout=layout)
            
            # Convert to JSON
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "chart_data": chart_json,
                "last_price": float(df['close'].iloc[-1]),
                "price_change": float(df['close'].pct_change().iloc[-1]),
                "volume": int(df['tick_volume'].iloc[-1])
            }
            
        except Exception as e:
            print(f"Error getting chart data: {e}")
            return {}
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        try:
            if trading_bot is None:
                return self.trade_history
            
            # Get active trades from bot
            active_trades = trading_bot.active_trades
            
            trades = []
            for trade_id, trade in active_trades.items():
                trades.append({
                    "id": trade_id,
                    "symbol": trade.get("symbol", ""),
                    "type": trade.get("type", ""),
                    "action": "BUY" if trade.get("type") == "buy" else "SELL",
                    "size": trade.get("size", 0.0),
                    "entry_price": trade.get("entry_price", 0.0),
                    "entry_time": trade.get("entry_time", datetime.now()).isoformat(),
                    "current_price": trade.get("current_price", 0.0),
                    "pnl": trade.get("profit", 0.0),
                    "status": "OPEN"
                })
            
            return trades
            
        except Exception as e:
            print(f"Error getting trade history: {e}")
            return []
    
    def update_config(self, new_config: Dict) -> bool:
        """Update bot configuration"""
        try:
            # Merge with existing config
            self.config.update(new_config)
            
            # Save to file
            with open('bot_config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
            
            return True
            
        except Exception as e:
            print(f"Error updating config: {e}")
            return False

# Create dashboard manager
dashboard_manager = DashboardManager()

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """Get bot status"""
    status = dashboard_manager.get_bot_status()
    return jsonify(status)

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start the bot"""
    success = dashboard_manager.start_bot()
    return jsonify({"success": success})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop the bot"""
    success = dashboard_manager.stop_bot()
    return jsonify({"success": success})

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """Get chart data for a symbol"""
    timeframe = request.args.get('timeframe', '1H')
    bars = int(request.args.get('bars', 100))
    
    chart_data = dashboard_manager.get_chart_data(symbol, timeframe, bars)
    return jsonify(chart_data)

@app.route('/api/trades')
def api_trades():
    """Get trade history"""
    trades = dashboard_manager.get_trade_history()
    return jsonify(trades)

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify(dashboard_manager.config)
    else:
        new_config = request.json
        success = dashboard_manager.update_config(new_config)
        return jsonify({"success": success})

@app.route('/api/symbols')
def api_symbols():
    """Get available symbols"""
    symbols = dashboard_manager.config.get("mt5", {}).get("symbols", [])
    return jsonify(symbols)

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status_update', dashboard_manager.get_bot_status())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def background_updates():
    """Background task to send updates to clients"""
    while True:
        try:
            # Get updated status
            status = dashboard_manager.get_bot_status()
            
            # Emit to all connected clients
            socketio.emit('status_update', status)
            
            # Wait before next update
            time.sleep(5)
            
        except Exception as e:
            print(f"Error in background updates: {e}")
            time.sleep(5)

# Start background updates
update_thread = threading.Thread(target=background_updates)
update_thread.daemon = True
update_thread.start()

# Create templates directory and HTML file
os.makedirs('templates', exist_ok=True)

# Create dashboard HTML template
dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex Trading Bot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .status-card h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .status-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .status-running { color: #28a745; }
        .status-stopped { color: #dc3545; }
        .status-positive { color: #28a745; }
        .status-negative { color: #dc3545; }
        .controls {
            text-align: center;
            margin-bottom: 20px;
        }
        .btn {
            padding: 12px 24px;
            margin: 0 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .btn-start {
            background-color: #28a745;
            color: white;
        }
        .btn-start:hover { background-color: #218838; }
        .btn-stop {
            background-color: #dc3545;
            color: white;
        }
        .btn-stop:hover { background-color: #c82333; }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .trades-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .trades-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .trades-table th,
        .trades-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .trades-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .symbol-selector {
            margin-bottom: 20px;
            text-align: center;
        }
        .symbol-selector select {
            padding: 8px 16px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        .timeframe-selector {
            display: inline-block;
            margin-left: 10px;
        }
        .timeframe-btn {
            padding: 6px 12px;
            margin: 0 2px;
            border: 1px solid #ddd;
            background: white;
            cursor: pointer;
            border-radius: 3px;
        }
        .timeframe-btn.active {
            background-color: #007bff;
            color: white;
            border-color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Advanced Forex Trading Bot Dashboard</h1>
            <p>Real-time monitoring and control for your automated trading system</p>
        </div>

        <div class="controls">
            <button id="startBtn" class="btn btn-start">Start Bot</button>
            <button id="stopBtn" class="btn btn-stop" disabled>Stop Bot</button>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>Bot Status</h3>
                <div id="botStatus" class="status-value status-stopped">STOPPED</div>
            </div>
            <div class="status-card">
                <h3>Active Trades</h3>
                <div id="activeTrades" class="status-value">0</div>
            </div>
            <div class="status-card">
                <h3>Total P&L</h3>
                <div id="totalPnl" class="status-value">$0.00</div>
            </div>
            <div class="status-card">
                <h3>Daily P&L</h3>
                <div id="dailyPnl" class="status-value">$0.00</div>
            </div>
            <div class="status-card">
                <h3>Risk Level</h3>
                <div id="riskLevel" class="status-value">LOW</div>
            </div>
            <div class="status-card">
                <h3>Last Update</h3>
                <div id="lastUpdate" class="status-value">Never</div>
            </div>
        </div>

        <div class="symbol-selector">
            <label for="symbolSelect">Select Symbol:</label>
            <select id="symbolSelect">
                <option value="EURUSD">EURUSD</option>
                <option value="GBPUSD">GBPUSD</option>
                <option value="USDJPY">USDJPY</option>
                <option value="AUDUSD">AUDUSD</option>
            </select>
            <div class="timeframe-selector">
                <button class="timeframe-btn active" data-timeframe="M5">M5</button>
                <button class="timeframe-btn" data-timeframe="M15">M15</button>
                <button class="timeframe-btn" data-timeframe="M30">M30</button>
                <button class="timeframe-btn" data-timeframe="H1">H1</button>
                <button class="timeframe-btn" data-timeframe="H4">H4</button>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h3>Price Chart</h3>
                <div id="priceChart"></div>
            </div>
            <div class="chart-container">
                <h3>Technical Indicators</h3>
                <div id="indicatorsChart"></div>
            </div>
        </div>

        <div class="trades-container">
            <h3>Active Trades</h3>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Action</th>
                        <th>Size</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>P&L</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="tradesTableBody">
                    <tr>
                        <td colspan="8" style="text-align: center;">No active trades</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // DOM elements
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const botStatus = document.getElementById('botStatus');
        const activeTrades = document.getElementById('activeTrades');
        const totalPnl = document.getElementById('totalPnl');
        const dailyPnl = document.getElementById('dailyPnl');
        const riskLevel = document.getElementById('riskLevel');
        const lastUpdate = document.getElementById('lastUpdate');
        const symbolSelect = document.getElementById('symbolSelect');
        const priceChart = document.getElementById('priceChart');
        const indicatorsChart = document.getElementById('indicatorsChart');
        const tradesTableBody = document.getElementById('tradesTableBody');
        
        // Current timeframe
        let currentTimeframe = 'M5';
        
        // Event listeners
        startBtn.addEventListener('click', startBot);
        stopBtn.addEventListener('click', stopBot);
        symbolSelect.addEventListener('change', updateCharts);
        
        // Timeframe buttons
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                currentTimeframe = this.dataset.timeframe;
                updateCharts();
            });
        });
        
        // Socket events
        socket.on('status_update', updateStatus);
        socket.on('connect', function() {
            console.log('Connected to server');
            updateStatus();
        });
        
        // Functions
        function startBot() {
            fetch('/api/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Bot started successfully');
                    } else {
                        console.log('Failed to start bot');
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        function stopBot() {
            fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Bot stopped successfully');
                    } else {
                        console.log('Failed to stop bot');
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update status display
                    botStatus.textContent = data.running ? 'RUNNING' : 'STOPPED';
                    botStatus.className = 'status-value ' + (data.running ? 'status-running' : 'status-stopped');
                    
                    activeTrades.textContent = data.active_trades || 0;
                    totalPnl.textContent = '$' + (data.total_pnl || 0).toFixed(2);
                    dailyPnl.textContent = '$' + (data.daily_pnl || 0).toFixed(2);
                    riskLevel.textContent = data.risk_level || 'UNKNOWN';
                    
                    if (data.last_update) {
                        const date = new Date(data.last_update);
                        lastUpdate.textContent = date.toLocaleString();
                    }
                    
                    // Update button states
                    startBtn.disabled = data.running;
                    stopBtn.disabled = !data.running;
                    
                    // Update charts if bot is running
                    if (data.running) {
                        updateCharts();
                        updateTrades();
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        function updateCharts() {
            const symbol = symbolSelect.value;
            
            fetch(`/api/chart/${symbol}?timeframe=${currentTimeframe}&bars=100`)
                .then(response => response.json())
                .then(data => {
                    if (data.chart_data) {
                        const chartData = JSON.parse(data.chart_data);
                        Plotly.newPlot('priceChart', chartData.data, chartData.layout);
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        function updateTrades() {
            fetch('/api/trades')
                .then(response => response.json())
                .then(trades => {
                    if (trades.length === 0) {
                        tradesTableBody.innerHTML = '<tr><td colspan="8" style="text-align: center;">No active trades</td></tr>';
                        return;
                    }
                    
                    tradesTableBody.innerHTML = trades.map(trade => `
                        <tr>
                            <td>${trade.symbol}</td>
                            <td>${trade.type}</td>
                            <td>${trade.action}</td>
                            <td>${trade.size}</td>
                            <td>${trade.entry_price}</td>
                            <td>${trade.current_price}</td>
                            <td class="${trade.pnl >= 0 ? 'status-positive' : 'status-negative'}">$${trade.pnl.toFixed(2)}</td>
                            <td>${trade.status}</td>
                        </tr>
                    `).join('');
                })
                .catch(error => console.error('Error:', error));
        }
        
        // Initial load
        updateStatus();
        updateCharts();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            if (botStatus.textContent === 'RUNNING') {
                updateStatus();
            }
        }, 30000);
    </script>
</body>
</html>'''

# Write the HTML template
with open('templates/dashboard.html', 'w') as f:
    f.write(dashboard_html)

if __name__ == '__main__':
    print("Starting Trading Dashboard...")
    print("Open your browser and go to: http://localhost:8080")
    
    # Start the Flask app
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)