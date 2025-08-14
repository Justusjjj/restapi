from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from forex_bot import ForexTradingBot
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global bot instance
bot = None
bot_status = "stopped"
bot_thread = None

def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    np.random.seed(42)
    
    # Generate sample price data
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        data.append({
            'timestamp': date.isoformat(),
            'open': price * (1 + np.random.normal(0, 0.0002)),
            'high': price * (1 + abs(np.random.normal(0, 0.0005))),
            'low': price * (1 - abs(np.random.normal(0, 0.0005))),
            'close': price,
            'volume': np.random.randint(1000, 10000)
        })
    
    return data

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get bot status"""
    global bot, bot_status
    
    if bot is None:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Bot not initialized'
        })
    
    try:
        balance = bot.get_account_balance()
        positions = bot.get_open_positions()
        
        return jsonify({
            'status': bot_status,
            'balance': balance,
            'open_positions': len(positions),
            'total_trades': len(bot.trades),
            'last_update': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global bot, bot_status, bot_thread
    
    try:
        if bot is None:
            # Initialize bot with demo mode
            bot = ForexTradingBot()
            bot_status = "running"
            
            # Start bot in background thread
            bot_thread = threading.Thread(target=run_bot_background)
            bot_thread.daemon = True
            bot_thread.start()
            
            return jsonify({'success': True, 'message': 'Bot started successfully'})
        else:
            return jsonify({'success': False, 'message': 'Bot already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_status
    
    try:
        bot_status = "stopped"
        return jsonify({'success': True, 'message': 'Bot stopped successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analysis/<symbol>')
def get_analysis(symbol):
    """Get strategy analysis for a symbol"""
    global bot
    
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    try:
        analysis = bot.run_strategy_analysis(symbol)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    global bot
    
    if bot is None:
        return jsonify([])
    
    try:
        # Return last 50 trades
        recent_trades = bot.trades[-50:] if len(bot.trades) > 50 else bot.trades
        
        # Convert datetime objects to strings
        for trade in recent_trades:
            if 'timestamp' in trade:
                trade['timestamp'] = trade['timestamp'].isoformat()
        
        return jsonify(recent_trades)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/positions')
def get_positions():
    """Get current open positions"""
    global bot
    
    if bot is None:
        return jsonify([])
    
    try:
        positions = bot.get_open_positions()
        return jsonify(positions)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market_data/<symbol>')
def get_market_data(symbol):
    """Get market data for a symbol"""
    try:
        # For demo purposes, generate sample data
        # In production, this would fetch real market data
        data = create_sample_data()
        
        # Filter to last 100 data points
        return jsonify(data[-100:])
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update bot settings"""
    global bot
    
    if request.method == 'GET':
        if bot is None:
            return jsonify({})
        
        return jsonify({
            'symbols': bot.symbols,
            'timeframe': bot.timeframe,
            'position_size': bot.position_size,
            'max_positions': bot.max_positions,
            'stop_loss_pips': bot.stop_loss_pips,
            'take_profit_pips': bot.take_profit_pips,
            'max_daily_loss': bot.max_daily_loss,
            'max_portfolio_risk': bot.max_portfolio_risk
        })
    
    elif request.method == 'POST':
        if bot is None:
            return jsonify({'success': False, 'message': 'Bot not initialized'})
        
        try:
            data = request.json
            
            # Update bot settings
            if 'symbols' in data:
                bot.symbols = data['symbols']
            if 'timeframe' in data:
                bot.timeframe = data['timeframe']
            if 'position_size' in data:
                bot.position_size = float(data['position_size'])
            if 'max_positions' in data:
                bot.max_positions = int(data['max_positions'])
            if 'stop_loss_pips' in data:
                bot.stop_loss_pips = int(data['stop_loss_pips'])
            if 'take_profit_pips' in data:
                bot.take_profit_pips = int(data['take_profit_pips'])
            if 'max_daily_loss' in data:
                bot.max_daily_loss = float(data['max_daily_loss'])
            if 'max_portfolio_risk' in data:
                bot.max_portfolio_risk = float(data['max_portfolio_risk'])
            
            return jsonify({'success': True, 'message': 'Settings updated successfully'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

def run_bot_background():
    """Run bot in background thread"""
    global bot, bot_status
    
    try:
        while bot_status == "running":
            # Execute trading cycle
            bot.execute_trades()
            
            # Emit status update via WebSocket
            try:
                balance = bot.get_account_balance()
                positions = bot.get_open_positions()
                
                socketio.emit('bot_update', {
                    'status': bot_status,
                    'balance': balance,
                    'open_positions': len(positions),
                    'total_trades': len(bot.trades),
                    'timestamp': datetime.now().isoformat()
                })
            except:
                pass
            
            # Wait before next cycle
            time.sleep(60)  # 1 minute intervals for demo
            
    except Exception as e:
        print(f"Bot error: {e}")
        bot_status = "error"

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    create_dashboard_template()
    
    print("Starting Forex Trading Bot Dashboard...")
    print("Open http://localhost:5000 in your browser")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

def create_dashboard_template():
    """Create the HTML dashboard template"""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .status-card {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .status-card h3 {
            color: white;
        }
        
        .status-indicator {
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
            background: #ff4757;
        }
        
        .status-indicator.running {
            background: #2ed573;
        }
        
        .status-indicator.stopped {
            background: #ffa502;
        }
        
        .metric {
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .control-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }
        
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: #2ed573;
            color: white;
        }
        
        .btn-primary:hover {
            background: #26d0a8;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: #ff4757;
            color: white;
        }
        
        .btn-danger:hover {
            background: #ff3742;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #747d8c;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #57606f;
            transform: translateY(-2px);
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .chart-container h3 {
            color: #667eea;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .trades-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .trades-table th,
        .trades-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        .trades-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #667eea;
        }
        
        .trades-table tr:hover {
            background: #f8f9fa;
        }
        
        .positive {
            color: #2ed573;
        }
        
        .negative {
            color: #ff4757;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .control-buttons {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 200px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Forex Trading Bot</h1>
            <p>Advanced algorithmic trading with real-time monitoring</p>
        </div>
        
        <div class="dashboard-grid">
            <div class="card status-card">
                <h3>Bot Status</h3>
                <div class="status-indicator" id="statusIndicator"></div>
                <span id="statusText">Stopped</span>
                <div class="metric" id="balance">$0.00</div>
                <div class="metric-label">Account Balance</div>
                <div class="control-buttons">
                    <button class="btn btn-primary" onclick="startBot()">Start Bot</button>
                    <button class="btn btn-danger" onclick="stopBot()">Stop Bot</button>
                </div>
            </div>
            
            <div class="card">
                <h3>Trading Statistics</h3>
                <div class="metric" id="totalTrades">0</div>
                <div class="metric-label">Total Trades</div>
                <div class="metric" id="openPositions">0</div>
                <div class="metric-label">Open Positions</div>
            </div>
            
            <div class="card">
                <h3>Performance</h3>
                <div class="metric" id="winRate">0%</div>
                <div class="metric-label">Win Rate</div>
                <div class="metric" id="totalPnL">$0.00</div>
                <div class="metric-label">Total P&L</div>
            </div>
            
            <div class="card">
                <h3>Risk Management</h3>
                <div class="metric" id="maxDrawdown">0%</div>
                <div class="metric-label">Max Drawdown</div>
                <div class="metric" id="sharpeRatio">0.00</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Equity Curve</h3>
            <canvas id="equityChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Recent Trades</h3>
            <div id="tradesContainer">
                <div class="loading">Loading trades...</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Strategy Analysis</h3>
            <div class="control-buttons">
                <button class="btn btn-secondary" onclick="analyzeSymbol('EUR/USD')">EUR/USD</button>
                <button class="btn btn-secondary" onclick="analyzeSymbol('GBP/USD')">GBP/USD</button>
                <button class="btn btn-secondary" onclick="analyzeSymbol('USD/JPY')">USD/JPY</button>
                <button class="btn btn-secondary" onclick="analyzeSymbol('AUD/USD')">AUD/USD</button>
            </div>
            <div id="analysisContainer">
                <div class="loading">Select a symbol to analyze</div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // Chart instances
        let equityChart = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeDashboard();
            setupWebSocket();
        });
        
        function initializeDashboard() {
            updateStatus();
            loadTrades();
            initializeEquityChart();
        }
        
        function setupWebSocket() {
            socket.on('connect', function() {
                console.log('Connected to server');
            });
            
            socket.on('bot_update', function(data) {
                updateDashboard(data);
            });
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateDashboard(data);
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }
        
        function updateDashboard(data) {
            // Update status
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            
            statusIndicator.className = 'status-indicator ' + (data.status || 'stopped');
            statusText.textContent = data.status || 'Stopped';
            
            // Update metrics
            if (data.balance !== undefined) {
                document.getElementById('balance').textContent = '$' + parseFloat(data.balance).toFixed(2);
            }
            
            if (data.total_trades !== undefined) {
                document.getElementById('totalTrades').textContent = data.total_trades;
            }
            
            if (data.open_positions !== undefined) {
                document.getElementById('openPositions').textContent = data.open_positions;
            }
        }
        
        function startBot() {
            fetch('/api/start_bot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus();
                } else {
                    alert('Error starting bot: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error starting bot');
            });
        }
        
        function stopBot() {
            fetch('/api/stop_bot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus();
                } else {
                    alert('Error stopping bot: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error stopping bot');
            });
        }
        
        function loadTrades() {
            fetch('/api/trades')
                .then(response => response.json())
                .then(trades => {
                    displayTrades(trades);
                })
                .catch(error => {
                    console.error('Error loading trades:', error);
                });
        }
        
        function displayTrades(trades) {
            const container = document.getElementById('tradesContainer');
            
            if (!trades || trades.length === 0) {
                container.innerHTML = '<div class="loading">No trades found</div>';
                return;
            }
            
            let html = '<table class="trades-table">';
            html += '<thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Price</th><th>Size</th><th>P&L</th></tr></thead><tbody>';
            
            // Show last 10 trades
            const recentTrades = trades.slice(-10).reverse();
            
            recentTrades.forEach(trade => {
                const time = new Date(trade.timestamp).toLocaleString();
                const pnl = trade.pnl || 0;
                const pnlClass = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '';
                const pnlText = pnl !== 0 ? '$' + pnl.toFixed(2) : '-';
                
                html += `<tr>
                    <td>${time}</td>
                    <td>${trade.symbol || 'N/A'}</td>
                    <td>${trade.side || 'N/A'}</td>
                    <td>${trade.price ? trade.price.toFixed(5) : 'N/A'}</td>
                    <td>${trade.size || 'N/A'}</td>
                    <td class="${pnlClass}">${pnlText}</td>
                </tr>`;
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        function initializeEquityChart() {
            const ctx = document.getElementById('equityChart').getContext('2d');
            
            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Account Balance',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: 'rgba(0,0,0,0.1)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(0,0,0,0.1)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
        
        function analyzeSymbol(symbol) {
            const container = document.getElementById('analysisContainer');
            container.innerHTML = '<div class="loading">Analyzing ' + symbol + '...</div>';
            
            fetch('/api/analysis/' + symbol)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        container.innerHTML = '<div class="loading">Error: ' + data.error + '</div>';
                        return;
                    }
                    
                    displayAnalysis(data);
                })
                .catch(error => {
                    console.error('Error analyzing symbol:', error);
                    container.innerHTML = '<div class="loading">Error analyzing symbol</div>';
                });
        }
        
        function displayAnalysis(analysis) {
            const container = document.getElementById('analysisContainer');
            
            let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">';
            
            // Current price
            html += `<div class="card">
                <h4>Current Price</h4>
                <div class="metric">${analysis.current_price ? analysis.current_price.toFixed(5) : 'N/A'}</div>
            </div>`;
            
            // Combined signal
            const combinedSignal = analysis.signals?.combined;
            if (combinedSignal) {
                const signalColor = combinedSignal.signal === 'BUY' ? '#2ed573' : 
                                  combinedSignal.signal === 'SELL' ? '#ff4757' : '#747d8c';
                
                html += `<div class="card">
                    <h4>Signal</h4>
                    <div class="metric" style="color: ${signalColor}">${combinedSignal.signal}</div>
                    <div class="metric-label">Confidence: ${(combinedSignal.confidence * 100).toFixed(1)}%</div>
                </div>`;
            }
            
            // RSI
            const rsi = analysis.indicators?.rsi;
            if (rsi) {
                const rsiColor = rsi < 30 ? '#2ed573' : rsi > 70 ? '#ff4757' : '#747d8c';
                html += `<div class="card">
                    <h4>RSI</h4>
                    <div class="metric" style="color: ${rsiColor}">${rsi.toFixed(1)}</div>
                </div>`;
            }
            
            // MACD
            const macd = analysis.indicators?.macd;
            if (macd) {
                const macdColor = macd > 0 ? '#2ed573' : '#ff4757';
                html += `<div class="card">
                    <h4>MACD</h4>
                    <div class="metric" style="color: ${macdColor}">${macd.toFixed(6)}</div>
                </div>`;
            }
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        // Auto-refresh every 30 seconds
        setInterval(updateStatus, 30000);
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(template_content)