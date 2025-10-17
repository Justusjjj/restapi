#!/usr/bin/env python3
"""
🌐 Enhanced Trading Dashboard
Real-time monitoring and control interface for the Enhanced Forex Trading System

Features:
- Real-time market data visualization
- Advanced charting with technical indicators
- Pattern recognition display
- Performance analytics
- Risk monitoring
- Trade management
- News sentiment analysis
- Market regime detection
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
import plotly.express as px

# Trading system integration
from enhanced_forex_trading_system import EnhancedForexTradingSystem

# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global trading system instance
trading_system = None
system_thread = None

class EnhancedDashboardManager:
    def __init__(self):
        self.trading_system = None
        self.config = self.load_config()
        self.status = {
            "running": False,
            "last_update": None,
            "active_trades": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "risk_level": "LOW",
            "market_regime": "unknown",
            "patterns_detected": 0,
            "sentiment_score": 0.0
        }
        self.chart_data = {}
        self.performance_data = []
        self.news_data = []
        
    def load_config(self) -> Dict:
        """Load dashboard configuration"""
        return {
            "update_interval": 5,
            "chart_periods": 100,
            "risk_colors": {
                "LOW": "#28a745",
                "MEDIUM": "#ffc107", 
                "HIGH": "#dc3545"
            },
            "regime_colors": {
                "trending": "#007bff",
                "ranging": "#6c757d",
                "volatile": "#fd7e14",
                "mixed": "#6f42c1"
            }
        }
    
    def start_trading_system(self):
        """Start the trading system"""
        global trading_system, system_thread
        
        if trading_system is None:
            trading_system = EnhancedForexTradingSystem()
            system_thread = threading.Thread(target=trading_system.start)
            system_thread.daemon = True
            system_thread.start()
            
            self.trading_system = trading_system
            self.status["running"] = True
            return True
        return False
    
    def stop_trading_system(self):
        """Stop the trading system"""
        global trading_system
        
        if trading_system:
            trading_system.stop()
            self.status["running"] = False
            return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if self.trading_system:
            status = self.trading_system.get_status()
            self.status.update(status)
        
        return self.status
    
    def get_chart_data(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get chart data for a symbol"""
        if not self.trading_system:
            return {"error": "Trading system not running"}
        
        try:
            # Get market data
            df = self.trading_system.get_market_data(symbol, timeframe, 100)
            if df.empty:
                return {"error": "No data available"}
            
            # Calculate technical indicators
            df = self.trading_system.calculate_technical_indicators(df)
            
            # Create candlestick chart
            candlestick = go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            )
            
            # Create technical indicator traces
            traces = [candlestick]
            
            # RSI
            if 'rsi' in df.columns:
                rsi_trace = go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    yaxis='y2',
                    line=dict(color='purple', width=2)
                )
                traces.append(rsi_trace)
            
            # Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                bb_upper = go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(128,128,128,0.5)', width=1),
                    showlegend=False
                )
                bb_middle = go.Scatter(
                    x=df.index,
                    y=df['bb_middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='rgba(128,128,128,0.8)', width=1),
                    showlegend=False
                )
                bb_lower = go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(128,128,128,0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                )
                traces.extend([bb_upper, bb_middle, bb_lower])
            
            # MACD
            if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                macd_trace = go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    mode='lines',
                    name='MACD',
                    yaxis='y3',
                    line=dict(color='blue', width=2)
                )
                macd_signal_trace = go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    mode='lines',
                    name='MACD Signal',
                    yaxis='y3',
                    line=dict(color='red', width=2)
                )
                macd_hist_trace = go.Bar(
                    x=df.index,
                    y=df['macd_histogram'],
                    name='MACD Histogram',
                    yaxis='y3',
                    marker_color='gray'
                )
                traces.extend([macd_trace, macd_signal_trace, macd_hist_trace])
            
            # Create layout
            layout = go.Layout(
                title=f'{symbol} - {timeframe} Chart',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Price'),
                yaxis2=dict(
                    title='RSI',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                yaxis3=dict(
                    title='MACD',
                    overlaying='y',
                    side='right',
                    anchor='free',
                    position=0.95
                ),
                hovermode='x unified',
                showlegend=True,
                height=600
            )
            
            # Create figure
            fig = go.Figure(data=traces, layout=layout)
            
            return {
                "chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                "data_points": len(df),
                "last_price": df['close'].iloc[-1],
                "price_change": df['close'].pct_change().iloc[-1] * 100
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_pattern_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get pattern analysis for a symbol"""
        if not self.trading_system:
            return {"error": "Trading system not running"}
        
        try:
            # Get patterns from trading system
            patterns = self.trading_system.pattern_signals.get(symbol, {})
            
            # Format pattern data for display
            pattern_summary = []
            
            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, dict) and pattern_data.get('detected'):
                    pattern_summary.append({
                        "name": pattern_name.replace('_', ' ').title(),
                        "detected": True,
                        "confidence": pattern_data.get('confidence', 0),
                        "type": pattern_data.get('type', ''),
                        "strength": pattern_data.get('strength', 0)
                    })
            
            return {
                "patterns": pattern_summary,
                "total_patterns": len(pattern_summary),
                "high_confidence": len([p for p in pattern_summary if p['confidence'] > 0.7])
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_sentiment_analysis(self) -> Dict[str, Any]:
        """Get sentiment analysis data"""
        if not self.trading_system:
            return {"error": "Trading system not running"}
        
        try:
            sentiment_data = self.trading_system.sentiment_scores
            
            # Calculate overall sentiment
            sentiments = [data.get('sentiment', 0) for data in sentiment_data.values()]
            confidences = [data.get('confidence', 0) for data in sentiment_data.values()]
            
            if sentiments:
                overall_sentiment = np.mean(sentiments)
                overall_confidence = np.mean(confidences)
                
                # Categorize sentiment
                if overall_sentiment > 0.3:
                    sentiment_label = "Bullish"
                    sentiment_color = "#28a745"
                elif overall_sentiment < -0.3:
                    sentiment_label = "Bearish"
                    sentiment_color = "#dc3545"
                else:
                    sentiment_label = "Neutral"
                    sentiment_color = "#6c757d"
                
                return {
                    "overall_sentiment": overall_sentiment,
                    "overall_confidence": overall_confidence,
                    "sentiment_label": sentiment_label,
                    "sentiment_color": sentiment_color,
                    "symbol_sentiments": sentiment_data
                }
            else:
                return {
                    "overall_sentiment": 0,
                    "overall_confidence": 0,
                    "sentiment_label": "Unknown",
                    "sentiment_color": "#6c757d",
                    "symbol_sentiments": {}
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.trading_system:
            return {"error": "Trading system not running"}
        
        try:
            metrics = self.trading_system.performance_metrics
            
            # Calculate additional metrics
            total_trades = metrics.get('total_trades', 0)
            winning_trades = metrics.get('winning_trades', 0)
            total_pnl = metrics.get('total_pnl', 0)
            
            # Calculate win rate
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate average win/loss
            trade_history = self.trading_system.trade_history
            if trade_history:
                pnls = [t.get('pnl', 0) for t in trade_history if 'pnl' in t]
                if pnls:
                    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
                    avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
                else:
                    avg_win = avg_loss = 0
            else:
                avg_win = avg_loss = 0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "active_trades": metrics.get('active_trades', 0),
                "last_update": metrics.get('last_update', datetime.now())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        if not self.trading_system:
            return {"error": "Trading system not running"}
        
        try:
            # Calculate current risk level
            active_trades = len(self.trading_system.active_trades)
            total_pnl = self.trading_system.performance_metrics.get('total_pnl', 0)
            
            # Simple risk calculation
            if active_trades > 10:
                risk_level = "HIGH"
            elif active_trades > 5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Calculate position sizes
            position_sizes = []
            for trade in self.trading_system.active_trades.values():
                position_sizes.append(trade.get('size', 0))
            
            total_exposure = sum(position_sizes)
            max_position = max(position_sizes) if position_sizes else 0
            
            return {
                "risk_level": risk_level,
                "active_trades": active_trades,
                "total_exposure": total_exposure,
                "max_position_size": max_position,
                "avg_position_size": total_exposure / active_trades if active_trades > 0 else 0,
                "total_pnl": total_pnl,
                "risk_color": self.config["risk_colors"].get(risk_level, "#6c757d")
            }
            
        except Exception as e:
            return {"error": str(e)}

# Initialize dashboard manager
dashboard_manager = EnhancedDashboardManager()

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('enhanced_dashboard.html')

@app.route('/api/status')
def api_status():
    """Get system status"""
    return jsonify(dashboard_manager.get_system_status())

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """Get chart data for symbol"""
    timeframe = request.args.get('timeframe', '1h')
    return jsonify(dashboard_manager.get_chart_data(symbol, timeframe))

@app.route('/api/patterns/<symbol>')
def api_patterns(symbol):
    """Get pattern analysis for symbol"""
    return jsonify(dashboard_manager.get_pattern_analysis(symbol))

@app.route('/api/sentiment')
def api_sentiment():
    """Get sentiment analysis"""
    return jsonify(dashboard_manager.get_sentiment_analysis())

@app.route('/api/performance')
def api_performance():
    """Get performance metrics"""
    return jsonify(dashboard_manager.get_performance_metrics())

@app.route('/api/risk')
def api_risk():
    """Get risk metrics"""
    return jsonify(dashboard_manager.get_risk_metrics())

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start trading system"""
    success = dashboard_manager.start_trading_system()
    return jsonify({"success": success})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop trading system"""
    success = dashboard_manager.stop_trading_system()
    return jsonify({"success": success})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', dashboard_manager.get_system_status())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle update request from client"""
    emit('status_update', dashboard_manager.get_system_status())

# Background task for real-time updates
def background_updates():
    """Send periodic updates to connected clients"""
    while True:
        try:
            if dashboard_manager.trading_system:
                # Send status update
                status = dashboard_manager.get_system_status()
                socketio.emit('status_update', status)
                
                # Send performance update
                performance = dashboard_manager.get_performance_metrics()
                socketio.emit('performance_update', performance)
                
                # Send risk update
                risk = dashboard_manager.get_risk_metrics()
                socketio.emit('risk_update', risk)
            
            time.sleep(dashboard_manager.config["update_interval"])
        except Exception as e:
            print(f"Error in background updates: {e}")
            time.sleep(10)

# Start background task
update_thread = threading.Thread(target=background_updates)
update_thread.daemon = True
update_thread.start()

if __name__ == '__main__':
    print("🌐 Starting Enhanced Trading Dashboard...")
    print("📱 Open your browser and go to: http://localhost:8080")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the enhanced dashboard HTML template
    create_enhanced_dashboard_template()
    
    # Start the dashboard
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)

def create_enhanced_dashboard_template():
    """Create the enhanced dashboard HTML template"""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Forex Trading Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .status-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
        }
        .pattern-badge {
            display: inline-block;
            padding: 5px 10px;
            margin: 2px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .pattern-detected {
            background-color: #28a745;
            color: white;
        }
        .pattern-high-confidence {
            background-color: #007bff;
            color: white;
        }
        .sentiment-positive {
            color: #28a745;
        }
        .sentiment-negative {
            color: #dc3545;
        }
        .sentiment-neutral {
            color: #6c757d;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #6c757d;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-running {
            background-color: #28a745;
            animation: pulse 2s infinite;
        }
        .status-stopped {
            background-color: #dc3545;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <div class="row">
            <div class="col-12">
                <div class="status-card">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <h2><i class="fas fa-chart-line"></i> Enhanced Forex Trading Dashboard</h2>
                            <p class="mb-0">Real-time market analysis and trading system monitoring</p>
                        </div>
                        <div class="col-md-6 text-end">
                            <div class="mb-2">
                                <span id="status-indicator" class="status-indicator status-stopped"></span>
                                <span id="system-status">System Stopped</span>
                            </div>
                            <div>
                                <button id="start-btn" class="btn btn-success me-2" onclick="startSystem()">
                                    <i class="fas fa-play"></i> Start System
                                </button>
                                <button id="stop-btn" class="btn btn-danger" onclick="stopSystem()" disabled>
                                    <i class="fas fa-stop"></i> Stop System
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="row">
            <!-- Left Column -->
            <div class="col-lg-8">
                <!-- Charts -->
                <div class="chart-container">
                    <h4><i class="fas fa-chart-candlestick"></i> Market Charts</h4>
                    <div class="mb-3">
                        <select id="symbol-select" class="form-select d-inline-block w-auto me-2">
                            <option value="EUR/USDT">EUR/USDT</option>
                            <option value="GBP/USDT">GBP/USDT</option>
                            <option value="USD/JPY">USD/JPY</option>
                            <option value="AUD/USDT">AUD/USDT</option>
                            <option value="USD/CAD">USD/CAD</option>
                            <option value="NZD/USDT">NZD/USDT</option>
                        </select>
                        <select id="timeframe-select" class="form-select d-inline-block w-auto">
                            <option value="1h">1 Hour</option>
                            <option value="4h">4 Hours</option>
                            <option value="1d">1 Day</option>
                        </select>
                    </div>
                    <div id="chart-container" class="loading">
                        <i class="fas fa-spinner fa-spin"></i> Loading chart...
                    </div>
                </div>

                <!-- Pattern Analysis -->
                <div class="metric-card">
                    <h4><i class="fas fa-search"></i> Pattern Analysis</h4>
                    <div id="pattern-analysis" class="loading">
                        <i class="fas fa-spinner fa-spin"></i> Loading patterns...
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div class="col-lg-4">
                <!-- Performance Metrics -->
                <div class="metric-card">
                    <h4><i class="fas fa-trophy"></i> Performance</h4>
                    <div id="performance-metrics" class="loading">
                        <i class="fas fa-spinner fa-spin"></i> Loading metrics...
                    </div>
                </div>

                <!-- Risk Metrics -->
                <div class="metric-card">
                    <h4><i class="fas fa-shield-alt"></i> Risk Management</h4>
                    <div id="risk-metrics" class="loading">
                        <i class="fas fa-spinner fa-spin"></i> Loading risk data...
                    </div>
                </div>

                <!-- Sentiment Analysis -->
                <div class="metric-card">
                    <h4><i class="fas fa-heart"></i> Market Sentiment</h4>
                    <div id="sentiment-analysis" class="loading">
                        <i class="fas fa-spinner fa-spin"></i> Loading sentiment...
                    </div>
                </div>

                <!-- Active Trades -->
                <div class="metric-card">
                    <h4><i class="fas fa-exchange-alt"></i> Active Trades</h4>
                    <div id="active-trades" class="loading">
                        <i class="fas fa-spinner fa-spin"></i> Loading trades...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // WebSocket connection
        const socket = io();
        
        // Global variables
        let currentSymbol = 'EUR/USDT';
        let currentTimeframe = '1h';
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadInitialData();
            setupEventListeners();
        });
        
        // Setup event listeners
        function setupEventListeners() {
            document.getElementById('symbol-select').addEventListener('change', function() {
                currentSymbol = this.value;
                loadChart();
            });
            
            document.getElementById('timeframe-select').addEventListener('change', function() {
                currentTimeframe = this.value;
                loadChart();
            });
        }
        
        // Load initial data
        function loadInitialData() {
            loadStatus();
            loadChart();
            loadPatterns();
            loadPerformance();
            loadRisk();
            loadSentiment();
        }
        
        // Load system status
        function loadStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateStatus(data))
                .catch(error => console.error('Error loading status:', error));
        }
        
        // Update status display
        function updateStatus(data) {
            const statusIndicator = document.getElementById('status-indicator');
            const systemStatus = document.getElementById('system-status');
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            if (data.running) {
                statusIndicator.className = 'status-indicator status-running';
                systemStatus.textContent = 'System Running';
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                statusIndicator.className = 'status-indicator status-stopped';
                systemStatus.textContent = 'System Stopped';
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }
        
        // Load chart data
        function loadChart() {
            const chartContainer = document.getElementById('chart-container');
            chartContainer.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading chart...';
            
            fetch(`/api/chart/${currentSymbol}?timeframe=${currentTimeframe}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        chartContainer.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    } else {
                        const chartData = JSON.parse(data.chart);
                        Plotly.newPlot('chart-container', chartData.data, chartData.layout, {responsive: true});
                    }
                })
                .catch(error => {
                    console.error('Error loading chart:', error);
                    chartContainer.innerHTML = '<div class="alert alert-danger">Error loading chart</div>';
                });
        }
        
        // Load pattern analysis
        function loadPatterns() {
            const patternContainer = document.getElementById('pattern-analysis');
            patternContainer.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading patterns...';
            
            fetch(`/api/patterns/${currentSymbol}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        patternContainer.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    } else {
                        displayPatterns(data);
                    }
                })
                .catch(error => {
                    console.error('Error loading patterns:', error);
                    patternContainer.innerHTML = '<div class="alert alert-danger">Error loading patterns</div>';
                });
        }
        
        // Display patterns
        function displayPatterns(data) {
            const container = document.getElementById('pattern-analysis');
            
            if (data.patterns && data.patterns.length > 0) {
                let html = `<div class="mb-2">
                    <strong>Detected Patterns:</strong> ${data.total_patterns}
                    <span class="badge bg-primary ms-2">${data.high_confidence} High Confidence</span>
                </div>`;
                
                data.patterns.forEach(pattern => {
                    const confidenceClass = pattern.confidence > 0.7 ? 'pattern-high-confidence' : 'pattern-detected';
                    html += `<span class="pattern-badge ${confidenceClass}">
                        ${pattern.name} (${Math.round(pattern.confidence * 100)}%)
                    </span>`;
                });
                
                container.innerHTML = html;
            } else {
                container.innerHTML = '<div class="text-muted">No patterns detected</div>';
            }
        }
        
        // Load performance metrics
        function loadPerformance() {
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('performance-metrics').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    } else {
                        displayPerformance(data);
                    }
                })
                .catch(error => {
                    console.error('Error loading performance:', error);
                    document.getElementById('performance-metrics').innerHTML = '<div class="alert alert-danger">Error loading performance</div>';
                });
        }
        
        // Display performance metrics
        function displayPerformance(data) {
            const container = document.getElementById('performance-metrics');
            
            const pnlClass = data.total_pnl >= 0 ? 'text-success' : 'text-danger';
            const winRateClass = data.win_rate >= 50 ? 'text-success' : 'text-danger';
            
            container.innerHTML = `
                <div class="row text-center">
                    <div class="col-6">
                        <div class="h4 ${pnlClass}">$${data.total_pnl.toFixed(2)}</div>
                        <small class="text-muted">Total P&L</small>
                    </div>
                    <div class="col-6">
                        <div class="h4 ${winRateClass}">${data.win_rate.toFixed(1)}%</div>
                        <small class="text-muted">Win Rate</small>
                    </div>
                </div>
                <hr>
                <div class="row">
                    <div class="col-6">
                        <strong>Total Trades:</strong><br>
                        <span class="h6">${data.total_trades}</span>
                    </div>
                    <div class="col-6">
                        <strong>Active Trades:</strong><br>
                        <span class="h6">${data.active_trades}</span>
                    </div>
                </div>
                <div class="row mt-2">
                    <div class="col-6">
                        <strong>Avg Win:</strong><br>
                        <span class="text-success">$${data.avg_win.toFixed(2)}</span>
                    </div>
                    <div class="col-6">
                        <strong>Avg Loss:</strong><br>
                        <span class="text-danger">$${data.avg_loss.toFixed(2)}</span>
                    </div>
                </div>
            `;
        }
        
        // Load risk metrics
        function loadRisk() {
            fetch('/api/risk')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('risk-metrics').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    } else {
                        displayRisk(data);
                    }
                })
                .catch(error => {
                    console.error('Error loading risk:', error);
                    document.getElementById('risk-metrics').innerHTML = '<div class="alert alert-danger">Error loading risk</div>';
                });
        }
        
        // Display risk metrics
        function displayRisk(data) {
            const container = document.getElementById('risk-metrics');
            
            container.innerHTML = `
                <div class="mb-3">
                    <strong>Risk Level:</strong>
                    <span class="badge" style="background-color: ${data.risk_color}">${data.risk_level}</span>
                </div>
                <div class="row">
                    <div class="col-6">
                        <strong>Active Trades:</strong><br>
                        <span class="h6">${data.active_trades}</span>
                    </div>
                    <div class="col-6">
                        <strong>Total Exposure:</strong><br>
                        <span class="h6">$${data.total_exposure.toFixed(2)}</span>
                    </div>
                </div>
                <div class="row mt-2">
                    <div class="col-6">
                        <strong>Max Position:</strong><br>
                        <span class="h6">$${data.max_position_size.toFixed(2)}</span>
                    </div>
                    <div class="col-6">
                        <strong>Avg Position:</strong><br>
                        <span class="h6">$${data.avg_position_size.toFixed(2)}</span>
                    </div>
                </div>
            `;
        }
        
        // Load sentiment analysis
        function loadSentiment() {
            fetch('/api/sentiment')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('sentiment-analysis').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    } else {
                        displaySentiment(data);
                    }
                })
                .catch(error => {
                    console.error('Error loading sentiment:', error);
                    document.getElementById('sentiment-analysis').innerHTML = '<div class="alert alert-danger">Error loading sentiment</div>';
                });
        }
        
        // Display sentiment analysis
        function displaySentiment(data) {
            const container = document.getElementById('sentiment-analysis');
            
            const sentimentClass = data.sentiment_label === 'Bullish' ? 'sentiment-positive' : 
                                 data.sentiment_label === 'Bearish' ? 'sentiment-negative' : 'sentiment-neutral';
            
            container.innerHTML = `
                <div class="text-center mb-3">
                    <div class="h4 ${sentimentClass}">${data.sentiment_label}</div>
                    <div class="text-muted">Confidence: ${Math.round(data.overall_confidence * 100)}%</div>
                </div>
                <div class="progress mb-3">
                    <div class="progress-bar" style="width: ${Math.abs(data.overall_sentiment) * 100}%"></div>
                </div>
                <div class="text-muted">
                    <small>Overall Sentiment: ${data.overall_sentiment.toFixed(3)}</small>
                </div>
            `;
        }
        
        // Start system
        function startSystem() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadStatus();
                        loadInitialData();
                    }
                })
                .catch(error => console.error('Error starting system:', error));
        }
        
        // Stop system
        function stopSystem() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadStatus();
                    }
                })
                .catch(error => console.error('Error stopping system:', error));
        }
        
        // WebSocket event handlers
        socket.on('status_update', function(data) {
            updateStatus(data);
        });
        
        socket.on('performance_update', function(data) {
            displayPerformance(data);
        });
        
        socket.on('risk_update', function(data) {
            displayRisk(data);
        });
        
        // Auto-refresh every 30 seconds
        setInterval(loadInitialData, 30000);
    </script>
</body>
</html>'''
    
    with open('templates/enhanced_dashboard.html', 'w') as f:
        f.write(template_content)