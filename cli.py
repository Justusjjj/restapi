import os
import json
import typer
import numpy as np
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

from mtf_pipeline import MT5Connector, MT5Credentials, MultiTimeframeAnalyzer
from datasets import DatasetBuilder
from deep_model import train_model, evaluate_model
from execution import atr, size_from_risk, execute_order
from backtest_decider import DecisionRuleBacktester
from rl_trading import train_rl_agent, evaluate_rl_agent, TradingEnvironment
from hybrid_transformer_tree import train_hybrid_model, evaluate_hybrid_model, HybridTransformerTree
from mistake_learning import MistakeLearningSystem, analyze_trade_outcome, get_demo_account_balance
from advanced_ensemble import AdvancedEnsembleModel, MarketRegimeDetector, create_advanced_ensemble_config
from portfolio_optimizer import AdvancedPortfolioOptimizer, MultiAssetCorrelationAnalyzer
from advanced_order_manager import AdvancedOrderManager, AdvancedOrder, OrderType
from advanced_news_sentiment import AdvancedNewsSentimentAnalyzer
from advanced_backtester import AdvancedBacktester
from performance_dashboard import RealTimePerformanceDashboard, PerformanceAnalytics
from ict_strategy import ICTStrategy
from momentum_strategies import MomentumStrategy, MeanReversionStrategy
from breakout_strategies import BreakoutStrategy, RangeTradingStrategy
from multi_timeframe_strategies import MultiTimeframeStrategy, StrategyManager
from strategy_optimizer import StrategyOptimizer

app = typer.Typer(add_completion=False)


def _read_env_int(name: str):
	val = os.getenv(name)
	try:
		return int(val) if val is not None and val != "" else None
	except Exception:
		return None


@app.command()
def analyze(symbol: str = typer.Argument(..., help="Symbol like EURUSD"), model_pt: str = typer.Option(None)):
	"""Run multi-timeframe analysis and print JSON output."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5. Check terminal and credentials.")
		typer.Exit(code=1)
	analyzer = MultiTimeframeAnalyzer(model_pt=model_pt)
	result = analyzer.analyze(symbol, mt5c)
	mt5c.shutdown()
	typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def decide(symbol: str = typer.Argument(..., help="Symbol like EURUSD"), model_pt: str = typer.Option(None)):
	"""Run analysis and output a trade decision (BUY/SELL/HOLD)."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5. Check terminal and credentials.")
		typer.Exit(code=1)
	analyzer = MultiTimeframeAnalyzer(model_pt=model_pt)
	analysis = analyzer.analyze(symbol, mt5c)
	decision = analyzer.decision(analysis)
	mt5c.shutdown()
	typer.echo(json.dumps({"decision": decision, "analysis": analysis}, indent=2, default=str))


@app.command("build-dataset")
def build_dataset(symbol: str = typer.Argument(...), out_npz: str = typer.Option("dataset.npz", help="Output .npz path"), seq_len: int = typer.Option(64), horizon: int = typer.Option(12)):
	"""Build a supervised dataset of sequences for deep model training."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	builder = DatasetBuilder(symbol, mt5c)
	X, y = builder.sequences_from_features(horizon=horizon, seq_len=seq_len)
	mt5c.shutdown()
	np.savez_compressed(out_npz, X=X, y=y)
	typer.echo(f"Saved dataset: {out_npz} (X={X.shape}, y={y.shape})")


@app.command("train")
def train(npz: str = typer.Argument(...), epochs: int = typer.Option(25), batch_size: int = typer.Option(64), lr: float = typer.Option(1e-3), out_pt: str = typer.Option("model.pt")):
	"""Train LSTM classifier on saved dataset."""
	data = np.load(npz)
	X, y = data["X"], data["y"]
	model = train_model(X, y, epochs=epochs, batch_size=batch_size, lr=lr)
	import torch
	torch.save(model.state_dict(), out_pt)
	typer.echo(f"Saved model: {out_pt}")


@app.command("evaluate")
def evaluate(npz: str = typer.Argument(...), model_pt: str = typer.Argument(...)):
	"""Evaluate trained model accuracy."""
	import torch
	from deep_model import LSTMClassifier
	data = np.load(npz)
	X, y = data["X"], data["y"]
	model = LSTMClassifier(input_dim=X.shape[-1])
	state = torch.load(model_pt, map_location="cpu")
	model.load_state_dict(state)
	metrics = evaluate_model(model, X, y)
	typer.echo(json.dumps(metrics, indent=2))


@app.command("trade-once")
def trade_once(symbol: str = typer.Argument(...), model_pt: str = typer.Option(None), risk_pct: float = typer.Option(0.01), atr_mult: float = typer.Option(1.5)):
	"""Perform one analysis->decision->execution cycle with ATR-based sizing and AI mistake learning."""
	from mtf_pipeline import TIMEFRAME_MAP
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Initialize AI learning system
	ai_learner = MistakeLearningSystem()
	
	analyzer = MultiTimeframeAnalyzer(model_pt=model_pt)
	analysis = analyzer.analyze(symbol, mt5c)
	decision = analyzer.decision(analysis)
	
	# Apply AI adaptations
	context = {
		"volatility": analysis.get("H1", {}).get("ict_signal", {}).get("composite_score", 0.5),
		"confidence": analysis.get("H1", {}).get("ict_signal", {}).get("confidence", 0.5),
		"hour": datetime.now().hour,
		"news_impact": analysis.get("sentiment", {}).get("composite_score", 0.5)
	}
	
	features = {
		"confidence": context["confidence"],
		"volatility": context["volatility"],
		"bias_alignment": 1.0 if all(b == decision.get("action", "HOLD") for b in analysis.get("bias", {}).values()) else 0.0
	}
	
	should_trade, ai_reason, confidence_adjustment = ai_learner.apply_adaptations(
		symbol, decision.get("action", "HOLD"), context, features
	)
	
	if not should_trade:
		mt5c.shutdown()
		return typer.echo(json.dumps({
			"decision": decision, 
			"ai_override": True, 
			"ai_reason": ai_reason,
			"confidence_adjustment": confidence_adjustment
		}, indent=2))
	
	if decision.get("action") == "HOLD":
		mt5c.shutdown()
		return typer.echo(json.dumps({"decision": decision}, indent=2))
	
	# Get actual demo account balance
	demo_balance = get_demo_account_balance()
	
	# ATR sizing on H1
	h1 = mt5c.fetch_rates(symbol, "H1", 1000)
	atr_val = atr(h1)
	if atr_val <= 0:
		volume = 0.01
		sl_pips = 50.0
	else:
		# Convert ATR price to pips assuming 1 pip = 0.0001
		sl_pips = max((atr_val * atr_mult) / 0.0001, 20.0)
		volume = size_from_risk(balance=demo_balance, risk_pct=risk_pct, stop_pips=sl_pips, pip_value=10.0)
	
	# SL/TP based on current price and pips
	tick = mt5.symbol_info_tick(symbol)
	price = tick.ask if decision["action"] == "BUY" else tick.bid
	pip = 0.0001
	sl = price - sl_pips * pip if decision["action"] == "BUY" else price + sl_pips * pip
	tp = price + sl_pips * pip if decision["action"] == "BUY" else price - sl_pips * pip
	
	res = execute_order(symbol, decision["action"], sl=sl, tp=tp, volume=volume)
	
	# Record trade for AI learning
	trade_result = {
		"symbol": symbol,
		"action": decision["action"],
		"price": price,
		"volume": volume,
		"pnl": 0.0,  # Will be updated when position closes
		"max_drawdown": 0.0
	}
	
	# Analyze for mistakes (simplified - in real implementation, track actual P&L)
	expected_outcome = "profit" if decision["action"] in ["BUY", "SELL"] else "hold"
	mistake = analyze_trade_outcome(trade_result, expected_outcome, context, features)
	
	if mistake:
		lesson = ai_learner.record_mistake(
			symbol=mistake.symbol,
			action=mistake.action,
			price=mistake.price,
			context=mistake.context,
			mistake_type=mistake.mistake_type,
			outcome=mistake.outcome,
			severity=mistake.severity,
			features=mistake.features_at_mistake
		)
		print(f"🤖 AI Learning: {lesson}")
	
	mt5c.shutdown()
	return typer.echo(json.dumps({
		"decision": decision, 
		"order": res, 
		"demo_balance": demo_balance,
		"ai_reason": ai_reason,
		"confidence_adjustment": confidence_adjustment
	}, indent=2, default=str))


@app.command("backtest-decider")
def backtest_decider(symbol: str = typer.Argument(...), model_pt: str = typer.Option(None), risk_pct: float = typer.Option(0.01), atr_mult: float = typer.Option(1.5), out_json: str = typer.Option("backtest_results.json")):
	"""Backtest decision rules over historical data."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	backtester = DecisionRuleBacktester(model_pt=model_pt)
	results = backtester.run_backtest(symbol, mt5c, risk_per_trade=risk_pct, atr_multiplier=atr_mult)
	mt5c.shutdown()
	
	# Save results
	with open(out_json, 'w') as f:
		json.dump(results, f, indent=2, default=str)
	
	typer.echo(f"Backtest completed. Results saved to {out_json}")
	typer.echo(f"Total Return: {results.get('total_return', 0):.2%}")
	typer.echo(f"Win Rate: {results.get('win_rate', 0):.2%}")
	typer.echo(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")


@app.command("rl-train")
def rl_train(symbol: str = typer.Argument(...), timesteps: int = typer.Option(100000), lr: float = typer.Option(3e-4), out_model: str = typer.Option("ppo_trading_model")):
	"""Train RL agent (PPO) on trading environment."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Get training data
	h1_data = mt5c.fetch_rates(symbol, "H1", 5000)
	mt5c.shutdown()
	
	if h1_data.empty:
		typer.echo("No data available for training.")
		typer.Exit(code=1)
	
	# Train RL agent
	model = train_rl_agent(h1_data, symbol, total_timesteps=timesteps, learning_rate=lr, model_save_path=out_model)
	
	typer.echo(f"RL training completed. Model saved to {out_model}")


@app.command("rl-evaluate")
def rl_evaluate(symbol: str = typer.Argument(...), model_path: str = typer.Argument(...), episodes: int = typer.Option(10)):
	"""Evaluate trained RL agent."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Get evaluation data
	h1_data = mt5c.fetch_rates(symbol, "H1", 2000)
	mt5c.shutdown()
	
	if h1_data.empty:
		typer.echo("No data available for evaluation.")
		typer.Exit(code=1)
	
	# Load model and evaluate
	from stable_baselines3 import PPO
	model = PPO.load(model_path)
	
	results = evaluate_rl_agent(model, h1_data, symbol, num_episodes=episodes)
	
	typer.echo(json.dumps(results, indent=2))


@app.command("hybrid-train")
def hybrid_train(npz: str = typer.Argument(...), tree_type: str = typer.Option("random_forest", help="Tree type: random_forest, gradient_boosting, single_tree"), epochs: int = typer.Option(50), batch_size: int = typer.Option(64), lr: float = typer.Option(1e-3), out_model: str = typer.Option("hybrid_model.pt")):
	"""Train hybrid Transformer + Decision Tree model."""
	data = np.load(npz)
	X, y = data["X"], data["y"]
	
	model = train_hybrid_model(X, y, tree_type=tree_type, epochs=epochs, batch_size=batch_size, lr=lr)
	model.save_model(out_model)
	
	typer.echo(f"Hybrid model training completed. Model saved to {out_model}")


@app.command("hybrid-evaluate")
def hybrid_evaluate(npz: str = typer.Argument(...), model_path: str = typer.Argument(...)):
	"""Evaluate hybrid Transformer + Decision Tree model."""
	data = np.load(npz)
	X, y = data["X"], data["y"]
	
	model = HybridTransformerTree()
	model.load_model(model_path)
	
	results = evaluate_hybrid_model(model, X, y)
	
	typer.echo(json.dumps(results, indent=2))


@app.command("hybrid-decide")
def hybrid_decide(symbol: str = typer.Argument(...), model_path: str = typer.Argument(...)):
	"""Make trading decision using hybrid Transformer + Decision Tree model."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Get recent data for prediction
	m15_data = mt5c.fetch_rates(symbol, "M15", 1000)
	mt5c.shutdown()
	
	if m15_data.empty:
		typer.echo("No data available.")
		typer.Exit(code=1)
	
	# Prepare features (same as in datasets.py)
	df = m15_data.copy()
	df["ret"] = df["close"].pct_change()
	df["vol"] = df["ret"].rolling(20).std()
	df["ma_fast"] = df["close"].ewm(span=10).mean()
	df["ma_slow"] = df["close"].ewm(span=30).mean()
	df["bb_mid"] = df["close"].rolling(20).mean()
	df["bb_std"] = df["close"].rolling(20).std()
	df = df.dropna()
	
	feat_cols = ["ret", "vol", "ma_fast", "ma_slow", "bb_mid", "bb_std", "high", "low", "open", "close", "volume"]
	seq_len = 64
	
	if len(df) < seq_len:
		typer.echo("Insufficient data for prediction.")
		typer.Exit(code=1)
	
	# Get latest sequence
	latest_seq = df[feat_cols].values[-seq_len:]
	X_pred = latest_seq[np.newaxis, ...]
	
	# Load model and predict
	model = HybridTransformerTree()
	model.load_model(model_path)
	
	prediction = model.predict(X_pred)[0]
	probabilities = model.predict_proba(X_pred)[0]
	
	# Convert to trading decision
	action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
	action = action_map.get(prediction, "HOLD")
	
	decision = {
		"action": action,
		"confidence": float(max(probabilities)),
		"probabilities": {
			"HOLD": float(probabilities[0]),
			"BUY": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
			"SELL": float(probabilities[2]) if len(probabilities) > 2 else 0.0
		},
		"feature_importance": model._get_feature_importance(),
		"timestamp": datetime.now().isoformat()
	}
	
	typer.echo(json.dumps(decision, indent=2))


@app.command("ai-learning-status")
def ai_learning_status():
	"""Show AI learning progress and adaptation rules."""
	ai_learner = MistakeLearningSystem()
	summary = ai_learner.get_learning_summary()
	
	typer.echo(json.dumps(summary, indent=2))


@app.command("ai-reset-learning")
def ai_reset_learning():
	"""Reset AI learning database (clear all mistakes and rules)."""
	import os
	if os.path.exists("mistakes_db.json"):
		os.remove("mistakes_db.json")
		typer.echo("🤖 AI Learning database reset successfully")
	else:
		typer.echo("No learning database found to reset")


@app.command("ai-show-rules")
def ai_show_rules():
	"""Show current AI adaptation rules."""
	ai_learner = MistakeLearningSystem()
	
	if not ai_learner.adaptation_rules:
		typer.echo("No adaptation rules learned yet")
		return
	
	typer.echo("🤖 AI Adaptation Rules:")
	for pattern, rule in ai_learner.adaptation_rules.items():
		typer.echo(f"\nPattern: {pattern}")
		typer.echo(f"Action: {rule['action']}")
		typer.echo(f"Strength: {rule['strength']:.2f}")
		typer.echo(f"Conditions: {rule['conditions']}")
		typer.echo(f"Created: {rule['created_at']}")


@app.command("demo-balance")
def demo_balance():
	"""Show current demo account balance."""
	balance = get_demo_account_balance()
	typer.echo(f"💰 Demo Account Balance: ${balance:.2f}")


@app.command("ensemble-train")
def ensemble_train(npz: str = typer.Argument(...), epochs: int = typer.Option(50), batch_size: int = typer.Option(64), lr: float = typer.Option(1e-3), out_model: str = typer.Option("ensemble_model")):
	"""Train advanced ensemble model with dynamic model selection."""
	data = np.load(npz)
	X, y = data["X"], data["y"]
	
	# Create ensemble configuration
	config = create_advanced_ensemble_config()
	
	# Initialize ensemble
	ensemble = AdvancedEnsembleModel(config)
	
	# Train ensemble
	results = ensemble.train_ensemble(X, y, validation_split=0.2)
	
	# Save ensemble
	ensemble.save_ensemble(out_model)
	
	typer.echo(f"Advanced ensemble training completed. Model saved to {out_model}")


@app.command("ensemble-predict")
def ensemble_predict(npz: str = typer.Argument(...), model_path: str = typer.Argument(...)):
	"""Make predictions using advanced ensemble model."""
	data = np.load(npz)
	X = data["X"]
	
	# Load ensemble
	ensemble = AdvancedEnsembleModel({})
	ensemble.load_ensemble(model_path)
	
	# Make predictions
	predictions = ensemble.predict_ensemble(X, use_regime_selection=True)
	
	typer.echo(json.dumps(predictions, indent=2))


@app.command("portfolio-optimize")
def portfolio_optimize(symbols: str = typer.Argument(...), method: str = typer.Option("risk_parity", help="risk_parity or mean_variance")):
	"""Optimize portfolio using advanced methods."""
	symbol_list = symbols.split(",")
	
	# Initialize portfolio optimizer
	optimizer = AdvancedPortfolioOptimizer(symbol_list)
	
	# This would need MT5 connection in real implementation
	typer.echo(f"Portfolio optimization for {symbol_list} using {method} method")


@app.command("advanced-backtest")
def advanced_backtest(symbols: str = typer.Argument(...), method: str = typer.Option("walk_forward", help="walk_forward or regime_aware")):
	"""Run advanced backtesting with walk-forward analysis."""
	symbol_list = symbols.split(",")
	
	# Initialize advanced backtester
	backtester = AdvancedBacktester(symbol_list)
	backtester.setup_walk_forward_analysis()
	
	typer.echo(f"Advanced backtesting for {symbol_list} using {method} method")


@app.command("news-sentiment")
def news_sentiment(symbols: str = typer.Argument(...), hours_back: int = typer.Option(24)):
	"""Get real-time news sentiment analysis."""
	symbol_list = symbols.split(",")
	
	# Initialize news analyzer
	analyzer = AdvancedNewsSentimentAnalyzer()
	
	# Get sentiment summary
	sentiment_data = analyzer.get_sentiment_summary(symbol_list, hours_back)
	
	typer.echo(json.dumps(sentiment_data, indent=2))


@app.command("start-dashboard")
def start_dashboard(symbols: str = typer.Argument(...), port: int = typer.Option(8050)):
	"""Start real-time performance dashboard."""
	symbol_list = symbols.split(",")
	
	# Initialize dashboard
	dashboard = RealTimePerformanceDashboard(symbol_list)
	
	typer.echo(f"Starting dashboard for {symbol_list} on port {port}")
	dashboard.run_dashboard(port=port)


@app.command("strategy-test")
def strategy_test(symbol: str = typer.Argument(...), strategy_name: str = typer.Argument(...)):
	"""Test a specific trading strategy."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Initialize strategy
	strategies = {
		"ict": ICTStrategy(symbol, "H1"),
		"momentum": MomentumStrategy(symbol, "H1"),
		"mean_reversion": MeanReversionStrategy(symbol, "H1"),
		"breakout": BreakoutStrategy(symbol, "H1"),
		"range_trading": RangeTradingStrategy(symbol, "H1"),
		"multi_timeframe": MultiTimeframeStrategy(symbol, "H1")
	}
	
	if strategy_name not in strategies:
		typer.echo(f"Unknown strategy: {strategy_name}")
		typer.echo(f"Available strategies: {', '.join(strategies.keys())}")
		mt5c.shutdown()
		typer.Exit(code=1)
	
	strategy = strategies[strategy_name]
	
	# Get data
	data = mt5c.fetch_rates(symbol, "H1", 1000)
	mt5c.shutdown()
	
	if data.empty:
		typer.echo("No data available.")
		typer.Exit(code=1)
	
	# Generate signal
	if strategy_name == "multi_timeframe":
		# Multi-timeframe needs all timeframes
		mtf_data = {}
		for tf in ["H4", "H1", "M15"]:
			tf_data = mt5c.fetch_rates(symbol, tf, 500)
			if not tf_data.empty:
				mtf_data[tf] = tf_data
		
		signal = strategy.generate_signal(mtf_data)
	else:
		# Single timeframe strategies can use H1 data
		signal = strategy.generate_signal(data)
	
	if signal:
		result = {
			"strategy": strategy_name,
			"symbol": symbol,
			"action": signal.action,
			"entry_price": signal.entry_price,
			"stop_loss": signal.stop_loss,
			"take_profit": signal.take_profit,
			"confidence": signal.confidence,
			"signal_strength": signal.signal_strength.value,
			"reason": signal.reason,
			"risk_reward_ratio": signal.risk_reward_ratio,
			"timestamp": signal.timestamp.isoformat()
		}
		typer.echo(json.dumps(result, indent=2))
	else:
		typer.echo(json.dumps({"strategy": strategy_name, "signal": "HOLD", "reason": "No signal generated"}, indent=2))


@app.command("strategy-backtest")
def strategy_backtest(symbol: str = typer.Argument(...), strategy_name: str = typer.Option("all", help="Strategy name or 'all' for all strategies")):
	"""Run backtest for trading strategies."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Initialize optimizer
	optimizer = StrategyOptimizer(symbol)
	
	# Get data for all timeframes
	data = {}
	for tf in ["H4", "H1", "M15"]:
		tf_data = mt5c.fetch_rates(symbol, tf, 1000)
		if not tf_data.empty:
			data[tf] = tf_data
	
	mt5c.shutdown()
	
	if not data:
		typer.echo("No data available.")
		typer.Exit(code=1)
	
	# Run backtest
	if strategy_name == "all":
		results = optimizer.run_comprehensive_backtest(data)
	else:
		# Test specific strategy
		if strategy_name not in optimizer.strategies:
			typer.echo(f"Unknown strategy: {strategy_name}")
			typer.echo(f"Available strategies: {', '.join(optimizer.strategies.keys())}")
			typer.Exit(code=1)
		
		strategy = optimizer.strategies[strategy_name]
		performance = optimizer._backtest_strategy(strategy, data, 1000)
		results = {strategy_name: performance}
	
	# Generate report
	optimizer.generate_performance_report(f"{symbol}_{strategy_name}_backtest_report.html")
	
	# Export results
	optimizer.export_results(f"{symbol}_{strategy_name}_results.json")
	
	# Show summary
	rankings = optimizer.get_strategy_rankings()
	best_strategy = optimizer.get_best_strategy()
	
	summary = {
		"symbol": symbol,
		"best_strategy": best_strategy,
		"rankings": rankings,
		"total_strategies_tested": len(results),
		"report_generated": f"{symbol}_{strategy_name}_backtest_report.html"
	}
	
	typer.echo(json.dumps(summary, indent=2))


@app.command("strategy-optimize")
def strategy_optimize(symbol: str = typer.Argument(...), strategy_name: str = typer.Argument(...)):
	"""Optimize strategy parameters."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Get data
	data = {}
	for tf in ["H4", "H1", "M15"]:
		tf_data = mt5c.fetch_rates(symbol, tf, 1000)
		if not tf_data.empty:
			data[tf] = tf_data
	
	mt5c.shutdown()
	
	if not data:
		typer.echo("No data available.")
		typer.Exit(code=1)
	
	# Initialize optimizer
	optimizer = StrategyOptimizer(symbol)
	
	# Define parameter grids for optimization
	parameter_grids = {
		"ICT": {
			"order_block_lookback": [15, 20, 25],
			"fvg_lookback": [8, 10, 12],
			"min_confidence": [0.5, 0.6, 0.7],
			"risk_reward_min": [1.2, 1.5, 2.0]
		},
		"Momentum": {
			"ema_fast": [10, 12, 15],
			"ema_slow": [24, 26, 30],
			"rsi_period": [12, 14, 16],
			"min_trend_strength": [0.5, 0.6, 0.7]
		},
		"Mean_Reversion": {
			"bb_period": [18, 20, 22],
			"bb_std": [1.8, 2.0, 2.2],
			"rsi_period": [12, 14, 16],
			"min_reversal_strength": [0.5, 0.6, 0.7]
		},
		"Breakout": {
			"consolidation_periods": [18, 20, 22],
			"breakout_threshold": [0.0003, 0.0005, 0.0007],
			"min_breakout_strength": [0.6, 0.7, 0.8]
		},
		"Range_Trading": {
			"range_periods": [45, 50, 55],
			"min_range_size": [0.0008, 0.001, 0.0012],
			"max_range_size": [0.004, 0.005, 0.006],
			"min_bounces": [2, 3, 4]
		}
	}
	
	if strategy_name not in parameter_grids:
		typer.echo(f"Optimization not available for strategy: {strategy_name}")
		typer.echo(f"Available for optimization: {', '.join(parameter_grids.keys())}")
		typer.Exit(code=1)
	
	# Run optimization
	parameter_grid = parameter_grids[strategy_name]
	results = optimizer.optimize_strategy_parameters(strategy_name, data, parameter_grid)
	
	typer.echo(json.dumps(results, indent=2))


@app.command("strategy-manager")
def strategy_manager(symbol: str = typer.Argument(...), action: str = typer.Argument(...)):
	"""Manage strategy ensemble."""
	creds = MT5Credentials(
		login=_read_env_int("MT5_LOGIN"),
		password=os.getenv("MT5_PASSWORD"),
		server=os.getenv("MT5_SERVER"),
	)
	mt5c = MT5Connector(creds)
	if not mt5c.initialize():
		typer.echo("Failed to initialize MT5.")
		typer.Exit(code=1)
	
	# Initialize strategy manager
	manager = StrategyManager(symbol)
	
	# Get data
	data = {}
	for tf in ["H4", "H1", "M15"]:
		tf_data = mt5c.fetch_rates(symbol, tf, 1000)
		if not tf_data.empty:
			data[tf] = tf_data
	
	mt5c.shutdown()
	
	if not data:
		typer.echo("No data available.")
		typer.Exit(code=1)
	
	if action == "ensemble-signal":
		# Generate ensemble signal
		signal = manager.generate_ensemble_signal(data)
		
		if signal:
			result = {
				"action": signal.action,
				"entry_price": signal.entry_price,
				"stop_loss": signal.stop_loss,
				"take_profit": signal.take_profit,
				"confidence": signal.confidence,
				"reason": signal.reason,
				"individual_signals": signal.additional_info.get("individual_signals", {}),
				"votes": signal.additional_info.get("votes", {}),
				"active_strategies": signal.additional_info.get("active_strategies", [])
			}
			typer.echo(json.dumps(result, indent=2))
		else:
			typer.echo(json.dumps({"signal": "HOLD", "reason": "No ensemble signal generated"}, indent=2))
	
	elif action == "performance":
		# Get strategy performance
		performance = manager.get_strategy_performance()
		typer.echo(json.dumps(performance, indent=2))
	
	elif action == "set-weights":
		# Set strategy weights (would need parameters)
		typer.echo("Strategy weight setting requires additional parameters")
	
	else:
		typer.echo(f"Unknown action: {action}")
		typer.echo("Available actions: ensemble-signal, performance, set-weights")


if __name__ == "__main__":
	app()