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


if __name__ == "__main__":
	app()