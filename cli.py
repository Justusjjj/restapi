import os
import json
import typer
import numpy as np
import pandas as pd

from mtf_pipeline import MT5Connector, MT5Credentials, MultiTimeframeAnalyzer
from datasets import DatasetBuilder
from deep_model import train_model, evaluate_model
from execution import atr, size_from_risk, execute_order
from backtest_decider import DecisionRuleBacktester
from rl_trading import train_rl_agent, evaluate_rl_agent, TradingEnvironment

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
	"""Perform one analysis->decision->execution cycle with ATR-based sizing."""
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
	analyzer = MultiTimeframeAnalyzer(model_pt=model_pt)
	analysis = analyzer.analyze(symbol, mt5c)
	decision = analyzer.decision(analysis)
	if decision.get("action") == "HOLD":
		mt5c.shutdown()
		return typer.echo(json.dumps({"decision": decision}, indent=2))
	# ATR sizing on H1
	h1 = mt5c.fetch_rates(symbol, "H1", 1000)
	atr_val = atr(h1)
	if atr_val <= 0:
		volume = 0.01
		sl_pips = 50.0
	else:
		# Convert ATR price to pips assuming 1 pip = 0.0001
		sl_pips = max((atr_val * atr_mult) / 0.0001, 20.0)
		volume = size_from_risk(balance=float(os.getenv("PAPER_BALANCE", 10000)), risk_pct=risk_pct, stop_pips=sl_pips, pip_value=10.0)
	# SL/TP based on current price and pips
	tick = mt5.symbol_info_tick(symbol)
	price = tick.ask if decision["action"] == "BUY" else tick.bid
	pip = 0.0001
	sl = price - sl_pips * pip if decision["action"] == "BUY" else price + sl_pips * pip
	tp = price + sl_pips * pip if decision["action"] == "BUY" else price - sl_pips * pip
	res = execute_order(symbol, decision["action"], sl=sl, tp=tp, volume=volume)
	mt5c.shutdown()
	return typer.echo(json.dumps({"decision": decision, "order": res}, indent=2, default=str))


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


if __name__ == "__main__":
	app()