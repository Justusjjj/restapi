import os
import json
import typer
import numpy as np

from mtf_pipeline import MT5Connector, MT5Credentials, MultiTimeframeAnalyzer
from datasets import DatasetBuilder
from deep_model import train_model, evaluate_model

app = typer.Typer(add_completion=False)


def _read_env_int(name: str):
	val = os.getenv(name)
	try:
		return int(val) if val is not None and val != "" else None
	except Exception:
		return None


@app.command()
def analyze(symbol: str = typer.Argument(..., help="Symbol like EURUSD")):
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
	analyzer = MultiTimeframeAnalyzer()
	result = analyzer.analyze(symbol, mt5c)
	mt5c.shutdown()
	typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def decide(symbol: str = typer.Argument(..., help="Symbol like EURUSD")):
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
	analyzer = MultiTimeframeAnalyzer()
	analysis = analyzer.analyze(symbol, mt5c)
	decision = analyzer.decision(analysis)
	mt5c.shutdown()
	typer.echo(json.dumps(decision, indent=2, default=str))


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


if __name__ == "__main__":
	app()