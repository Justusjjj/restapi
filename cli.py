import os
import json
import typer

from mtf_pipeline import MT5Connector, MT5Credentials, MultiTimeframeAnalyzer

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


if __name__ == "__main__":
	app()