import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from ict_price_action import ICTPriceActionAnalyzer
from news_sentiment_analyzer import NewsSentimentAnalyzer


TIMEFRAME_MAP = {
	"M1": mt5.TIMEFRAME_M1,
	"M5": mt5.TIMEFRAME_M5,
	"M15": mt5.TIMEFRAME_M15,
	"M30": mt5.TIMEFRAME_M30,
	"H1": mt5.TIMEFRAME_H1,
	"H4": mt5.TIMEFRAME_H4,
	"D1": mt5.TIMEFRAME_D1,
	"W1": mt5.TIMEFRAME_W1,
	"MN1": mt5.TIMEFRAME_MN1,
}


@dataclass
class MT5Credentials:
	login: Optional[int]
	password: Optional[str]
	server: Optional[str]


class MT5Connector:
	"""
	Thin wrapper around MetaTrader5 Python connector for safe init/login and data fetch.
	"""
	def __init__(self, credentials: MT5Credentials):
		self.credentials = credentials
		self._initialized = False

	def initialize(self, max_retries: int = 3, retry_delay_sec: int = 3) -> bool:
		for attempt in range(max_retries):
			if not mt5.initialize():
				time.sleep(retry_delay_sec)
				continue
			if self.credentials.login and self.credentials.password and self.credentials.server:
				if not mt5.login(self.credentials.login, self.credentials.password, self.credentials.server):
					mt5.shutdown()
					time.sleep(retry_delay_sec)
					continue
			self._initialized = True
			return True
		return False

	def shutdown(self) -> None:
		if self._initialized:
			mt5.shutdown()
			self._initialized = False

	def fetch_rates(self, symbol: str, timeframe: str, bars: int = 1500) -> pd.DataFrame:
		if timeframe not in TIMEFRAME_MAP:
			raise ValueError(f"Unsupported timeframe: {timeframe}")
		rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, bars)
		if rates is None:
			return pd.DataFrame()
		df = pd.DataFrame(rates)
		df["time"] = pd.to_datetime(df["time"], unit="s")
		df.set_index("time", inplace=True)
		return df.rename(columns={"tick_volume": "volume"})


class MultiTimeframeAnalyzer:
	"""
	Builds higher-timeframe bias and key levels (MN1/W1/H4), structure on H1, and confirmation on M15/M5/M1.
	Integrates ICT-style levels plus basic sentiment context and optional deep model confirmation.
	"""
	def __init__(self, model_pt: Optional[str] = None, seq_len: int = 64, horizon: int = 12):
		self.ict = ICTPriceActionAnalyzer()
		self.news = NewsSentimentAnalyzer()
		self.model = None
		self.seq_len = seq_len
		self.horizon = horizon
		if model_pt and os.path.exists(model_pt):
			try:
				import torch
				from deep_model import LSTMClassifier
				self.model = LSTMClassifier(input_dim=11)
				state = torch.load(model_pt, map_location="cpu")
				self.model.load_state_dict(state)
				self.model.eval()
			except Exception:
				self.model = None

	def _key_levels(self, df: pd.DataFrame, lookback: int = 200) -> Dict[str, float]:
		if df.empty:
			return {}
		window = df.tail(min(len(df), lookback))
		levels = {
			"swing_high": float(window["high"].rolling(20).max().iloc[-1]),
			"swing_low": float(window["low"].rolling(20).min().iloc[-1]),
			"range_mid": float((window["high"].rolling(20).max().iloc[-1] + window["low"].rolling(20).min().iloc[-1]) / 2.0),
		}
		return levels

	def _trend_direction(self, df: pd.DataFrame) -> str:
		if df.empty:
			return "UNKNOWN"
		sma_fast = df["close"].ewm(span=20).mean()
		sma_slow = df["close"].ewm(span=50).mean()
		if sma_fast.iloc[-1] > sma_slow.iloc[-1]:
			return "BULLISH"
		if sma_fast.iloc[-1] < sma_slow.iloc[-1]:
			return "BEARISH"
		return "NEUTRAL"

	def _latest_sequence(self, m15: pd.DataFrame) -> Optional[np.ndarray]:
		if m15.empty or len(m15) < self.seq_len + self.horizon + 1:
			return None
		df = m15.copy()
		df["ret"] = df["close"].pct_change()
		df["vol"] = df["ret"].rolling(20).std()
		df["ma_fast"] = df["close"].ewm(span=10).mean()
		df["ma_slow"] = df["close"].ewm(span=30).mean()
		df["bb_mid"] = df["close"].rolling(20).mean()
		df["bb_std"] = df["close"].rolling(20).std()
		df = df.dropna()
		feat_cols = ["ret", "vol", "ma_fast", "ma_slow", "bb_mid", "bb_std", "high", "low", "open", "close", "volume"]
		if len(df) < self.seq_len:
			return None
		seq = df[feat_cols].values[-self.seq_len:]
		return seq[np.newaxis, ...]

	def analyze(self, symbol: str, mt5c: MT5Connector) -> Dict:
		# Fetch required timeframes
		mn1 = mt5c.fetch_rates(symbol, "MN1", 600)
		w1 = mt5c.fetch_rates(symbol, "W1", 1200)
		h4 = mt5c.fetch_rates(symbol, "H4", 3000)
		h1 = mt5c.fetch_rates(symbol, "H1", 4000)
		m15 = mt5c.fetch_rates(symbol, "M15", 4000)
		m5 = mt5c.fetch_rates(symbol, "M5", 4000)
		m1 = mt5c.fetch_rates(symbol, "M1", 4000)

		# Higher timeframe bias and key levels
		bias_mn1 = self._trend_direction(mn1)
		bias_w1 = self._trend_direction(w1)
		bias_h4 = self._trend_direction(h4)
		levels_mn1 = self._key_levels(mn1)
		levels_w1 = self._key_levels(w1)
		levels_h4 = self._key_levels(h4)

		# H1 structure: trend breaks, order blocks, FVG, liquidity using ICT analyzer
		h1_ict = self.ict.get_comprehensive_analysis(h1)

		# LTF confirmation
		m15_ict = self.ict.get_comprehensive_analysis(m15)
		m5_ict = self.ict.get_comprehensive_analysis(m5)
		m1_ict = self.ict.get_comprehensive_analysis(m1)

		# Lightweight sentiment context (per currency)
		base = symbol[:3]
		quote = symbol[3:]
		news_df = self.news.fetch_forex_news([base, quote])
		news_sent = self.news.analyze_news_sentiment(news_df)
		composite = self.news.calculate_composite_sentiment(news_sent, pd.DataFrame())

		# Deep model confirmation on M15 sequence
		deep_pred = None
		if self.model is not None:
			seq = self._latest_sequence(m15)
			if seq is not None:
				import torch
				with torch.no_grad():
					xt = torch.tensor(seq, dtype=torch.float32)
					logits = self.model(xt)
					probs = torch.softmax(logits, dim=1).numpy()[0]
					deep_pred = {"up": float(probs[1]), "down": float(probs[0])}

		analysis = {
			"symbol": symbol,
			"bias": {"MN1": bias_mn1, "W1": bias_w1, "H4": bias_h4},
			"levels": {"MN1": levels_mn1, "W1": levels_w1, "H4": levels_h4},
			"H1": h1_ict,
			"M15": m15_ict,
			"M5": m5_ict,
			"M1": m1_ict,
			"sentiment": composite,
			"deep": deep_pred,
			"timestamp": datetime.utcnow().isoformat(),
		}
		return analysis

	def decision(self, analysis: Dict) -> Dict:
		"""
		Rule-set: HTF consensus + ICT LTF confirmation + optional deep up/down tilt.
		"""
		biases = analysis.get("bias", {})
		if len({biases.get("MN1"), biases.get("W1"), biases.get("H4")} - {None}) < 3:
			return {"action": "HOLD", "reason": "no_consensus_bias"}

		long_bias = all(b == "BULLISH" for b in [biases["MN1"], biases["W1"], biases["H4"]])
		short_bias = all(b == "BEARISH" for b in [biases["MN1"], biases["W1"], biases["H4"]])

		def has_ltf_confirm(tf_data: Dict) -> bool:
			try:
				order_flow_imbalance = tf_data.get("order_flow", {}).get("order_flow_imbalance", 0)
				patterns_sum = tf_data.get("candlestick_patterns", {}).get("total_patterns", 0)
				return (patterns_sum or 0) > 0 and order_flow_imbalance is not None
			except Exception:
				return False

		ltf_ok = any(has_ltf_confirm(analysis.get(tf, {})) for tf in ["M15", "M5", "M1"]) 
		if not ltf_ok:
			return {"action": "HOLD", "reason": "no_ltf_confirmation"}

		deep = analysis.get("deep") or {}
		up_p = deep.get("up", 0.5)
		down_p = deep.get("down", 0.5)

		if long_bias and up_p >= 0.55:
			return {"action": "BUY", "risk": 0.5, "reason": "HTF_bullish_LTF_ok_deep_up"}
		if short_bias and down_p >= 0.55:
			return {"action": "SELL", "risk": 0.5, "reason": "HTF_bearish_LTF_ok_deep_down"}
		return {"action": "HOLD", "reason": "mixed_or_low_confidence"}