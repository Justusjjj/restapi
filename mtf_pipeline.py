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
	Integrates ICT-style levels plus basic sentiment context.
	"""
	def __init__(self):
		self.ict = ICTPriceActionAnalyzer()
		self.news = NewsSentimentAnalyzer()

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

		analysis = {
			"symbol": symbol,
			"bias": {"MN1": bias_mn1, "W1": bias_w1, "H4": bias_h4},
			"levels": {"MN1": levels_mn1, "W1": levels_w1, "H4": levels_h4},
			"H1": h1_ict,
			"M15": m15_ict,
			"M5": m5_ict,
			"M1": m1_ict,
			"sentiment": composite,
			"timestamp": datetime.utcnow().isoformat(),
		}
		return analysis

	def decision(self, analysis: Dict) -> Dict:
		"""
		Simple rules to align trades: trade with MN1/W1/H4 bias, require H1 structure confluence,
		and LTF confirmation signals from M15/M5/M1 (e.g., presence of breakout/liquidity sweep + FVG/OB).
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

		if long_bias:
			return {"action": "BUY", "risk": 0.5, "reason": "HTF_bullish_with_LTF_confirm"}
		if short_bias:
			return {"action": "SELL", "risk": 0.5, "reason": "HTF_bearish_with_LTF_confirm"}
		return {"action": "HOLD", "reason": "mixed_bias"}