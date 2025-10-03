import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from mtf_pipeline import MT5Connector, MT5Credentials
from ict_price_action import ICTPriceActionAnalyzer
from news_sentiment_analyzer import NewsSentimentAnalyzer


class DatasetBuilder:
	def __init__(self, symbol: str, connector: MT5Connector):
		self.symbol = symbol
		self.mt5c = connector
		self.ict = ICTPriceActionAnalyzer()
		self.news = NewsSentimentAnalyzer()

	def _fetch(self, timeframe: str, bars: int) -> pd.DataFrame:
		from mtf_pipeline import TIMEFRAME_MAP
		rates = self.mt5c.fetch_rates(self.symbol, timeframe, bars)
		return rates

	def build_features(self) -> Dict[str, pd.DataFrame]:
		data = {
			"MN1": self._fetch("MN1", 600),
			"W1": self._fetch("W1", 1200),
			"H4": self._fetch("H4", 4000),
			"H1": self._fetch("H1", 6000),
			"M15": self._fetch("M15", 8000),
		}
		# ICT on H1/M15 for structure signals
		ict_h1 = self.ict.get_comprehensive_analysis(data["H1"]) if not data["H1"].empty else {}
		ict_m15 = self.ict.get_comprehensive_analysis(data["M15"]) if not data["M15"].empty else {}
		# Sentiment
		base = self.symbol[:3]
		quote = self.symbol[3:]
		news_df = self.news.fetch_forex_news([base, quote])
		news_sent = self.news.analyze_news_sentiment(news_df)
		sent = self.news.calculate_composite_sentiment(news_sent, pd.DataFrame())
		return {"data": data, "ict_h1": ict_h1, "ict_m15": ict_m15, "sentiment": sent}

	def sequences_from_features(self, horizon: int = 12, seq_len: int = 64) -> Tuple[np.ndarray, np.ndarray]:
		bundle = self.build_features()
		data = bundle["data"]["M15"]
		if data.empty or len(data) < seq_len + horizon + 1:
			return np.zeros((0, seq_len, 16)), np.zeros((0,))
		# Basic numeric features as example; can be extended
		df = data.copy()
		df["ret"] = df["close"].pct_change()
		df["vol"] = df["ret"].rolling(20).std()
		df["ma_fast"] = df["close"].ewm(span=10).mean()
		df["ma_slow"] = df["close"].ewm(span=30).mean()
		df["bb_mid"] = df["close"].rolling(20).mean()
		df["bb_std"] = df["close"].rolling(20).std()
		df = df.dropna()
		feat_cols = ["ret", "vol", "ma_fast", "ma_slow", "bb_mid", "bb_std", "high", "low", "open", "close", "volume"]
		X_list, y_list = [], []
		values = df[feat_cols].values
		closes = df["close"].values
		for i in range(seq_len, len(df) - horizon):
			seq = values[i - seq_len:i]
			fwd_ret = (closes[i + horizon] - closes[i]) / closes[i]
			label = 1 if fwd_ret > 0 else 0
			X_list.append(seq)
			y_list.append(label)
		return np.array(X_list), np.array(y_list)