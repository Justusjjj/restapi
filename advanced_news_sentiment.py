import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import threading
import time
from dataclasses import dataclass


@dataclass
class NewsItem:
	"""Structured news item with sentiment analysis"""
	title: str
	content: str
	source: str
	published_at: datetime
	symbols: List[str]
	sentiment_score: float
	sentiment_label: str
	impact_score: float
	keywords: List[str]
	relevance_score: float


class AdvancedNewsSentimentAnalyzer:
	"""
	Advanced real-time news sentiment analyzer with NLP processing.
	Processes multiple news sources and provides real-time sentiment analysis.
	"""
	
	def __init__(self, api_keys: Dict[str, str] = None):
		self.api_keys = api_keys or {}
		self.sentiment_analyzer = SentimentIntensityAnalyzer()
		self.lemmatizer = WordNetLemmatizer()
		
		# Download required NLTK data
		try:
			nltk.download('punkt', quiet=True)
			nltk.download('stopwords', quiet=True)
			nltk.download('wordnet', quiet=True)
			nltk.download('vader_lexicon', quiet=True)
		except:
			pass
		
		self.stop_words = set(stopwords.words('english'))
		
		# Forex-specific keywords and their impact weights
		self.forex_keywords = {
			# Economic indicators
			"gdp": 0.8, "inflation": 0.9, "unemployment": 0.7, "interest rate": 1.0,
			"federal reserve": 1.0, "central bank": 0.9, "monetary policy": 0.9,
			"fiscal policy": 0.8, "budget": 0.6, "deficit": 0.7,
			
			# Currency-specific
			"dollar": 0.8, "euro": 0.8, "pound": 0.8, "yen": 0.8, "franc": 0.7,
			"australian dollar": 0.7, "canadian dollar": 0.7, "swiss franc": 0.7,
			
			# Market conditions
			"recession": 1.0, "growth": 0.8, "recovery": 0.7, "crisis": 1.0,
			"volatility": 0.8, "stability": 0.6, "uncertainty": 0.9,
			
			# Trading terms
			"forex": 0.6, "currency": 0.7, "exchange rate": 0.8, "trade": 0.7,
			"export": 0.6, "import": 0.6, "balance of trade": 0.8
		}
		
		# News sources configuration
		self.news_sources = {
			"newsapi": {
				"url": "https://newsapi.org/v2/everything",
				"params": {
					"apiKey": self.api_keys.get("newsapi"),
					"language": "en",
					"sortBy": "publishedAt",
					"pageSize": 50
				}
			},
			"alpha_vantage": {
				"url": "https://www.alphavantage.co/query",
				"params": {
					"apikey": self.api_keys.get("alpha_vantage"),
					"function": "NEWS_SENTIMENT"
				}
			}
		}
		
		# Real-time processing
		self.news_queue = []
		self.processed_news = []
		self.sentiment_cache = {}
		self.processing_active = True
		
		# Start background processing
		self.processing_thread = threading.Thread(target=self._process_news_queue, daemon=True)
		self.processing_thread.start()
	
	def fetch_news(self, symbols: List[str], hours_back: int = 24) -> List[NewsItem]:
		"""
		Fetch news from multiple sources for given symbols
		"""
		
		all_news = []
		
		# Fetch from NewsAPI
		newsapi_news = self._fetch_newsapi(symbols, hours_back)
		all_news.extend(newsapi_news)
		
		# Fetch from Alpha Vantage
		av_news = self._fetch_alpha_vantage(symbols, hours_back)
		all_news.extend(av_news)
		
		# Add to processing queue
		for news in all_news:
			self.news_queue.append(news)
		
		return all_news
	
	def _fetch_newsapi(self, symbols: List[str], hours_back: int) -> List[NewsItem]:
		"""Fetch news from NewsAPI"""
		
		if not self.api_keys.get("newsapi"):
			return []
		
		news_items = []
		
		try:
			# Create search query
			query_terms = []
			for symbol in symbols:
				# Convert symbol to search terms
				if symbol.startswith("EUR"):
					query_terms.extend(["euro", "european", "ecb"])
				elif symbol.startswith("GBP"):
					query_terms.extend(["pound", "sterling", "uk", "britain", "boe"])
				elif symbol.startswith("USD"):
					query_terms.extend(["dollar", "us", "america", "federal reserve"])
				elif symbol.startswith("JPY"):
					query_terms.extend(["yen", "japan", "bank of japan"])
			
			query = " OR ".join(query_terms[:5])  # Limit query length
			
			# Calculate time range
			from_date = datetime.now() - timedelta(hours=hours_back)
			
			params = self.news_sources["newsapi"]["params"].copy()
			params.update({
				"q": query,
				"from": from_date.isoformat(),
				"domains": "reuters.com,bloomberg.com,cnbc.com,marketwatch.com,forexfactory.com"
			})
			
			response = requests.get(
				self.news_sources["newsapi"]["url"],
				params=params,
				timeout=10
			)
			
			if response.status_code == 200:
				data = response.json()
				
				for article in data.get("articles", []):
					news_item = NewsItem(
						title=article.get("title", ""),
						content=article.get("description", ""),
						source=article.get("source", {}).get("name", "NewsAPI"),
						published_at=datetime.fromisoformat(
							article.get("publishedAt", "").replace("Z", "+00:00")
						),
						symbols=symbols,
						sentiment_score=0.0,
						sentiment_label="neutral",
						impact_score=0.0,
						keywords=[],
						relevance_score=0.0
					)
					news_items.append(news_item)
		
		except Exception as e:
			print(f"Error fetching NewsAPI news: {e}")
		
		return news_items
	
	def _fetch_alpha_vantage(self, symbols: List[str], hours_back: int) -> List[NewsItem]:
		"""Fetch news from Alpha Vantage"""
		
		if not self.api_keys.get("alpha_vantage"):
			return []
		
		news_items = []
		
		try:
			params = self.news_sources["alpha_vantage"]["params"].copy()
			params["tickers"] = ",".join(symbols[:3])  # Limit to 3 symbols
			params["limit"] = 50
			
			response = requests.get(
				self.news_sources["alpha_vantage"]["url"],
				params=params,
				timeout=10
			)
			
			if response.status_code == 200:
				data = response.json()
				
				for article in data.get("feed", []):
					news_item = NewsItem(
						title=article.get("title", ""),
						content=article.get("summary", ""),
						source="Alpha Vantage",
						published_at=datetime.fromisoformat(
							article.get("time_published", "").replace("Z", "+00:00")
						),
						symbols=article.get("ticker_sentiment", []),
						sentiment_score=float(article.get("overall_sentiment_score", 0)),
						sentiment_label=article.get("overall_sentiment_label", "neutral"),
						impact_score=float(article.get("relevance_score", 0)),
						keywords=[],
						relevance_score=float(article.get("relevance_score", 0))
					)
					news_items.append(news_item)
		
		except Exception as e:
			print(f"Error fetching Alpha Vantage news: {e}")
		
		return news_items
	
	def _process_news_queue(self):
		"""Background processing of news queue"""
		
		while self.processing_active:
			try:
				if self.news_queue:
					news_item = self.news_queue.pop(0)
					processed_news = self._analyze_news_sentiment(news_item)
					self.processed_news.append(processed_news)
					
					# Keep only recent news (last 7 days)
					cutoff_date = datetime.now() - timedelta(days=7)
					self.processed_news = [
						news for news in self.processed_news 
						if news.published_at > cutoff_date
					]
				
				time.sleep(1)  # Process every second
				
			except Exception as e:
				print(f"News processing error: {e}")
				time.sleep(5)
	
	def _analyze_news_sentiment(self, news_item: NewsItem) -> NewsItem:
		"""Perform comprehensive sentiment analysis on news item"""
		
		# Combine title and content
		text = f"{news_item.title} {news_item.content}"
		
		# Clean and preprocess text
		cleaned_text = self._preprocess_text(text)
		
		# Extract keywords
		keywords = self._extract_keywords(cleaned_text)
		news_item.keywords = keywords
		
		# Calculate relevance score
		relevance_score = self._calculate_relevance_score(cleaned_text, news_item.symbols)
		news_item.relevance_score = relevance_score
		
		# Calculate impact score
		impact_score = self._calculate_impact_score(keywords)
		news_item.impact_score = impact_score
		
		# Perform sentiment analysis
		sentiment_scores = self._calculate_sentiment_scores(cleaned_text)
		
		# Combine sentiment scores
		combined_score = (
			sentiment_scores["vader"] * 0.4 +
			sentiment_scores["textblob"] * 0.3 +
			sentiment_scores["keyword"] * 0.3
		)
		
		news_item.sentiment_score = combined_score
		news_item.sentiment_label = self._get_sentiment_label(combined_score)
		
		return news_item
	
	def _preprocess_text(self, text: str) -> str:
		"""Clean and preprocess text for analysis"""
		
		# Convert to lowercase
		text = text.lower()
		
		# Remove special characters and numbers
		text = re.sub(r'[^a-zA-Z\s]', '', text)
		
		# Tokenize
		tokens = word_tokenize(text)
		
		# Remove stop words
		tokens = [token for token in tokens if token not in self.stop_words]
		
		# Lemmatize
		tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
		
		return " ".join(tokens)
	
	def _extract_keywords(self, text: str) -> List[str]:
		"""Extract relevant keywords from text"""
		
		tokens = text.split()
		keywords = []
		
		for token in tokens:
			if token in self.forex_keywords:
				keywords.append(token)
		
		# Also extract multi-word phrases
		phrases = [
			"interest rate", "federal reserve", "central bank", "monetary policy",
			"fiscal policy", "australian dollar", "canadian dollar", "swiss franc",
			"balance of trade", "exchange rate"
		]
		
		for phrase in phrases:
			if phrase in text:
				keywords.append(phrase)
		
		return list(set(keywords))
	
	def _calculate_relevance_score(self, text: str, symbols: List[str]) -> float:
		"""Calculate relevance score for given symbols"""
		
		score = 0.0
		
		# Check for symbol-specific terms
		symbol_terms = {
			"EURUSD": ["euro", "dollar", "europe", "us", "ecb", "federal reserve"],
			"GBPUSD": ["pound", "sterling", "dollar", "uk", "britain", "us", "boe", "federal reserve"],
			"USDJPY": ["dollar", "yen", "us", "japan", "federal reserve", "bank of japan"],
			"EURGBP": ["euro", "pound", "europe", "uk", "britain", "ecb", "boe"]
		}
		
		for symbol in symbols:
			if symbol in symbol_terms:
				for term in symbol_terms[symbol]:
					if term in text:
						score += 0.1
		
		# Check for general forex terms
		forex_terms = ["forex", "currency", "exchange", "trade", "market"]
		for term in forex_terms:
			if term in text:
				score += 0.05
		
		return min(score, 1.0)
	
	def _calculate_impact_score(self, keywords: List[str]) -> float:
		"""Calculate impact score based on keywords"""
		
		score = 0.0
		
		for keyword in keywords:
			if keyword in self.forex_keywords:
				score += self.forex_keywords[keyword]
		
		return min(score, 1.0)
	
	def _calculate_sentiment_scores(self, text: str) -> Dict[str, float]:
		"""Calculate multiple sentiment scores"""
		
		scores = {}
		
		# VADER sentiment
		vader_scores = self.sentiment_analyzer.polarity_scores(text)
		scores["vader"] = vader_scores["compound"]
		
		# TextBlob sentiment
		blob = TextBlob(text)
		scores["textblob"] = blob.sentiment.polarity
		
		# Keyword-based sentiment
		positive_keywords = [
			"growth", "increase", "rise", "strong", "positive", "improve",
			"recovery", "expansion", "boost", "gain", "surge", "rally"
		]
		negative_keywords = [
			"decline", "fall", "drop", "weak", "negative", "worse",
			"recession", "crisis", "crash", "loss", "plunge", "slump"
		]
		
		positive_count = sum(1 for word in positive_keywords if word in text)
		negative_count = sum(1 for word in negative_keywords if word in text)
		
		if positive_count + negative_count > 0:
			scores["keyword"] = (positive_count - negative_count) / (positive_count + negative_count)
		else:
			scores["keyword"] = 0.0
		
		return scores
	
	def _get_sentiment_label(self, score: float) -> str:
		"""Convert sentiment score to label"""
		
		if score > 0.1:
			return "positive"
		elif score < -0.1:
			return "negative"
		else:
			return "neutral"
	
	def get_sentiment_summary(self, symbols: List[str], hours_back: int = 24) -> Dict:
		"""Get comprehensive sentiment summary for symbols"""
		
		# Fetch latest news
		news_items = self.fetch_news(symbols, hours_back)
		
		# Wait for processing
		time.sleep(2)
		
		# Filter processed news for symbols
		relevant_news = [
			news for news in self.processed_news
			if any(symbol in news.symbols for symbol in symbols)
		]
		
		if not relevant_news:
			return {
				"symbols": symbols,
				"total_news": 0,
				"avg_sentiment": 0.0,
				"sentiment_label": "neutral",
				"impact_score": 0.0,
				"relevance_score": 0.0,
				"news_count_by_sentiment": {"positive": 0, "neutral": 0, "negative": 0},
				"top_keywords": [],
				"recent_news": []
			}
		
		# Calculate summary statistics
		sentiment_scores = [news.sentiment_score for news in relevant_news]
		impact_scores = [news.impact_score for news in relevant_news]
		relevance_scores = [news.relevance_score for news in relevant_news]
		
		avg_sentiment = np.mean(sentiment_scores)
		avg_impact = np.mean(impact_scores)
		avg_relevance = np.mean(relevance_scores)
		
		# Count by sentiment
		sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
		for news in relevant_news:
			sentiment_counts[news.sentiment_label] += 1
		
		# Top keywords
		all_keywords = []
		for news in relevant_news:
			all_keywords.extend(news.keywords)
		
		keyword_counts = {}
		for keyword in all_keywords:
			keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
		
		top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
		
		# Recent news (last 5 items)
		recent_news = sorted(relevant_news, key=lambda x: x.published_at, reverse=True)[:5]
		
		return {
			"symbols": symbols,
			"total_news": len(relevant_news),
			"avg_sentiment": avg_sentiment,
			"sentiment_label": self._get_sentiment_label(avg_sentiment),
			"impact_score": avg_impact,
			"relevance_score": avg_relevance,
			"news_count_by_sentiment": sentiment_counts,
			"top_keywords": [{"keyword": k, "count": c} for k, c in top_keywords],
			"recent_news": [
				{
					"title": news.title,
					"source": news.source,
					"published_at": news.published_at.isoformat(),
					"sentiment_score": news.sentiment_score,
					"sentiment_label": news.sentiment_label,
					"impact_score": news.impact_score,
					"keywords": news.keywords
				}
				for news in recent_news
			]
		}
	
	def get_real_time_sentiment(self, symbols: List[str]) -> Dict:
		"""Get real-time sentiment for trading decisions"""
		
		# Get sentiment summary for last 4 hours
		sentiment_data = self.get_sentiment_summary(symbols, hours_back=4)
		
		# Calculate trading signal
		sentiment_score = sentiment_data["avg_sentiment"]
		impact_score = sentiment_data["impact_score"]
		relevance_score = sentiment_data["relevance_score"]
		
		# Combine scores for trading decision
		trading_score = (
			sentiment_score * 0.5 +
			impact_score * 0.3 +
			relevance_score * 0.2
		)
		
		# Generate trading signal
		if trading_score > 0.3:
			signal = "BUY"
			confidence = min(trading_score, 1.0)
		elif trading_score < -0.3:
			signal = "SELL"
			confidence = min(abs(trading_score), 1.0)
		else:
			signal = "HOLD"
			confidence = 0.5
		
		return {
			"symbols": symbols,
			"signal": signal,
			"confidence": confidence,
			"sentiment_score": sentiment_score,
			"impact_score": impact_score,
			"relevance_score": relevance_score,
			"trading_score": trading_score,
			"news_count": sentiment_data["total_news"],
			"timestamp": datetime.now().isoformat()
		}
	
	def stop_processing(self):
		"""Stop background processing"""
		
		self.processing_active = False
		if self.processing_thread.is_alive():
			self.processing_thread.join(timeout=5)