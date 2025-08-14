#!/usr/bin/env python3
"""
News Filter and Social Media Sentiment Analysis System
Integrates news, Twitter, Reddit, and other sources for trading decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class NewsSentimentAnalyzer:
    """
    Advanced News Filter and Social Media Sentiment Analysis System
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.news_cache = {}
        self.sentiment_cache = {}
        self.impact_scores = {}
        self._setup_logging()
        
    def _default_config(self):
        """Default configuration for news and sentiment analysis"""
        return {
            'news_sources': ['forexfactory', 'fxstreet', 'investing', 'marketwatch'],
            'social_sources': ['twitter', 'reddit', 'telegram', 'discord'],
            'update_interval': 300,  # 5 minutes
            'cache_duration': 3600,  # 1 hour
            'sentiment_threshold': 0.6,
            'impact_threshold': 0.7,
            'max_news_age': 24,  # hours
            'api_keys': {},
            'risk_levels': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('news_sentiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_forex_news(self, symbols=None, hours_back=24):
        """
        Fetch forex news from multiple sources
        
        Args:
            symbols: List of currency pairs to filter
            hours_back: How many hours back to fetch
            
        Returns:
            DataFrame with news data
        """
        try:
            all_news = []
            
            # Fetch from ForexFactory (simulated)
            ff_news = self._fetch_forexfactory_news(symbols, hours_back)
            if ff_news:
                all_news.extend(ff_news)
            
            # Fetch from FXStreet (simulated)
            fxs_news = self._fetch_fxstreet_news(symbols, hours_back)
            if fxs_news:
                all_news.extend(fxs_news)
            
            # Fetch from Investing.com (simulated)
            inv_news = self._fetch_investing_news(symbols, hours_back)
            if inv_news:
                all_news.extend(inv_news)
            
            # Fetch from MarketWatch (simulated)
            mw_news = self._fetch_marketwatch_news(symbols, hours_back)
            if mw_news:
                all_news.extend(mw_news)
            
            if all_news:
                news_df = pd.DataFrame(all_news)
                news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
                news_df = news_df.sort_values('timestamp', ascending=False)
                
                # Cache the news
                self.news_cache['forex_news'] = {
                    'data': news_df,
                    'timestamp': datetime.now()
                }
                
                return news_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching forex news: {e}")
            return pd.DataFrame()
    
    def _fetch_forexfactory_news(self, symbols, hours_back):
        """Simulate fetching news from ForexFactory"""
        try:
            # Simulate API response
            sample_news = [
                {
                    'source': 'ForexFactory',
                    'title': 'ECB Interest Rate Decision - Euro Zone',
                    'summary': 'European Central Bank maintains interest rates at current levels',
                    'impact': 'high',
                    'currency': 'EUR',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'url': 'https://forexfactory.com/ecb-rate-decision',
                    'sentiment': 'neutral'
                },
                {
                    'source': 'ForexFactory',
                    'title': 'US Non-Farm Payrolls Report',
                    'summary': 'US employment data shows strong job growth',
                    'impact': 'high',
                    'currency': 'USD',
                    'timestamp': datetime.now() - timedelta(hours=6),
                    'url': 'https://forexfactory.com/nfp-report',
                    'sentiment': 'positive'
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_news = []
                for news in sample_news:
                    if any(symbol in news['currency'] for symbol in symbols):
                        filtered_news.append(news)
                return filtered_news
            
            return sample_news
            
        except Exception as e:
            self.logger.error(f"Error fetching ForexFactory news: {e}")
            return []
    
    def _fetch_fxstreet_news(self, symbols, hours_back):
        """Simulate fetching news from FXStreet"""
        try:
            sample_news = [
                {
                    'source': 'FXStreet',
                    'title': 'GBP/USD Technical Analysis: Pound Sterling gains momentum',
                    'summary': 'Technical indicators show bullish momentum for GBP/USD pair',
                    'impact': 'medium',
                    'currency': 'GBP',
                    'timestamp': datetime.now() - timedelta(hours=1),
                    'url': 'https://fxstreet.com/gbp-usd-analysis',
                    'sentiment': 'positive'
                },
                {
                    'source': 'FXStreet',
                    'title': 'USD/JPY: Dollar strengthens against Yen',
                    'summary': 'Risk-off sentiment drives USD/JPY higher',
                    'impact': 'medium',
                    'currency': 'USD',
                    'timestamp': datetime.now() - timedelta(hours=3),
                    'url': 'https://fxstreet.com/usd-jpy-analysis',
                    'sentiment': 'positive'
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_news = []
                for news in sample_news:
                    if any(symbol in news['currency'] for symbol in symbols):
                        filtered_news.append(news)
                return filtered_news
            
            return sample_news
            
        except Exception as e:
            self.logger.error(f"Error fetching FXStreet news: {e}")
            return []
    
    def _fetch_investing_news(self, symbols, hours_back):
        """Simulate fetching news from Investing.com"""
        try:
            sample_news = [
                {
                    'source': 'Investing.com',
                    'title': 'Federal Reserve Policy Meeting Minutes',
                    'summary': 'Fed minutes reveal cautious approach to future rate hikes',
                    'impact': 'high',
                    'currency': 'USD',
                    'timestamp': datetime.now() - timedelta(hours=4),
                    'url': 'https://investing.com/fed-minutes',
                    'sentiment': 'neutral'
                },
                {
                    'source': 'Investing.com',
                    'title': 'Bank of England Inflation Report',
                    'summary': 'BOE expects inflation to remain above target',
                    'impact': 'high',
                    'currency': 'GBP',
                    'timestamp': datetime.now() - timedelta(hours=8),
                    'url': 'https://investing.com/boe-inflation',
                    'sentiment': 'positive'
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_news = []
                for news in sample_news:
                    if any(symbol in news['currency'] for symbol in symbols):
                        filtered_news.append(news)
                return filtered_news
            
            return sample_news
            
        except Exception as e:
            self.logger.error(f"Error fetching Investing.com news: {e}")
            return []
    
    def _fetch_marketwatch_news(self, symbols, hours_back):
        """Simulate fetching news from MarketWatch"""
        try:
            sample_news = [
                {
                    'source': 'MarketWatch',
                    'title': 'Global Risk Sentiment Improves',
                    'summary': 'Markets show increased appetite for risk assets',
                    'impact': 'medium',
                    'currency': 'AUD',
                    'timestamp': datetime.now() - timedelta(hours=5),
                    'url': 'https://marketwatch.com/risk-sentiment',
                    'sentiment': 'positive'
                },
                {
                    'source': 'MarketWatch',
                    'title': 'Swiss National Bank Policy Decision',
                    'summary': 'SNB maintains negative interest rates',
                    'impact': 'medium',
                    'currency': 'CHF',
                    'timestamp': datetime.now() - timedelta(hours=7),
                    'url': 'https://marketwatch.com/snb-policy',
                    'sentiment': 'neutral'
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_news = []
                for news in sample_news:
                    if any(symbol in news['currency'] for symbol in symbols):
                        filtered_news.append(news)
                return filtered_news
            
            return sample_news
            
        except Exception as e:
            self.logger.error(f"Error fetching MarketWatch news: {e}")
            return []
    
    def analyze_news_sentiment(self, news_data):
        """
        Analyze sentiment of news articles
        
        Args:
            news_data: DataFrame with news data
            
        Returns:
            DataFrame with sentiment analysis
        """
        try:
            if news_data.empty:
                return pd.DataFrame()
            
            sentiment_results = []
            
            for _, news in news_data.iterrows():
                # Analyze title sentiment
                title_sentiment = self._analyze_text_sentiment(news['title'])
                
                # Analyze summary sentiment
                summary_sentiment = self._analyze_text_sentiment(news['summary'])
                
                # Calculate overall sentiment score
                overall_sentiment = (title_sentiment + summary_sentiment) / 2
                
                # Determine sentiment category
                if overall_sentiment > 0.6:
                    sentiment_category = 'positive'
                elif overall_sentiment < 0.4:
                    sentiment_category = 'negative'
                else:
                    sentiment_category = 'neutral'
                
                # Calculate impact score
                impact_score = self._calculate_news_impact(news)
                
                sentiment_results.append({
                    'source': news['source'],
                    'title': news['title'],
                    'currency': news['currency'],
                    'impact': news['impact'],
                    'timestamp': news['timestamp'],
                    'title_sentiment': title_sentiment,
                    'summary_sentiment': summary_sentiment,
                    'overall_sentiment': overall_sentiment,
                    'sentiment_category': sentiment_category,
                    'impact_score': impact_score,
                    'url': news['url']
                })
            
            sentiment_df = pd.DataFrame(sentiment_results)
            
            # Cache sentiment results
            self.sentiment_cache['news_sentiment'] = {
                'data': sentiment_df,
                'timestamp': datetime.now()
            }
            
            return sentiment_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return pd.DataFrame()
    
    def _analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text using keyword analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between 0 and 1
        """
        try:
            if not text:
                return 0.5
            
            text_lower = text.lower()
            
            # Positive keywords
            positive_words = [
                'bullish', 'gains', 'rises', 'strengthens', 'improves', 'positive',
                'growth', 'strong', 'higher', 'up', 'gain', 'rise', 'strength',
                'recovery', 'expansion', 'optimistic', 'favorable', 'beneficial'
            ]
            
            # Negative keywords
            negative_words = [
                'bearish', 'falls', 'declines', 'weakens', 'deteriorates', 'negative',
                'loss', 'weak', 'lower', 'down', 'fall', 'decline', 'weakness',
                'recession', 'contraction', 'pessimistic', 'unfavorable', 'harmful'
            ]
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment score
            total_words = positive_count + negative_count
            if total_words == 0:
                return 0.5  # Neutral if no sentiment words found
            
            sentiment_score = positive_count / total_words
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return 0.5
    
    def _calculate_news_impact(self, news):
        """
        Calculate impact score for news article
        
        Args:
            news: News article data
            
        Returns:
            Impact score between 0 and 1
        """
        try:
            impact_score = 0.0
            
            # Base impact from news source
            source_impact = {
                'forexfactory': 0.9,
                'fxstreet': 0.8,
                'investing': 0.8,
                'marketwatch': 0.7
            }
            
            impact_score += source_impact.get(news['source'], 0.5) * 0.3
            
            # Impact level multiplier
            impact_multiplier = {
                'high': 1.0,
                'medium': 0.7,
                'low': 0.4
            }
            
            impact_score += impact_multiplier.get(news['impact'], 0.5) * 0.4
            
            # Time decay factor
            time_diff = datetime.now() - news['timestamp']
            hours_old = time_diff.total_seconds() / 3600
            
            if hours_old <= 1:
                time_factor = 1.0
            elif hours_old <= 4:
                time_factor = 0.8
            elif hours_old <= 12:
                time_factor = 0.6
            else:
                time_factor = 0.4
            
            impact_score += time_factor * 0.3
            
            return min(1.0, impact_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating news impact: {e}")
            return 0.5
    
    def fetch_social_media_sentiment(self, symbols=None, hours_back=24):
        """
        Fetch social media sentiment from multiple sources
        
        Args:
            symbols: List of currency pairs to filter
            hours_back: How many hours back to fetch
            
        Returns:
            DataFrame with social media sentiment data
        """
        try:
            all_sentiment = []
            
            # Fetch Twitter sentiment (simulated)
            twitter_sentiment = self._fetch_twitter_sentiment(symbols, hours_back)
            if twitter_sentiment:
                all_sentiment.extend(twitter_sentiment)
            
            # Fetch Reddit sentiment (simulated)
            reddit_sentiment = self._fetch_reddit_sentiment(symbols, hours_back)
            if reddit_sentiment:
                all_sentiment.extend(reddit_sentiment)
            
            # Fetch Telegram sentiment (simulated)
            telegram_sentiment = self._fetch_telegram_sentiment(symbols, hours_back)
            if telegram_sentiment:
                all_sentiment.extend(telegram_sentiment)
            
            # Fetch Discord sentiment (simulated)
            discord_sentiment = self._fetch_discord_sentiment(symbols, hours_back)
            if discord_sentiment:
                all_sentiment.extend(discord_sentiment)
            
            if all_sentiment:
                sentiment_df = pd.DataFrame(all_sentiment)
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                sentiment_df = sentiment_df.sort_values('timestamp', ascending=False)
                
                # Cache sentiment results
                self.sentiment_cache['social_sentiment'] = {
                    'data': sentiment_df,
                    'timestamp': datetime.now()
                }
                
                return sentiment_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching social media sentiment: {e}")
            return pd.DataFrame()
    
    def _fetch_twitter_sentiment(self, symbols, hours_back):
        """Simulate fetching Twitter sentiment"""
        try:
            sample_tweets = [
                {
                    'source': 'Twitter',
                    'username': '@ForexTrader',
                    'content': 'EUR/USD looking bullish today! Strong support at 1.0950 #forex #EURUSD',
                    'currency': 'EUR',
                    'timestamp': datetime.now() - timedelta(hours=1),
                    'sentiment_score': 0.8,
                    'engagement': 150,
                    'verified': True
                },
                {
                    'source': 'Twitter',
                    'username': '@FXAnalyst',
                    'content': 'GBP/USD breaks key resistance. Momentum building for further gains #GBPUSD #forex',
                    'currency': 'GBP',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'sentiment_score': 0.7,
                    'engagement': 89,
                    'verified': True
                },
                {
                    'source': 'Twitter',
                    'username': '@MarketWatcher',
                    'content': 'USD/JPY struggling to maintain gains. Risk-off sentiment returning #USDJPY',
                    'currency': 'USD',
                    'timestamp': datetime.now() - timedelta(hours=3),
                    'sentiment_score': 0.3,
                    'engagement': 67,
                    'verified': False
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_tweets = []
                for tweet in sample_tweets:
                    if any(symbol in tweet['currency'] for symbol in symbols):
                        filtered_tweets.append(tweet)
                return filtered_tweets
            
            return sample_tweets
            
        except Exception as e:
            self.logger.error(f"Error fetching Twitter sentiment: {e}")
            return []
    
    def _fetch_reddit_sentiment(self, symbols, hours_back):
        """Simulate fetching Reddit sentiment"""
        try:
            sample_posts = [
                {
                    'source': 'Reddit',
                    'subreddit': 'r/Forex',
                    'username': 'u/ForexEnthusiast',
                    'title': 'EUR/USD Analysis - Bullish breakout confirmed',
                    'content': 'Technical analysis shows EUR/USD has broken above key resistance...',
                    'currency': 'EUR',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'sentiment_score': 0.8,
                    'upvotes': 45,
                    'comments': 12
                },
                {
                    'source': 'Reddit',
                    'subreddit': 'r/Forex',
                    'username': 'u/TradingPro',
                    'title': 'GBP/USD: Caution advised ahead of BOE meeting',
                    'content': 'Market uncertainty around Bank of England decision...',
                    'currency': 'GBP',
                    'timestamp': datetime.now() - timedelta(hours=4),
                    'sentiment_score': 0.4,
                    'upvotes': 23,
                    'comments': 8
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_posts = []
                for post in sample_posts:
                    if any(symbol in post['currency'] for symbol in symbols):
                        filtered_posts.append(post)
                return filtered_posts
            
            return sample_posts
            
        except Exception as e:
            self.logger.error(f"Error fetching Reddit sentiment: {e}")
            return []
    
    def _fetch_telegram_sentiment(self, symbols, hours_back):
        """Simulate fetching Telegram sentiment"""
        try:
            sample_messages = [
                {
                    'source': 'Telegram',
                    'channel': 'Forex Signals Pro',
                    'username': 'Admin',
                    'content': '🚀 EUR/USD BUY signal! Entry: 1.0980, SL: 1.0950, TP: 1.1050',
                    'currency': 'EUR',
                    'timestamp': datetime.now() - timedelta(hours=1),
                    'sentiment_score': 0.9,
                    'members': 5000
                },
                {
                    'source': 'Telegram',
                    'channel': 'FX Trading Group',
                    'username': 'Moderator',
                    'content': '⚠️ GBP/USD: Wait for confirmation before entering new positions',
                    'currency': 'GBP',
                    'timestamp': datetime.now() - timedelta(hours=3),
                    'sentiment_score': 0.5,
                    'members': 3000
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_messages = []
                for message in sample_messages:
                    if any(symbol in message['currency'] for symbol in symbols):
                        filtered_messages.append(message)
                return filtered_messages
            
            return sample_messages
            
        except Exception as e:
            self.logger.error(f"Error fetching Telegram sentiment: {e}")
            return []
    
    def _fetch_discord_sentiment(self, symbols, hours_back):
        """Simulate fetching Discord sentiment"""
        try:
            sample_messages = [
                {
                    'source': 'Discord',
                    'server': 'Forex Trading Community',
                    'username': 'Trader#1234',
                    'content': 'USD/JPY showing strong momentum. Risk-reward looks good here',
                    'currency': 'USD',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'sentiment_score': 0.7,
                    'reactions': 5
                },
                {
                    'source': 'Discord',
                    'server': 'FX Analysis Hub',
                    'username': 'Analyst#5678',
                    'content': 'AUD/USD: Mixed signals from technical indicators. Sideways movement expected',
                    'currency': 'AUD',
                    'timestamp': datetime.now() - timedelta(hours=5),
                    'sentiment_score': 0.5,
                    'reactions': 3
                }
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_messages = []
                for message in sample_messages:
                    if any(symbol in message['currency'] for symbol in symbols):
                        filtered_messages.append(message)
                return filtered_messages
            
            return sample_messages
            
        except Exception as e:
            self.logger.error(f"Error fetching Discord sentiment: {e}")
            return []
    
    def calculate_composite_sentiment(self, news_sentiment, social_sentiment):
        """
        Calculate composite sentiment score combining news and social media
        
        Args:
            news_sentiment: News sentiment DataFrame
            social_sentiment: Social media sentiment DataFrame
            
        Returns:
            Dictionary with composite sentiment scores
        """
        try:
            composite_scores = {}
            
            # Get unique currencies
            currencies = set()
            if not news_sentiment.empty:
                currencies.update(news_sentiment['currency'].unique())
            if not social_sentiment.empty:
                currencies.update(social_sentiment['currency'].unique())
            
            for currency in currencies:
                # News sentiment for this currency
                currency_news = news_sentiment[news_sentiment['currency'] == currency] if not news_sentiment.empty else pd.DataFrame()
                currency_social = social_sentiment[social_sentiment['currency'] == currency] if not social_sentiment.empty else pd.DataFrame()
                
                # Calculate weighted news sentiment
                news_score = 0.0
                news_weight = 0.0
                if not currency_news.empty:
                    for _, news in currency_news.iterrows():
                        weight = news['impact_score']
                        news_score += news['overall_sentiment'] * weight
                        news_weight += weight
                    
                    if news_weight > 0:
                        news_score = news_score / news_weight
                
                # Calculate weighted social sentiment
                social_score = 0.0
                social_weight = 0.0
                if not currency_social.empty:
                    for _, social in currency_social.iterrows():
                        # Weight by engagement/verification
                        if social['source'] == 'Twitter':
                            weight = 1.0 if social.get('verified', False) else 0.7
                            weight *= min(social.get('engagement', 0) / 100, 2.0)  # Cap at 2x
                        elif social['source'] == 'Reddit':
                            weight = min(social.get('upvotes', 0) / 50, 2.0)  # Cap at 2x
                        elif social['source'] == 'Telegram':
                            weight = min(social.get('members', 0) / 5000, 1.5)  # Cap at 1.5x
                        else:  # Discord
                            weight = min(social.get('reactions', 0) / 10, 1.0)  # Cap at 1x
                        
                        social_score += social['sentiment_score'] * weight
                        social_weight += weight
                    
                    if social_weight > 0:
                        social_score = social_score / social_weight
                
                # Calculate composite score (70% news, 30% social)
                if news_weight > 0 and social_weight > 0:
                    composite_score = 0.7 * news_score + 0.3 * social_score
                elif news_weight > 0:
                    composite_score = news_score
                elif social_weight > 0:
                    composite_score = social_score
                else:
                    composite_score = 0.5  # Neutral if no data
                
                # Determine sentiment category
                if composite_score > 0.6:
                    sentiment_category = 'bullish'
                elif composite_score < 0.4:
                    sentiment_category = 'bearish'
                else:
                    sentiment_category = 'neutral'
                
                composite_scores[currency] = {
                    'composite_score': composite_score,
                    'news_score': news_score,
                    'social_score': social_score,
                    'sentiment_category': sentiment_category,
                    'confidence': min(news_weight + social_weight, 1.0),
                    'last_updated': datetime.now()
                }
            
            return composite_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating composite sentiment: {e}")
            return {}
    
    def generate_trading_signals(self, composite_sentiment, market_data=None):
        """
        Generate trading signals based on sentiment analysis
        
        Args:
            composite_sentiment: Composite sentiment scores
            market_data: Optional market data for confirmation
            
        Returns:
            Dictionary with trading signals
        """
        try:
            trading_signals = {}
            
            for currency, sentiment_data in composite_sentiment.items():
                signal = {
                    'currency': currency,
                    'sentiment': sentiment_data['sentiment_category'],
                    'confidence': sentiment_data['confidence'],
                    'action': 'HOLD',
                    'strength': 0.0,
                    'reasoning': [],
                    'risk_level': 'medium',
                    'timestamp': datetime.now()
                }
                
                # Determine trading action based on sentiment
                sentiment_score = sentiment_data['composite_score']
                confidence = sentiment_data['confidence']
                
                if confidence < 0.3:
                    signal['action'] = 'HOLD'
                    signal['reasoning'].append('Insufficient sentiment data')
                elif sentiment_score > 0.7 and confidence > 0.5:
                    signal['action'] = 'BUY'
                    signal['strength'] = min(sentiment_score * confidence, 1.0)
                    signal['reasoning'].append('Strong bullish sentiment')
                elif sentiment_score < 0.3 and confidence > 0.5:
                    signal['action'] = 'SELL'
                    signal['strength'] = min((1 - sentiment_score) * confidence, 1.0)
                    signal['reasoning'].append('Strong bearish sentiment')
                else:
                    signal['action'] = 'HOLD'
                    signal['strength'] = 0.0
                    signal['reasoning'].append('Neutral or mixed sentiment')
                
                # Determine risk level
                if confidence > 0.8 and abs(sentiment_score - 0.5) > 0.4:
                    signal['risk_level'] = 'low'
                elif confidence < 0.5 or abs(sentiment_score - 0.5) < 0.2:
                    signal['risk_level'] = 'high'
                else:
                    signal['risk_level'] = 'medium'
                
                # Add market data confirmation if available
                if market_data and currency in market_data:
                    # Simple confirmation logic (can be enhanced)
                    if signal['action'] != 'HOLD':
                        signal['reasoning'].append('Market data confirmation available')
                
                trading_signals[currency] = signal
            
            return trading_signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return {}
    
    def get_sentiment_summary(self, symbols=None):
        """
        Get comprehensive sentiment summary
        
        Args:
            symbols: List of currencies to include
            
        Returns:
            Dictionary with sentiment summary
        """
        try:
            summary = {
                'timestamp': datetime.now(),
                'overall_market_sentiment': 'neutral',
                'currency_sentiments': {},
                'top_news': [],
                'trending_topics': [],
                'risk_assessment': 'medium'
            }
            
            # Get cached data
            news_sentiment = self.sentiment_cache.get('news_sentiment', {}).get('data', pd.DataFrame())
            social_sentiment = self.sentiment_cache.get('social_sentiment', {}).get('data', pd.DataFrame())
            
            if not news_sentiment.empty or not social_sentiment.empty:
                # Calculate composite sentiment
                composite_sentiment = self.calculate_composite_sentiment(news_sentiment, social_sentiment)
                
                # Filter by symbols if provided
                if symbols:
                    composite_sentiment = {k: v for k, v in composite_sentiment.items() if k in symbols}
                
                # Calculate overall market sentiment
                if composite_sentiment:
                    overall_score = np.mean([data['composite_score'] for data in composite_sentiment.values()])
                    if overall_score > 0.6:
                        summary['overall_market_sentiment'] = 'bullish'
                    elif overall_score < 0.4:
                        summary['overall_market_sentiment'] = 'bearish'
                    else:
                        summary['overall_market_sentiment'] = 'neutral'
                    
                    summary['currency_sentiments'] = composite_sentiment
                
                # Get top news
                if not news_sentiment.empty:
                    top_news = news_sentiment.nlargest(5, 'impact_score')
                    summary['top_news'] = top_news[['title', 'currency', 'impact', 'sentiment_category']].to_dict('records')
                
                # Get trending topics (simplified)
                summary['trending_topics'] = [
                    'ECB Policy Decision',
                    'US Economic Data',
                    'Risk Sentiment',
                    'Technical Breakouts',
                    'Central Bank Communications'
                ]
                
                # Risk assessment
                if composite_sentiment:
                    high_confidence_count = sum(1 for data in composite_sentiment.values() if data['confidence'] > 0.7)
                    total_currencies = len(composite_sentiment)
                    
                    if high_confidence_count / total_currencies > 0.7:
                        summary['risk_assessment'] = 'low'
                    elif high_confidence_count / total_currencies < 0.3:
                        summary['risk_assessment'] = 'high'
                    else:
                        summary['risk_assessment'] = 'medium'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment summary: {e}")
            return {}
    
    def update_sentiment_data(self, symbols=None):
        """
        Update all sentiment data
        
        Args:
            symbols: List of currencies to update
            
        Returns:
            Boolean indicating success
        """
        try:
            # Fetch latest news
            news_data = self.fetch_forex_news(symbols)
            if not news_data.empty:
                news_sentiment = self.analyze_news_sentiment(news_data)
            
            # Fetch latest social media sentiment
            social_sentiment = self.fetch_social_media_sentiment(symbols)
            
            # Calculate composite sentiment
            if 'news_sentiment' in locals() and not news_sentiment.empty:
                composite_sentiment = self.calculate_composite_sentiment(news_sentiment, social_sentiment)
                
                # Generate trading signals
                trading_signals = self.generate_trading_signals(composite_sentiment)
                
                # Cache trading signals
                self.sentiment_cache['trading_signals'] = {
                    'data': trading_signals,
                    'timestamp': datetime.now()
                }
            
            self.logger.info("Sentiment data updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating sentiment data: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    analyzer = NewsSentimentAnalyzer()
    
    # Fetch news and sentiment
    print("Fetching forex news...")
    news_data = analyzer.fetch_forex_news(['EUR', 'GBP', 'USD'])
    
    print("Analyzing news sentiment...")
    news_sentiment = analyzer.analyze_news_sentiment(news_data)
    
    print("Fetching social media sentiment...")
    social_sentiment = analyzer.fetch_social_media_sentiment(['EUR', 'GBP', 'USD'])
    
    print("Calculating composite sentiment...")
    composite_sentiment = analyzer.calculate_composite_sentiment(news_sentiment, social_sentiment)
    
    print("Generating trading signals...")
    trading_signals = analyzer.generate_trading_signals(composite_sentiment)
    
    print("Getting sentiment summary...")
    summary = analyzer.get_sentiment_summary(['EUR', 'GBP', 'USD'])
    
    # Display results
    print("\n=== NEWS SENTIMENT ===")
    print(news_sentiment[['currency', 'title', 'sentiment_category', 'impact_score']].head())
    
    print("\n=== SOCIAL SENTIMENT ===")
    print(social_sentiment[['source', 'currency', 'sentiment_score']].head())
    
    print("\n=== COMPOSITE SENTIMENT ===")
    for currency, data in composite_sentiment.items():
        print(f"{currency}: {data['sentiment_category']} (Score: {data['composite_score']:.3f})")
    
    print("\n=== TRADING SIGNALS ===")
    for currency, signal in trading_signals.items():
        print(f"{currency}: {signal['action']} - {signal['sentiment']} (Strength: {signal['strength']:.3f})")
    
    print("\n=== SENTIMENT SUMMARY ===")
    print(f"Overall Market Sentiment: {summary['overall_market_sentiment']}")
    print(f"Risk Assessment: {summary['risk_assessment']}")
    print(f"Top News: {len(summary['top_news'])} articles")