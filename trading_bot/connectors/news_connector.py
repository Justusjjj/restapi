from newsapi import NewsApiClient
import logging
from datetime import datetime, timedelta

# Local application imports
import config

class NewsConnector:
    """
    A connector to fetch financial news using the NewsAPI.
    """
    def __init__(self):
        self.api_key = config.NEWS_API_KEY
        if self.api_key == "YOUR_API_KEY" or not self.api_key:
            logging.warning("NewsAPI key not provided in config.py. News functionality will be disabled.")
            self.api_client = None
        else:
            self.api_client = NewsApiClient(api_key=self.api_key)

    def get_latest_forex_news(self, keywords=['forex', 'currency', 'USD', 'EUR', 'GBP', 'JPY', 'inflation', 'interest rate']):
        """
        Fetches the latest news articles related to a list of keywords.

        :param keywords: A list of strings to search for in news articles.
        :return: A list of articles, or None if the API is not configured.
        """
        if not self.api_client:
            return None

        try:
            # Search for articles from the last 24 hours
            from_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')

            all_articles = self.api_client.get_everything(
                q=' OR '.join(keywords),
                language='en',
                sort_by='publishedAt',
                from_param=from_date,
                page_size=20 # Limit to the 20 most recent articles
            )
            return all_articles['articles']

        except Exception as e:
            # Handle potential API errors (e.g., invalid key, rate limits)
            logging.error(f"Failed to fetch news from NewsAPI: {e}")
            return None

    def is_high_impact_news_imminent(self, articles):
        """
        A simple heuristic to check if high-impact news was published recently.

        :param articles: A list of news articles from the API.
        :return: True if high-impact news is found, False otherwise.
        """
        if not articles:
            return False

        # Define keywords that might indicate high-impact news
        high_impact_keywords = [
            'rate decision', 'non-farm payroll', 'nfp', 'cpi', 'inflation report',
            'fed meeting', 'ecb meeting', 'boe meeting', 'boj meeting', 'gdp report'
        ]

        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()

            # Check if any high-impact keyword is in the title or description
            for keyword in high_impact_keywords:
                if keyword in title or keyword in description:
                    logging.warning(f"High-impact news detected: '{article['title']}'")
                    return True

        return False
