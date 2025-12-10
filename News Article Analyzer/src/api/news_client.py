"""
# NewsClient: Handles Indian + International News Sources
# Why dual sources? Compare regional bias & global perspectives
"""

import logging
from gnews import GNews
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal
from dateutil import parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsClient():
    """
    This Client will Fetch news articales from newsapi.org
    
    Why NewsApi Client Library?
    - Official Python Client (handles authentication automatically)
    - Built-in pagination % error handling
    - Similar to tweepy for Twitter API
    """
    
    def __init__(self):
        """
        Initialize the GNews Client
        """
        self.default_language = 'en'
        self.default_period = '30d'

        self.requests_today = 0
        self.last_request_time = None
    

    def _log_request(self):
        self.requests_today += 1
        print(f"📊 Google News Requests so far: {self.requests_today}")

    def search_articles(
        self,
        query: str,
        region: Literal['indian', 'international', 'both'] = 'both',
        sources: Optional[List[str]] = None, 
        from_date: Optional[datetime] = None, 
        to_date: Optional[datetime] = None,
        strict_match: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Search for news articles by keyword and region
        
        Args: 
            query: Search Keywors (e.g., "artificial intelligence")
            region: Article Region (e.g., "indian", "international", "both")
            sources: List of news source (e.g., ["bbc-news", "cnn"])
            from_date: Start date for artices
            to_date: End date for artices
            strict_match: Boolean to match all keywords
            
        Returns:
            List of article dictionaries with separate lists
        """
        
        self._log_request()
        print(f"⏰ Smart Searching for: '{query}'")
        
        results = {
            'indian': [],
            'international': [],
            'combined': [],
        }
        
        try:
            # Fetch Indian news articles
            # Setting Up GNews for Indian News
            if region in ['indian', 'both']:
                google_in = GNews(language='en', country='IN', period=self.default_period, max_results=20)
                ind_news = google_in.get_news(query)

                for article in ind_news:
                    fmt = self.format_article(article, region='indian')
                    results['indian'].append(fmt)
                    results['combined'].append(fmt)
                
                print(f"🇮🇳 Total: {len(results['indian'])} Indian articles")
                                            
            # Fetching International news
            if region in ['international', 'both']:
                countries = ['US', 'GB', 'CA', 'AU', 'NZ', 'SG', 'MY', 'PH', 'ID', 'TH', 'VN', 'IN']
                
                def fetch_country(country_code):
                    try:
                        g_intl = GNews(language='en', country=country_code, period=self.default_period, max_results=10)
                        return g_intl.get_news(query)
                    except Exception as e:
                        print(f"⚠️ Failed to fetch for {country_code}: {e}")
                        return []

                import concurrent.futures
                intl_articles = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_country = {executor.submit(fetch_country, code): code for code in countries}

                    for future in concurrent.futures.as_completed(future_to_country):
                        intl_articles.extend(future.result())

                    seen_urls = set()
                    unique_intl = []

                    for art in intl_articles:
                        url = art.get('url')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            unique_intl.append(art)
                            
                    for article in unique_intl:
                        fmt = self.format_article(article, region='international')
                        results['international'].append(fmt)
                        results['combined'].append(fmt)
                
                print(f"� Fetched {len(results['international'])} International articles")                      
                
            print(f" ✅Fond {len(results['combined'])} articles via Google News")
            
            return results
        
        except Exception as e:
            print(f"❌ Error fetching from Google News: {e}")
            return results
        

    def get_sources_by_region(self, region='both'):
        """
        Mock function to keep app.py happy.
        Google News doesn't really work by 'source list' the same way.
        """
        dummy_sources = [
            {'id': 'google-news-in', 'name': 'Google News (India)'},
            {'id': 'international-news', 'name': 'Global Source'}
        ]
        return {'indian': dummy_sources, 'international': dummy_sources}


    def format_article(self, article: Dict, region: str = 'unknown') -> Dict:
        """
        Makes the format of the article suitable for processing

        Why format?
        - GNews return inconsistent data (some fieldsmay be None)
        - Prepare for data_adapter conversion to suitable format
        - Clean up unnecessary fields

        Args:
            article: Raw articale from GNews
            region: Region of the article

        Returns:
            Cleaned article dictionary
        """

        publisher = article.get('publisher', {})
        source_name = publisher.get('title', 'Google News') if isinstance(publisher, dict) else 'Google News'

        raw_date = article.get('published date', '')
        try:
            if raw_date:
                dt = parser.parse(raw_date)
                pub_date = dt.isoformat()
            else:
                pub_date = datetime.now().isoformat()
        except:
            pub_date = datetime.now().isoformat()
                

        return {
            'title': article.get('title', 'No Title'),
            'description': article.get('description', article.get('title', '')),
            'content': article.get('description', ''),
            'url': article.get('url', ''),
            'source': source_name,
            'author': source_name,
            'published_at': pub_date,
            'image_url': '',
            'region': region
        }

if __name__ == "__main__":
    client = NewsClient()
    
    print("🔍 Testing article search...")
    articles = client.search_articles(
        query="technology",
        region='both',
        from_date=datetime.now() - timedelta(days=2),
        to_date=datetime.now() - timedelta(days=1)
    )
    
    print(f"\n✅ Results:")
    print(f"   Indian: {len(articles['indian'])} articles")
    print(f"   International: {len(articles['international'])} articles")
    print(f"   Total: {len(articles['combined'])} articles")
    print(f"   API Usage: {client.requests_today}/100 requests")
    
    