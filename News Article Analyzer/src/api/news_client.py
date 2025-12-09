"""
# NewsClient: Handles Indian + International News Sources
# Why dual sources? Compare regional bias & global perspectives
"""

import logging
from gnews import GNews
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal
import time

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
        print("⏰ Smart Searching for: ")
        
        result = {
            'indian': [],
            'international': [],
            'combined': [],
        }
        
        try:
            
            def fetch_batch(sources_list, region_type):
                articles = []
                batch_size = 20
                
                for i in range(0, len(sources_list), batch_size):
                    batch = sources_list[i:i + batch_size]
                    
                    try:
                        self._check_rate_limit()
                        
                        response = self.client.get_everything(
                            q=api_query,
                            sources=','.join(batch),  
                            from_param=from_param,
                            to=to_param,
                            language=language,
                            sort_by=sort_by,
                            page_size=self.default_page_size
                        )
                        
                        self._log_request()
                        
                        if response.get('status') == 'ok':
                            batch_articles = response.get('articles', [])
                            
                            relevant_articles = []
                            
                            for article in batch_articles:
                                
                                if strict_match:
                                    if self._article_contains_all_keywords  (article, keywords):
                                        article['region'] = region_type
                                        article['formatted'] = self.format_article(article)
                                        relevant_articles.append(article)
                                else:
                                    article['region'] = region_type
                                    article['formatted'] = self.format_article(article)
                                    relevant_articles.append(article)
                            
                            articles.extend(relevant_articles)
                            print(f" ✅ Batch {i//batch_size + 1}: {len(relevant_articles)}/{len(batch_articles)}  relevant articles")
                            
                        elif response.get('code') == 'rateLimited':
                            print(f" ⚠️ Rate limit hit! Requests used: {self.requests_today}/100")
                            break
                        
                        time.sleep(1)
                    
                    except Exception as e:
                        print(f" ⚠️ Batch error: {e}")
                        continue
                
                return articles
            
            # Fetch Indian news articles
            if region in ['indian', 'both']:
                
                if sources:
                    indian_sources = [s for s in sources if s in self.INDIAN_SOURCES]
                else:
                    indian_sources = self.INDIAN_SOURCES
                    
                if indian_sources:
                    result['indian'] = fetch_batch(indian_sources, 'indian')
                    result['combined'].extend(result['indian'])
                    print(f"🇮🇳 Total: {len(result['indian'])} Indian articles")
                                            
            # Fetch International news
            if region in ['international', 'both']:
                
                if sources:
                    intl_sources = [s for s in sources if s in self.INTERNATIONAL_SOURCES]
                else:
                    intl_sources = self.INTERNATIONAL_SOURCES
                    
                if intl_sources:
                    result['international'] = fetch_batch(intl_sources, 'international')
                    result['combined'].extend(result['international'])
                    print(f"🌍 Total: {len(result['international'])} International articles")                        
                
            import random
            random.shuffle(result['combined'])
            
            return result
        
        except Exception as e:
            print(f"❌ Error fetching articles: {e}")
            return result
        
    def get_top_headlines(
        self,
        category: Optional[str] = None,
        country: str = 'us',
        sources: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        This gets current top headlines
        
        Why separate method?
        - Top headlines use different API endpoint
        - Different rate limits (faster updates)
        - Useful for "trending news" feature
        
        Args:
            categroy: News Category ('business', 'technology', etc.)
            country: Country code (e.g., 'us', 'gb')
            sources: Specific news sources
        
        Returns:
            List of headline articles
         """
         
        try:
            response = self.client.get_top_headlines(
                category=category,
                country=country,
                sources=','.join(sources) if sources else None,
                page_size=self.default_page_size
            )
            
            return response.get('articles', [])
        
        except Exception as e:
            print(f"❌ Error fetching headlines: {e}")
            return []
         
    def get_sources_by_region(
        self,
        region: Literal['indian', 'international', 'both'] = 'both'
    ) -> Dict[str, List[Dict]]:
        """
        This gets available sources by region
        """
        
        results = {
            'indian': [],
            'international': [],
        }
        
        try:
            self._check_rate_limit()
            
            all_sources_response = self.client.get_sources(language='en')
            self._log_request()
            
            all_sources = all_sources_response.get('sources', [])
            
            for source in all_sources:
                source_id = source.get('id', '')
                
                if source_id in self.INDIAN_SOURCES:
                    results['indian'].append(source)
                elif source_id in self.INTERNATIONAL_SOURCES:
                    results['international'].append(source)
                    
            return results
        
        except Exception as e:
            print(f"❌ Error fetching sources: {e}")
            return results
         
    def get_sources(
        self,
        category: Optional[str] = None,
        language: str = None,
        country: str = None,
    ) -> List[Dict]:
        """
        This helps in getting list of available news sources

        Why needed?
        - Populate sources selection dropdown in UI
        - Filter by category/language/country 

        Args:
            category: Filter by category
            language: Filter by language
            country: Filter by country

        Returns:
            List of source dictionaries with id, name, description
        """

        try:
            response = self.client.get_sources(
                category=category,
                language=language or self.client.default_language,
                country=country
            )

            return response.get('sources', [])

        except Exception as e:
            print(f"❌ Error fetching sources: {e}")
            return []

    def format_article(self, article: Dict) -> Dict:
        """
        Makes the format of the article suitable for processing

        Why format?
        - API return inconsistent data (some fieldsmay be None)
        - Prepare for data_adapter conversion to suitable format
        - Clean up unnecessary fields

        Args:
            article: Raw articale from API

        Returns:
            Cleaned article dictionary
        """

        return {
            'title': article.get('title', 'No Title'),
            'description': article.get('description', ''),
            'content': article.get('content', ''),
            'url': article.get('url', ''),
            'source': article.get('source', {}).get('name', 'Unknown'),
            'author': article.get('author', 'Unknown'),
            'published_at': article.get('publishedAt', ''),
            'image_url': article.get('urlToImage', ''),
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
    
    