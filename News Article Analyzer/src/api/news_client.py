"""
# NewsClient: Handles Indian + International News Sources
# Why dual sources? Compare regional bias & global perspectives
"""

import os
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv
import time

# Load environment variables from .env files 
load_dotenv()

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
        Initialize the News API Client
        
        Why load from .env?
        - Security: API keys shouldn't bein code
        - Flexibility: Easy to change without code changes
        """
        api_key = os.getenv('NEWS_API_KEY')
        
        if not api_key:
            raise ValueError("❌ NEWS_API_KEY not found in .env file!")
        
        self.client = NewsApiClient(api_key=api_key)
        self.default_language = os.getenv('DEFAULT_LANGUAGE', 'en')
        self.default_page_size = int(os.getenv('DEFAULT_PAGE_SIZE', 100))
        
        self.requests_today = 0
        self.last_request_time = None
        
    # ✅ VERIFIED INDIAN SOURCES (15+ channels)
    INDIAN_SOURCES = [
        # Native Indian Publishers
        'the-times-of-india',
        'the-hindu',
        'google-news-in',
        
        # Sports (Cricket essential for India)
        'espn-cric-info',
        
        # Business & Finance (High Indian readership)
        'business-insider',
        'reuters',
        'bloomberg',
        'fortune',
        
        # Technology
        'techcrunch',
        'the-verge',
        'wired',
        'hacker-news',
        # Global with Indian coverage
        'bbc-news',
        'cnn',
        'al-jazeera-english',
        'time',
        'national-geographic'
    ]
    
    # ✅ VERIFIED INTERNATIONAL SOURCES (15+ channels)
    INTERNATIONAL_SOURCES = [
        # US & Global Major News
        'associated-press',
        'abc-news',
        'bbc-news',
        'cnn',
        'fox-news',
        'nbc-news',
        'usa-today',
        'the-washington-post',
        'the-wall-street-journal',
        'politico',
        
        # Tech & Science
        'engadget',
        'new-scientist',
        'next-big-future',
        'recode',
        'ars-technica',
        
        # Entertainment & Gaming
        'entertainment-weekly',
        'ign',
        'polygon',
        'mashable',
        'vice-news'
    ]
    
    def _check_rate_limit(self):
        if self.requests_today >= 90:
            print(f"⚠️ WARNING: {self.requests_today}/100 requests used today!")
            
        if self.requests_today >= 100:
            raise Exception("❌ Daily rate limit (100 req) exceeded! Try again tomorrow")
        
    def _log_request(self):
        self.requests_today += 1
        self.last_request_time = datetime.now()
        print(f"📊 API Requests: {self.requests_today}")
        
    def _parse_keywords(self, query: str) -> List[str]:
        """
        Parse user query into individual keywords
        """
        stop_words = {'and', 'or', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'a', 'an'}
        
        keywords = query.lower().split()
        keywords = [k.strip() for k in keywords if k.strip() not in stop_words]
        
        return keywords
        
    def _build_api_query(self, keywords: List[str]) -> str:
        """
        Build News API query with OR logic for broader results
        """
        
        return ' OR '.join(keywords)
    
    def _article_contains_all_keywords(
        self, article: Dict, keywords: List[str]
    ) -> bool:
        """
        Check if article contains ALL keywords in title.description
        """
        
        title = (article.get('title', '') or '').lower()
        description = (article.get('description', '') or '').lower()
        
        combined_text = f"{title} {description}"
        
        matched_count = 0
        for keyword in keywords:
            if any(keyword in word for word in combined_text.split()):
                matched_count += 1
                
        required_matches = max(1, int(len(keywords) * 0.7))
        
        return matched_count >= required_matches
        
    def search_articles(
        self,
        query: str,
        region: Literal['indian', 'international', 'both'] = 'both',
        sources: Optional[List[str]] = None, 
        from_date: Optional[datetime] = None, 
        strict_match: bool = True,
        to_date: Optional[datetime] = None,
        language: str = None,
        sort_by: str = 'publishedAt',
    ) -> Dict[str, List[Dict]]:
        """
        Search for news articles by keyword and region
        
        Args: 
            query: Search Keywors (e.g., "artificial intelligence")
            region: Article Region (e.g., "indian", "international", "both")
            sources: List of news source (e.g., ["bbc-news", "cnn"])
            from_date: Start date for artices
            to_date: End date for artices
            languages: Article language (default: 'en')
            sort_by: Sort order ('publishedAt', 'relevancy', 'popularity')
            
        Returns:
            List of article dictionaries with separate lists
        """
        
        self._check_rate_limit()
        
        keywords = self._parse_keywords(query)
        print(f"🔍 Search keywords: {keywords}")
        
        api_query = self._build_api_query(keywords)
        print(f"📡 API query: {api_query}")
        
        language = language or self.default_language
        
        if not from_date:
            from_date = datetime.now() - timedelta(days=2)
        if not to_date:
            to_date = datetime.now() - timedelta(days=1)
            
        from_param = from_date.strftime('%Y-%m-%d')
        to_param = to_date.strftime('%Y-%m-%d')
        
        print(f"📅 Searching from {from_param} to {to_param}")
        print("⏰ Note: Articles have 24-hour delay on Free Plan")
        
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
    
    