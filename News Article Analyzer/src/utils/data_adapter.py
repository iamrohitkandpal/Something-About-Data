"""
# DataAdapter: This converts news articles into processing compatible format
"""

from typing import Dict, List
from datetime import datetime
import pandas as pd
import numpy as np

class DataAdapter:
    """
    This adapts news articles data to match proper data structure
    
    Why needed?
    - For proper charts/visualization of the data
    - Keeps the data format consistent in project
    - Data becomes easy for comparison
    
    Data Format:
    {
        'text': str,            # Main content
        'sentiment': str,       # positive/negative/neutral
        'score': float,         # -1 to +1
        'timestamp': datetime,  # When Posted
        'source': str,          # Where it came from
    }
    """
    
    @staticmethod
    def article_to_good_format(article: Dict) -> Dict:
        """
        Why this mapping?
        - article['title'] → tweet['text'] (main display text)
        - article['content'] → tweet['full_text'] (detailed content)
        - article['source'] → tweet['source'] (publication name)
        - article['published_at'] → tweet['timestamp'] (date)
        
        Args:
            article: News article dictionary
        
        Returns:
            Tweet-formatted dictionary
        """
        
        try:
            timestamp = datetime.strptime(
                article.get('published_at', ''),
                '%Y-%m-%dT%H:%M:%SZ'
            )
        except:
            timestamp = datetime.now()
            
        return {
            'text': article.get('title', 'No Title'),
            'full_text': article.get('description', ''),
            'content': article.get('content', ''),
            'sentiment': article.get('sentiment', 'neutral'),
            'score': article.get('sentiment_score', 0.0),
            'polarity': article.get('polarity', 0.0),
            'subjectivity': article.get('subjectivity', 0.0),
            'confidence': article.get('confidence', 0.0),
            'timestamp': timestamp,
            'source': article.get('source', 'Unknown'),
            'source_id': article.get('source_id', ''),
            'region': article.get('region', 'unknown'),
            'url': article.get('url', ''),
            'author': article.get('author', 'Unknown'),
            'image_url': article.get('image_url', ''),
        }
        
    @staticmethod
    def articles_to_dataframe(articles: List[Dict]) -> pd.DataFrame:
        """
        Articles to Pandas Dataframe Conversion
        
        Why Dataframe?
        - Easy filtering and sorting
        - Compatible with Plotly charts
        - Efficient for large datasets
        
        Args:
            articles: List of article dictionaries
        
        Returns:
            pandas Dataframe with standardized columns
        """
        
        if not articles:
            return pd.DataFrame()
        
        data = []
        
        for article in articles:
            source = article.get('source', {})
            
            if isinstance(source, dict):
                source_name = source.get('name', 'Unknown')
            else:
                source_name = str(source)
                
            sentiment_data = article.get('sentiment', {})
            
            if not data:
                print(f"\n🔍 DEBUG in articles_to_dataframe:")
                print(f"   article['sentiment'] = {sentiment_data}")
                print(f"   type = {type(sentiment_data)}")
            
            if isinstance(sentiment_data, dict):
                sentiment_label = sentiment_data.get('sentiment', 'neutral')
                score = float(sentiment_data.get('combined_score', 0.0))
                polarity = float(sentiment_data.get('polarity', 0.0))
                subjectivity = float(sentiment_data.get('subjectivity', 0.0))
                confidence = float(sentiment_data.get('confidence', 0.0))
                vader = float(sentiment_data.get('vader_compound', 0.0))
            elif isinstance(sentiment_data, str):
                sentiment_label = sentiment_data
                score = 0.0
                polarity = 0.0
                subjectivity = 0.0
                confidence = 0.0
                vader = 0.0
            else:
                sentiment_label = 'neutral'
                score = 0.0
                polarity = 0.0
                subjectivity = 0.0
                confidence = 0.0
                vader = 0.0
            
            try:
                published_at = article.get('publishedAt', article.get('published_at', ''))
                if published_at:
                    timestamp = pd.to_datetime(published_at)
                else:
                    timestamp = pd.NaT
            except:
                timestamp = pd.NaT
            
            data.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'text': article.get('processed_text', article.get('title', '')),
                'source': source_name,
                'url': article.get('url', ''),
                'published': article.get('publishedAt', article.get('published_at', '')),
                'timestamp': timestamp,
                'region': article.get('region', 'international'),
                
                'sentiment': sentiment_label,
                'score': score,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'confidence': confidence,
                'vader_compound': vader,
            })
            
        df = pd.DataFrame(data)
            
        print(f"\n📊 DEBUG after DataFrame creation:")
        print(f"   Sentiment column type: {df['sentiment'].dtype}")
        print(f"   Unique sentiments: {df['sentiment'].unique()}")
        print(f"   Value counts:\n{df['sentiment'].value_counts()}")
            
        return df
    
    @staticmethod
    def group_by_source(df: pd.DataFrame) -> pd.DataFrame:
        """
        This will group articles bt new source
        
        Why useful?
        - Compare sentiment across publications
        - Indetify biased sources
        - Show "BBC is 60% positive, CNN is 40% positive"
        
        Returns:
            DataFrame with source-level statistics
        """
        
        if df.empty:
            return pd.DataFrame()
        
        source_stats = df.groupby('source').agg({
            'sentiment': lambda x: x.value_counts().to_dict(),
            'score': ['mean', 'std', 'count'],
            'polarity': 'mean',
            'subjectivity': 'mean'
        }).reset_index()
        
        source_stats.columns = [
            'source', 'sentiment_dist', 'avg_score', 
            'score_std', 'article_count', 'avg_polarity', 'avg_subjectivity'
        ]
        
        return source_stats
    
    @staticmethod
    def group_by_region(df: pd.DataFrame) -> Dict:
        """
        Tis groups articles by region (Indian vs International)
        
        Why Important?
        - Compares regional sentiment differences
        - Shows cultural perspectives
        - Also detects coverage bias in articles
        
        Returns:
            Dictionary with regional statistics
            {
                'indian': {...stats...},
                'international': {...stats...},
                'comparison': {...differences...}
            }
        """
        
        if df.empty or 'region' not in df.columns:
            return {}
        
        indian_df = df[df['region'] == 'indian']
        intl_df = df[df['region'] == 'international']
        
        def calc_stats(region_df):
            if region_df.empty:
                return None
            
            num_cols = ['score', 'polarity', 'subjectivity']
            for col in num_cols:
                if col in region_df.columns:
                    region_df[col] = pd.to_numeric(region_df[col], errors='coerce').fillna(0.0)
            
            sentiments = region_df['sentiment'].value_counts()
            total = len(region_df)
            
            top_sources_dict = region_df['source'].value_counts().head(3).to_dict()
            
            return {
                'total_articles': total,
                'positive_pct': round((sentiments.get('positive', 0) / total * 100), 1),
                'neutral_pct': round((sentiments.get('neutral', 0) / total * 100), 1),
                'negative_pct': round((sentiments.get('negative', 0) / total * 100), 1),
                'avg_score': round(region_df['score'].mean(), 3),
                'avg_polarity': round(region_df['polarity'].mean(), 3),
                'avg_subjectivity': round(region_df['subjectivity'].mean(), 3),
                'top_sources': top_sources_dict
            }
            
        indian_stats = calc_stats(indian_df.copy())
        intl_stats = calc_stats(intl_df.copy())
        
        comparison = {}
        if indian_stats and intl_stats:
            comparison = {
                'sentiment_diff': indian_stats['avg_score'] - intl_stats['avg_score'],
                'polarity_diff': indian_stats['avg_polarity'] - intl_stats['avg_polarity'],
                'subjectivity_diff': indian_stats['avg_subjectivity'] - intl_stats['avg_subjectivity'],
                'more_positive': 'indian' if indian_stats['positive_pct'] > intl_stats['positive_pct'] else 'international'
            }
        
        return {
            'indian': indian_stats,
            'international': intl_stats,
            'comparison': comparison
        }
        
    @staticmethod
    def filter_by_sentiment(df: pd.DataFrame, sentiment: str) -> pd.DataFrame:
        """
        This filters the articles by sentiment type
        
        Why?
        - UI Feature: "Show only positive news"
        - Analysis: Focus on negative coverage
        - User preference: Avoid negative news
        """
        
        if df.empty:
            return df
        
        return df[df['sentiment'] == sentiment]
    
    @staticmethod
    def filter_by_date_range(
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        This filters articles bt date range
        
        Why?
        - For Historical analysis of data
        - Compare "sentiment this week vs last week"
        - To track sentiment trends over time
        """
        
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        return df[
            (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        ]
        
    @staticmethod
    def get_time_series_data(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        To aggregate sentiment by time period
        
        Why?
        - Show sentiment trends over time
        - Plot "sentiment score per day"
        - Indentify temporal patterns
        
        Args:
            freq: Pandas frequency ('D'=daily, 'W'=weekly, 'H'=hourly)
            
        Returns:
            DataFrame with time-indexed sentiment scores
        """
        
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy.set_index('timestamp', inplace=True)
        
        time_series = df_copy.resample(freq).agg({
            'score': 'mean',
            'sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
        }).reset_index()
        
        return time_series
    
if __name__ == "__main__":
    sample_articles = [
        {
            'title': 'Tech Breakthrough Announced',
            'description': 'New innovation changes industry',
            'content': 'Details of the breakthrough...',
            'source': 'TechCrunch',
            'source_id': 'techcrunch',
            'published_at': '2024-01-15T10:30:00Z',
            'region': 'international',
            'sentiment': 'positive',
            'sentiment_score': 0.75,
            'polarity': 0.8,
            'subjectivity': 0.6,
            'confidence': 0.75
        },
        {
            'title': 'Economic Crisis Deepens',
            'description': 'Markets react negatively',
            'content': 'Economic analysis...',
            'source': 'The Times of India',
            'source_id': 'the-times-of-india',
            'published_at': '2024-01-15T11:00:00Z',
            'region': 'indian',
            'sentiment': 'negative',
            'sentiment_score': -0.65,
            'polarity': -0.7,
            'subjectivity': 0.5,
            'confidence': 0.65
        }
    ]
    
    df = DataAdapter.articles_to_dataframe(sample_articles)
    print("📊 Converted to DataFrame:")
    print(df[['text', 'source', 'region', 'sentiment', 'score']].to_string())
    
    print("\n🌍 Regional Comparison:")
    regional_stats = DataAdapter.group_by_region(df)
    print(f"Indian: {regional_stats.get('indian')}")
    print(f"International: {regional_stats.get('international')}")
    print(f"Comparison: {regional_stats.get('comparison')}")
    
    print("\n📰 Source Statistics:")
    source_stats = DataAdapter.group_by_source(df)
    print(source_stats.to_string())