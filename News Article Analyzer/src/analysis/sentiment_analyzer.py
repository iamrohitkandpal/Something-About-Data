"""
# The Heart of this Project: Analyzes sentiment of news articles 
"""

import nltk
import numpy as np
import pandas as pd
import streamlit as st

from typing import Dict, List
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.resources
def _nltk_downloading():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

_nltk_downloading()

class SentimentAnalyzer:
    """
    Analyzes sentiment using TextBlob + VADER
    
    Why two libraries?
    - TextBlob: Good for general text, return polarity (-1 to +1)
    - VADER: Optimized for social media, handles slang/emojis
    - Combined = More accurate results (ensemble approach)
    """
    
    def __init__(self):
        """
        Initialize Sentiment Analyzers
        
        Initialization For?
        - Loading models once (Performacne Optimization)
        - Reuse Across Multiple Articles
        """
        
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text:str) -> Dict:
        """
        This analyzes sentiment of text
        
        Args: 
            text: Article text (title + description + content)
            
        Returns:
            Dictionary with sentiment scores
            {
                'polarity': -1 to +1 score,
                'subjectivity': 0 to 1 (objective vs subjective)
                'vader_compound': -1 to +1 VADER score,
                'sentiment': 'positive'/'neutral'/'negative',
                'confidence': 0 to 1
            }
            
        Why return multiple scores?
        - polarity: Overall positivity/negativity
        - subjectivity: FAct vs Opinion (imp for news right)
        - vader_compound: Alternative sentiment measure
        - sentiment: Simple label for visualization
        - confidence: How sure we are (for filtering low-quality predictions)
        """
        
        if not text or len(text.strip()) == 0:
            return self._empty_sentiment()
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        combined_score = (0.6 * polarity) + (0.4 * vader_compound)
        
        if combined_score >= 0.05:
            sentiment = 'positive'
        elif combined_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        confidence = abs(combined_score)
        
        return {
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'vader_compound': round(vader_compound, 3),
            'combined_score': round(combined_score, 3),
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'positive_score': vader_scores['pos'],
            'neutral_score': vader_scores['neu'],
            'negative_score': vader_scores['neg']
        }
        
    def analyze_article(self, article: Dict) -> Dict:
        """
        For complete article analyses
        
        Why separate method?
        - articles have multiple text fields (title, descriptions, content)
        - Need to combine them in a smart manner
        
        Args:
            article: Dictionary with title, description, content
            
        Returns:
            Article dictionary with added sentiment fields
        """
        
        # Combine text fields (title weighted more - it's the summary!)
        # Why this? Title is most important (like headline in newspaper)
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')[:1000]
        
        # Weighted text: title 3x, description 2x, content 1x
        # Why? Title is most condensed sentiment indicator
        full_text = f"{title} {title} {title} {description} {description} {content}"
        
        sentiment_data = self.analyze_text(full_text)
        
        article_with_sentiment = article.copy()
        
        article_with_sentiment['sentiment'] = sentiment_data
        article_with_sentiment['sentiment_label'] = sentiment_data['sentiment']
        article_with_sentiment['sentiment_score'] = sentiment_data['combined_score']
        article_with_sentiment['polarity'] = sentiment_data['polarity']
        article_with_sentiment['subjectivity'] = sentiment_data['subjectivity']
        article_with_sentiment['confidence'] = sentiment_data['confidence']
        
        return article_with_sentiment

    def analyze_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        For multiple articles sentiment analyses
        
        Why batch processing?
        - More efficient then one-by-one
        - Progress tracking for large datasets
        - Parallel processing possible (future enhancement)
        
        Args:
            articles: List of article dictionaries
        
        Returns :
            List of articles with sentiment data added
        """
        
        analyzed_articles = []
        
        for i, article in enumerate(articles):
            analyzed = self.analyze_article(article)
            analyzed_articles.append(analyzed)
            
            if (i + 1) % 10 == 0:
                print(f"📊 Analyzed {i + 1}/{len(articles)} articles...")
        
        return analyzed_articles
    
    def get_sentiment_summary(self, analyzed_articles: List[Dict]) -> Dict:
        """
        For Overall Statistics
        
        Why needed?
        - Dashboard needs aggregate stats
        - Compare sentiement across sources
        - Similar to rating distributions
        
        Args:
            articles: List of analyzed articles
        
        Returns:
            Summary statistics dictionary
        """
        
        if not analyzed_articles:
            return self._empty_summary()
        
        sentiments = []
        scores = []
        
        for article in analyzed_articles:
            sentiment_data = article.get('sentiment', {})
            
            if isinstance(sentiment_data, dict):
                sentiments.append(sentiment_data.get('sentiment', 'neutral'))
                scores.append(sentiment_data.get('combined_score', 0.0))
            elif isinstance(sentiment_data, str):
                sentiments.append(sentiment_data)
                scores.append(0.0)
            else:
                sentiments.append('neutral')
                scores.append(0.0)
                
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        
        total = len(analyzed_articles)
        
        return {
            'total_articles': total,
            'positive_count': sentiment_counts.get('positive', 0),
            'neutral_count': sentiment_counts.get('neutral', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'positive_percentage': round((sentiment_counts.get('positive', 0) / total * 100), 1),
            'neutral_percentage': round((sentiment_counts.get('neutral', 0) / total * 100), 1),
            'negative_percentage': round((sentiment_counts.get('negative', 0) / total * 100), 1),
            'average_score': round(sum(scores) / len(scores), 3) if scores else 0.0,
            'median_score': round(sorted(scores)[len(scores)//2], 3) if scores else 0.0,
            'std_dev': round(pd.Series(scores).std(), 3) if scores else 0.0,
        }
        
    def _empty_summary(self) -> Dict:
        """
        For returning empty summary for no articles
        """
        
        return {
            'total_articles': 0,
            'positive_count': 0,
            'neutral_count': 0,
            'negative_count': 0,
            'positive_percentage': 0.0,
            'neutral_percentage': 0.0,
            'negative_percentage': 0.0,
            'average_score': 0.0,
            'median_score': 0.0,
            'std_dev': 0.0
        }
        
    def _empty_sentiment(self) -> Dict:
        """
        For returning empty sentiment for invalid text
        """
        
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'vader_compound': 0.0,
            'combined_score': 0.0,
            'sentiment': 'neutral',
            'confidence': 0.0,
            'positive_score': 0.0,
            'neutral_score': 1.0,
            'negative_score': 0.0,
        }
        
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_articles = [
        {
            'title': 'Amazing breakthrough in AI technology!',
            'description': 'Scientists achieve remarkable results',
            'content': 'This is a very positive development for the field.'
        },
        {
            'title': 'Terrible disaster strikes region',
            'description': 'Devastating news from affected areas',
            'content': 'The situation is worse than expected.'
        },
        {
            'title': 'Weather forecast for tomorrow',
            'description': 'Expect partly cloudy skies',
            'content': 'Temperature will be moderate.'
        }
    ]
    
    analyzed = analyzer.analyze_batch(test_articles)
    summary = analyzer.get_sentiment_summary(analyzed)
    
    print("\n📊 Sentiment Analysis Results:")
    print(f"Positive: {summary['positive_count']} ({summary['positive_percentage']}%)")
    print(f"Neutral: {summary['neutral_count']} ({summary['neutral_percentage']}%)")
    print(f"Negative: {summary['negative_count']} ({summary['negative_percentage']}%)")