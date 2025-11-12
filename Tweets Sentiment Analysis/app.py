import tweepy
import pandas as pd
from datetime import datetime
import os
import json
import time
from typing import List, Dict
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import asyncio
import threading

class TwitterClient:
    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("⚠️ python-dotenv not installed. Install it with: pip install python-dotenv")
        
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        if not self.bearer_token:
            print("Twitter API credentials not found. Using simulated data.")
            self.client = None
        else:
            try:
                self.client = tweepy.Client(
                    bearer_token = self.bearer_token,
                    consumer_key = self.api_key,
                    consumer_secret = self.api_secret,
                    access_token = self.access_token,
                    access_token_secret = self.access_token_secret,
                    wait_on_rate_limit=True
                )
                print("✅ Twitter API client initialized successfully")
            except Exception as e:
                print(f"❌ Error initialized Twitter client: {e}")
                self.client = None
                
    def search_tweets(self, query: str, max_results: int = 10) -> List[Dict]:
        if not self.client:
            return self._get_simulated_tweets(max_results)
        
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations']  
            )
            
            print(f"📊 Twitter API Response: {tweets}")
            print(f"📊 Tweets data: {tweets.data if tweets else 'None'}")
            
            if not tweets.data:
                print("⚠️ No tweets found in API response. Using simulated data.")
                return self._get_simulated_tweets(max_results)
            
            tweet_list = []
            for tweet in tweets.data:
                tweet_data = {
                    'text': tweet.text,
                    'timestamp': tweet.created_at,
                    'user': f"user_{tweet.author_id}",
                    'likes': tweet.public_metrics['like_count'] if tweet.public_metrics else 0,
                    'retweets': tweet.public_metrics['retweet_count'] if tweet.public_metrics else 0,
                    'tweet_id': tweet.id
                }
                tweet_list.append(tweet_data)
                
            print(f"✅ Successfully processed {len(tweet_list)} tweets from API")
            return tweet_list
        
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return self._get_simulated_tweets(max_results)
        
    def _get_simulated_tweets(self, count: int) -> List[Dict]:
        from datetime import datetime, timedelta
        import random
        
        sample_tweets = [
            "Just discovered this amazing new coffee shop! ☕ #coffee #love",
            "Traffic is terrible today! 😤 #frustrated #commute",
            "Beautiful sunset tonight 🌅 #nature #peaceful",
            "My flight got delayed again... 😞 #travel #delays",
            "Excited for the weekend! 🎉 #happy #weekend",
            "This movie is absolutely incredible! Must watch 🍿 #movies",
            "Worst customer service ever! Very disappointed 😠 #complaint",
            "Learning Python is so rewarding! 💻 #coding #tech",
            "Rain ruined my picnic plans ☔ #weather #sad",
            "Just finished a great workout! 💪 #fitness #health"
        ]
        
        tweets = []
        for _ in range(count):
            tweet = {
                'text': random.choice(sample_tweets),
                'timestamp': datetime.now() - timedelta(seconds=random.randint(0, 3600)),
                'user': f"user_{random.randint(1000, 9999)}",
                'likes': random.randint(0, 100),
                'retweets': random.randint(0, 50),
                'tweet_id': f"sim_{random.randint(100000, 999999)}"
            }
            tweets.append(tweet)
        
        return tweets

class EnhancedSentimentAnalyzer:
    def __init__(self) -> None:
        self._setup_nltk()
        
        from textblob import TextBlob
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import re
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._download_nltk_data()
        
    def _setup_nltk(self):
        try:
            import nltk
            import ssl
            
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            import os
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.append(nltk_data_dir)
        
        except ImportError as e:
            print(f"⚠️ Could not import NLTK: {e}")
            raise
        
    def _download_nltk_data(self):
        import nltk
        try:
            for corpus in ['punkt', 'brown', 'vader_lexicon', 'punkt_tab']:
                try:
                    nltk.download(corpus, quiet=True)
                except Exception as e:
                    print(f"⚠️ Could not download NLTK corpus '{corpus}': {e}")
            print("✅ NLTK data downloaded successfully")
        except Exception as e:
            print(f"⚠️ Could not download NLTK data: {e}")
            print("Sentiment analysis may not work properly")
        
        
    def clean_text(self, text):
        import re
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = ' '.join(text.split())
        
        return text.strip()
        
    def analyze_sentiment(self, text):
        from textblob import TextBlob
        
        clean_text = self.clean_text(text)
        
        blob = TextBlob(clean_text)
        textblob_polarity = blob.sentiment.polarity
        
        vader_scores = self.vader_analyzer.polarity_scores(clean_text)
        vader_compound = vader_scores['compound']
        
        if textblob_polarity > 0.1 and vader_compound > 0.05:
            sentiment = 'Positive'
            confidence = (abs(textblob_polarity) + abs(vader_compound)) / 2
        elif textblob_polarity < -0.1 and vader_compound < -0.05:
            sentiment = 'Negative'
            confidence = (abs(textblob_polarity) + abs(vader_compound)) / 2
        else:
            sentiment = 'Neutral'
            confidence = 1 - abs(textblob_polarity - vader_compound)
            
        return {
            'sentiment': sentiment,
            'confidence': min(confidence, 1.0),
            'textblob_polarity': textblob_polarity,
            'vader_compound': vader_compound
        }
        
def main():
    st.set_page_config(
        page_title="Real-Time Twitter Sentiment Analysis",
        page_icon="🐦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
    }
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral-sentiment {
        color: #6c757d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🥹 Real-Time Twitter Sentiment Analysis Dashboard")
    st.markdown("---")
    
    if 'twitter_client' not in st.session_state:
        st.session_state.twitter_client = TwitterClient()
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = EnhancedSentimentAnalyzer()
    if 'tweets_data' not in st.session_state:
        st.session_state.tweets_data = []
    if 'search_query' not in st.session_state:
        st.session_state.search_query = "python OR javascript OR coding"
        
    # if st.session_state.twitter_client.client is not None:
    #     # st.warning("⚠️ **Rate Limits Active**\n\nTwitter allows 300 requests per 15 minutes. If you see 'sleeping' messages, the app is waiting for the limit to reset.")
    #     # st.info("💡 **Tip**: Disable auto-refresh to avoid hitting rate limits quickly")
    # else:
    #     st.info("ℹ️ **Info**: Using simulated tweet data. Connect Twitter API for live data.")
        
    st.sidebar.header("🔧 Controls")
    
    # Search configuration
    st.sidebar.subheader("Search Configuration")
    search_query = st.sidebar.text_input(
        "Search Query:", 
        value=st.session_state.search_query,
        help="Enter keywords to search for tweets. Use OR, AND for complex queries."
    )
    
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        
    tweet_count = st.sidebar.slider("Tweets per fetch:", 10, 50, 20)
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds):", 300, 1800, 900)
        
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()
            
        current_time = time.time()
        if current_time - st.session_state.last_refresh_time >= refresh_interval:
            with st.spinner("🔄 Auto-refreshing..."):
                fetch_tweets(search_query, tweet_count)
                st.session_state.last_refresh_time = current_time
                st.rerun()
                
        time_since_refresh = current_time - st.session_state.last_refresh_time
        time_until_refresh = refresh_interval - time_since_refresh
        if time_until_refresh > 0:
            st.sidebar.info(f"⏰ Next refresh in: {int(time_until_refresh)}s")
    
    if st.sidebar.button("🔍 Fetch New Tweets", type="primary"):
        fetch_tweets(search_query, tweet_count)
        
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state.tweets_data = []
        st.rerun()
        
    st.sidebar.subheader("🗃️ Analyze Your Tweet")
    user_tweet = st.sidebar.text_area("Enter your text:")
    if st.sidebar.button("Analyze Text") and user_tweet:
        analyze_user_tweet(user_tweet)
        
    if not st.session_state.tweets_data:
        st.info("👆 **Welcome!** Click '🔍 Fetch New Tweets' button to start analyzing Twitter sentiment!")
        st.markdown("""
        ### 🚀 How to use:
        1. **Enter search terms** in the sidebar (e.g., "python", "ai OR machine learning")
        2. **Adjust tweet count** (10-50 tweets per search)
        3. **Click 'Fetch New Tweets'** to start analysis
        4. **Try the text analyzer** to test custom text
        
        ### 📊 You'll see:
        - Real-time sentiment metrics
        - Interactive charts and graphs  
        - Word cloud visualization
        - Recent tweets table
        """)
    else:
        display_dashboard()
            
def fetch_tweets(query, count):
    with st.spinner("🔍 Fetching tweets..."):
        try:
            tweets = st.session_state.twitter_client.search_tweets(query, count)
             
            if not tweets:
                st.warning("No tweets found for the given query.")
                return 
            
            if len(st.session_state.tweets_data) > 1000:
                st.session_state.tweets_data = st.session_state.tweets_data[-500:]
             
            for tweet in tweets:
                analysis = st.session_state.sentiment_analyzer.analyze_sentiment(tweet['text'])
                 
                tweet_data = {
                     'text': tweet['text'],
                     'timestamp': tweet['timestamp'],
                     'user': tweet['user'],
                     'likes': tweet['likes'],
                     'retweets': tweet['retweets'],
                     'sentiment': analysis['sentiment'],
                     'confidence': analysis['confidence'],
                     'textblob_polarity': analysis['textblob_polarity'],
                     'vader_compound': analysis['vader_compound'],
                     'tweet_id': tweet.get('tweet_id', 'unknown')
                }
                 
                st.session_state.tweets_data.append(tweet_data)
                 
            if len(st.session_state.tweets_data) > 500:
                st.session_state.tweets_data = st.session_state.tweets_data[-500:]
            
            st.success(f"✅ Fetched and analyzed {len(tweets)} tweets!")
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error fetching tweets: {str(e)}")
            
def analyze_user_tweet(text):
    analysis = st.session_state.sentiment_analyzer.analyze_sentiment(text)
    
    sentiment_color ={
        'Positive': '🟢',
        'Negative': '🔴', 
        'Neutral': '🟡'
    }    
    
    st.sidebar.success(f"{sentiment_color[analysis['sentiment']]} Sentiment: **{analysis['sentiment']}**")
    st.sidebar.info(f" Confidence: **{analysis['confidence']:.2f}**")
    
    tweet_data = {
        'text': text,
        'timestamp': datetime.now(),
        'user': 'You',
        'likes': 0,
        'retweets': 0,
        'sentiment': analysis['sentiment'],
        'confidence': analysis['confidence'],
        'textblob_polarity': analysis['textblob_polarity'],
        'vader_compound': analysis['vader_compound'],
        'tweet_id': 'user_input'
    }
    
    st.session_state.tweets_data.append(tweet_data)
    
def display_dashboard():
    if not st.session_state.tweets_data:
        st.warning("No tweet data available. Please fetch some tweets first.")
        return
    
    df = pd.DataFrame(st.session_state.tweets_data)
    
    if df.empty:
        st.warning("Tweet data is empty. Please try fetching tweets again.")
        return
    
    display_metrics(df)
    display_charts(df)
    display_recent_tweets(df)
    
def display_metrics(df):
    if df.empty:
        st.info("No data available for metrics display.")
        return 
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_tweets = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    
    with col1:
        st.metric("📊 Total Tweets", total_tweets)
    
    with col2:
        positive_pct = (sentiment_counts.get('Positive', 0) / total_tweets) * 100
        st.metric("😊 Positive", f"{positive_pct:.1f}%", 
                 delta=f"{sentiment_counts.get('Positive', 0)} tweets")
    
    with col3:
        negative_pct = (sentiment_counts.get('Negative', 0) / total_tweets) * 100
        st.metric("😞 Negative", f"{negative_pct:.1f}%",
                 delta=f"{sentiment_counts.get('Negative', 0)} tweets")
    
    with col4:
        avg_confidence = df['confidence'].mean()
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
        
def display_charts(df):
    colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        
        colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
        fig = px.pie(
            values = sentiment_counts.values,
            names = sentiment_counts.index,
            color = sentiment_counts.index,
            color_discrete_map = colors,
            title = "Sentiment Breakdown"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("⌛ Sentiment Timeline")
        df_sorted = df.sort_values('timestamp').tail(50)
        
        fig = px.scatter(
            df_sorted,
            x='timestamp',
            y='confidence',
            color='sentiment',
            color_discrete_map=colors,
            title='Sentiment Confidence Over Time',
            hover_data=['text']
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Confidence Score")
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("💭 Word Cloud")
    create_wordcloud(df)
    
    st.subheader("📊 Confidence Score Distribution")
    fig = px.violin(
        df,
        x = 'sentiment',
        y = 'confidence',
        color = 'sentiment',
        color_discrete_map = colors,
        title = 'Confidence Score Distribution by Sentiment'
    )
    st.plotly_chart(fig, use_container_width=True)
    
def create_wordcloud(df):
    try:  
        if df.empty or 'text' not in df.columns:
            st.info("No text data available for word cloud")
            return
        
        all_text = ' '.join(df['text'].astype(str).tolist())
        
        import re
        all_text = re.sub(r'http\S+|www\S+|https\S+', '', all_text)
        all_text = re.sub(r'@\w+|#\w+', '', all_text)
        all_text = ' '.join(all_text.split())
        
        if len(all_text.strip()) < 10:
            st.info("Not enough text data for meaningful word cloud")
            return
        
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            collocations=False
        ).generate(all_text)
            
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        st.info("Skipping word cloud due to error")

def display_recent_tweets(df):
    """Display recent tweets in a table"""
    st.subheader("🐦 Recent Tweets")
    
    df_display = df.sort_values('timestamp', ascending=False).head(20).copy()
    
    df_display['Time'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%H:%M:%S')
    df_display['Tweet'] = df_display['text'].str[:100] + '...'
    df_display['Engagement'] = df_display['likes'] + df_display['retweets']
    
    display_cols = ['Time', 'Tweet', 'sentiment', 'confidence', 'Engagement']
    df_show = df_display[display_cols].copy()
    df_show['confidence'] = df_show['confidence'].round(3)
    
    def style_sentiment(val):
        if val == 'Positive':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Negative':
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #e2e3e5; color: #383d41'
        
    styled_df = df_show.style.map(style_sentiment, subset=['sentiment'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
if __name__ == "__main__":
    main()