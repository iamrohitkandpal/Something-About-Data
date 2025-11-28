import streamlit as st
import praw
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv
import time
import random

load_dotenv()

class RedditClient:
    def __init__(self):
        """Initialize Reddit API client"""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT'),
            )
            
            self.reddit.user.me()
            self.connected = True
            print("✅ Reddit API client initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Reddit API Initialized failed: {str(e)}")
            self.reddit = None
            self.connected = False

    def search_subreddit_posts(self, subreddit_name, query="", limit=50, time_filter="week", sort_by="hot"):
        """Search posts in a subreddit"""
        if not self.connected or self.reddit is None:
            return self._generate_fake_reddit_data(limit, subreddit_name)
        
        try:
            posts_data = []
            subreddit = self.reddit.subreddit(subreddit_name)
            
            if query:
                posts = subreddit.search(query, limit=limit, time_filter=time_filter, sort=sort_by)
            else:
                if sort_by == "hot":
                    posts = subreddit.hot(limit=limit)
                elif sort_by == "new":
                    posts = subreddit.new(limit=limit)
                elif sort_by == "top":
                    posts = subreddit.top(time_filter=time_filter, limit=limit)
                else:
                    posts = subreddit.hot(limit=limit)
            
            for post in posts:
                post.comments.replace_more(limit=0)
                top_comments = post.comments.list()[:5]
                
                post_data = {
                    'post_id': post.id,
                    'title': post.title,
                    'text': post.selftext if post.selftext else post.title,
                    'author': str(post.author) if post.author else '[deleted]',
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': f"https://reddit.com{post.permalink}",
                    'subreddit': subreddit_name,
                    'comments': [
                        {
                            'text': comment.body,
                            'score': comment.score,
                            'author': str(comment.author) if comment.author else '[deleted]',
                        }
                        for comment in top_comments
                    ]
                }
                posts_data.append(post_data)
            
            print(f"✅ Fetched {len(posts_data)} posts from r/{subreddit_name}")
            return posts_data
        
        except Exception as e:
            print(f"❌ Error fetching Reddit data: {str(e)}")
            st.error(f"Error: {str(e)}")
            return []
        
    def _generate_fake_reddit_data(self, count, subreddit_name):
        """Generate fake Reddit data for testing"""
        fake_posts = []
        sample_texts = [
            "This new Python library is amazing! Highly recommend it for data analysis.",
            "Having major issues with this framework. Documentation is terrible and confusing.",
            "Just launched my first data pipeline project! Feeling accomplished.",
            "Why is this technology so complicated? Really need help understanding it.",
            "Best tutorial I've found for machine learning. Clear and comprehensive.",
            "This update broke everything. Very disappointed with the changes.",
            "Absolutely love this community! So helpful and supportive.",
            "Can't figure this out. Spent hours debugging with no progress.",
            "Game changer for my workflow. Productivity increased significantly!",
            "Not impressed. Expected much better quality and performance."
        ]  
        
        for i in range(count):
            fake_posts.append({
                'post_id': f'fake_{i}',
                'title': f'Discussion about {subreddit_name} topic {i+1}',
                'text': random.choice(sample_texts),
                'author': f'user_{random.randint(1, 100)}',
                'score': random.randint(10, 1000),
                'upvote_ratio': random.uniform(0.7, 0.99),
                'num_comments': random.randint(5, 50),
                'created_utc': datetime.now() - timedelta(hours=random.randint(1, 168)),
                'url': f'https://reddit.com/r/{subreddit_name}/fake_{i}',
                'subreddit': subreddit_name,
                'comments': []  
            })
            
        return fake_posts

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using both TextBlob and VADER"""
        
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        avg_score = (textblob_polarity + vader_compound) / 2
        
        if avg_score > 0.05:
            sentiment = "Positive"
        elif avg_score < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        confidence = abs(avg_score)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'textblob_polarity': textblob_polarity,
            'vader_compound': vader_compound,
            'vader_scores': vader_scores
        }
        
# UTILITY FUNCTIONS
def fetch_reddit_posts(subreddit, query, count, time_filter, sort_by):
    """Fetch and analyze Reddit posts"""
    with st.spinner(f"🔍 Fetching posts from r/{subreddit}..."):
        try:
            posts = st.session_state.reddit_client.search_subreddit_posts(
                subreddit_name=subreddit,
                query=query,
                limit=count,
                time_filter=time_filter,
                sort_by=sort_by
            )
            
            if not posts:
                st.warning("No posts found for the given query.")
                return
            
            for post in posts:
                full_text = f"{post['title']} {post['text']}"
                analysis = st.session_state.sentiment_analyzer.analyze_sentiment(full_text)
                
                post_data = {
                    'text': full_text[:500],
                    'title': post['title'],
                    'timestamp': post['created_utc'],
                    'user': post['author'],
                    'score': post['score'],
                    'comments': post['num_comments'],
                    'sentiment': analysis['sentiment'],
                    'confidence': analysis['confidence'],
                    'textblob_polarity': analysis['textblob_polarity'],
                    'vader_compound': analysis['vader_compound'],
                    'post_id': post['post_id'],
                    'platform': 'Reddit',
                    'subreddit': post['subreddit'],
                    'upvote_ratio': post['upvote_ratio'],
                    'url': post['url'],
                }
                
                st.session_state.posts_data.append(post_data)
            
            if len(st.session_state.posts_data) > 500:
                st.session_state.posts_data = st.session_state.posts_data[-500]
                
            st.success(f"✅ Fetched and analyzed {len(posts)} Reddit posts!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error fetching Reddit posts: {str(e)}")

def display_metrics(df):
    """Display key metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Posts", len(df))
    
    with col2:
        positive_pct = (len(df[df['sentiment'] == "Positive"]) / len(df) * 100)
        st.metric("😀 Positve", f"{positive_pct:.1f}%")
    
    with col3:
        negative_pct = (len(df[df['sentiment'] == "Negative"]) / len(df) * 100)
        st.metric("😔 Negative", f"{negative_pct:.1f}%")
    
    with col4:
        neutral_pct = (len(df[df['sentiment'] == "Neutral"]) / len(df) * 100)
        st.metric("🙂 Neutral", f"{neutral_pct:.1f}%")
        
    
def display_charts(df):
    """Display visualization charts"""
    col1, col2 = st.columns(2)
    
    # Sentiment Distribution
    with col1:
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="📊 Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Over Time
    with col2:
        df_sorted = df.sort_values('timestamp')
        fig = px.scatter(
            df_sorted,
            x='timestamp',
            y='vader_compound',
            color='sentiment',
            title="📈 Sentiment Timeline",
            color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Top Posts by Engagement
    st.subheader("f🔥 Top Posts by Engagement")
    top_posts = df.nlargest(5, 'score')[['title', 'score', 'comments', 'sentiment', 'upvote_ratio']]
    st.dataframe(top_posts, use_container_width=True)
    
def display_recent_posts(df):
    """Display recent posts table"""
    st.subheader("📋 Recent Posts")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment:",
            options=['Positive', 'Negative', 'Neutral'],
            default=['Positive', 'Negative', 'Neutral']
        )
    
    with col2:
        sort_option = st.selectbox(
            "Sort by:",
            options=['timestamp', 'score', 'confidence'],
            index=0
        )
    
    # Apply filters
    filtered_df = df[df['sentiment'].isin(sentiment_filter)]
    filtered_df = filtered_df.sort_values(sort_option, ascending=False)
    
    # Display table
    display_columns = ['title', 'sentiment', 'score', 'comments', 'timestamp', 'user']
    st.dataframe(filtered_df[display_columns].head(20), use_container_width=True)


# MAIN APPLICATION
def main():
    st.set_page_config(
        page_title="Reddit Sentiment Analysis",
        page_icon="🔴",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .positive-sentiment { color: #28a745; font-weight: bold; }
    .negative-sentiment { color: #dc3545; font-weight: bold; }
    .neutral-sentiment { color: #6c757d; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Reddit Sentiment Analysis Dashboard")
    st.markdown("---")
    
    if 'reddit_client' not in st.session_state:
        st.session_state.reddit_client = RedditClient()
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = SentimentAnalyzer()
    if 'posts_data' not in st.session_state:
        st.session_state.posts_data = []
        
    st.sidebar.header("🔧 Controls")
    
    st.sidebar.subheader("🔴 Reddit Configuration")
    subreddit = st.sidebar.text_input(
        "Subreddit Name:",
        value="python",
        help="Enter subreddit name without 'r/' (e.g., 'python', 'technology')"
    )
    
    reddit_query = st.sidebar.text_input(
        "Search Query (optional):",
        value="",
        help="Leave empty to get hot posts"
    )
    
    reddit_count = st.sidebar.slider("Posts per fetch:", 10, 100, 50)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        time_filter = st.selectbox(
            "Time Filter:",
            ["hour", "day", "week", "month", "year"],
            index=2
        )
        
    with col2:
        sort_by = st.selectbox(
            "Sort By:",
            ["hot", "new", "top"],
            index=0
        )
        
    if st.sidebar.button("🔍 Fetch New Posts", type="primary"):
        fetch_reddit_posts(subreddit, reddit_query, reddit_count, time_filter, sort_by)
        
    if st.session_state.posts_data:
        df = pd.DataFrame(st.session_state.posts_data)
        
        display_metrics(df)
        st.markdown("---")
        display_charts(df)
        st.markdown("---")
        display_recent_posts(df)
    else:
        st.info("👆 Click 'Fetch New Posts' to start analyzing Reddit content!")
        
if __name__ == "__main__":
    main()