"""
UI Components: Reusable Streamlit interface elements
"""

import re
import pandas as pd
import streamlit as st

from html import unescape
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class UIComponents:
    """
    Reusable UI components for News Article Analyzer
    
    Why modular components?
    - Reusability across pages
    - Consistent design
    - Easy to maintain
    - Same pattern as Twitter project
    """
    
    @staticmethod
    def setup_page_config():
        """
        Contains the config for Streamlit page settings
        
        Why these settings?
        - wide layout: More space for charts
        - Custom icon: Professional look
        - Title: Browser Tab name
        """
        
        st.set_page_config(
            page_title="News Sentiment Analyzer",
            page_icon="📰",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    @staticmethod
    def apply_custom_css():
        st.markdown("""
        <style>
        /* Main color scheme */
        :root {
            --primary-color: #1DA1F2;
            --positive-color: #28a745;
            --negative-color: #dc3545;
            --neutral-color: #6c757d;
            --card-bg: #f8f9fa;
            --text-dark: #212529;
            --text-light: #6c757d;
        }

        /* Metric cards */
        .metric-card {
            background-color: var(--card-bg);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid var(--primary-color);
            margin: 10px 0;
        }

        /* Article cards - FIX: Dark text on light background */
        .article-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .article-card h3 {
            color: var(--text-dark) !important;  /* FIX: Dark text */
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .article-card p {
            color: var(--text-light) !important;  /* FIX: Gray text */
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 15px;
        }

        .article-card a {
            color: var(--primary-color) !important;  /* FIX: Blue link */
            text-decoration: none;
            font-weight: 500;
        }

        .article-card a:hover {
            text-decoration: underline;
        }

        /* Sentiment badges */
        .sentiment-positive {
            background-color: var(--positive-color);
            color: white;
            padding: 5px 12px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
        }

        .sentiment-negative {
            background-color: var(--negative-color);
            color: white;
            padding: 5px 12px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
        }

        .sentiment-neutral {
            background-color: var(--neutral-color);
            color: white;
            padding: 5px 12px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
        }

        /* Article metadata */
        .article-meta {
            color: var(--text-light);
            font-size: 13px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e9ecef;
        }

        /* Headers */
        h1 {
            color: #1DA1F2;
        }

        h2 {
            color: var(--text-dark);
            margin-top: 20px;
        }

        /* Streamlit expander fix */
        .streamlit-expanderHeader {
            color: var(--text-dark) !important;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)
        
    @staticmethod
    def render_header():
        """
        Display app header with branding
        
        Why separate method?
        - Consistent header across all pages
        - Easy to update branding
        - Professional look
        """
        
        st.markdown("""
        <h1 style='text-align: center;'>
            📰 News Article Sentiment Analyzer
        </h1>
        <p style='text-align: center; color: #6c757d;'>
            Real-time sentiment analysis of news from Indian & International sources
        </p>
        <hr>
        """, unsafe_allow_html=True)
        
    @staticmethod
    def render_sidebar_filters(sources: List[str]) -> Dict:
        """
        Renders sidebar with search filters
        
        Why sidebar?
        - Keeps main area for content
        - Easy access to filters
        
        Returns:
            Dictionary with user selections
        """
        
        with st.sidebar:
            st.header("🔍 Search Filters")
            
            query = st.text_input(
                "Search Keywords",
                placeholder="e.g., artificial intelligence, climate change",
                help="Enter keywords to search for articles"
            )
            
            st.subheader("🌍 Region Selection")
            region = st.radio(
                "Choose news region:",
                options=['both', 'indian', 'international'],
                format_func=lambda x: {
                    'both': '🌍 Both Indian & International',
                    'indian': '🇮🇳 Indian News Only',
                    'international': '🌍 International News Only',
                }[x]
            )
            
            strict_match = st.checkbox(
                "🎯 Strict keyword matching",
                value=True,
                help="Require ALL keywords to be present in article"
            )
            
            st.subheader("📰 News Sources")
            
            if region == 'indian':
                st.caption(f"📊 {len([s for s in sources if 'india' in s.lower() or 'hindu' in s.lower() or 'times' in s.lower()])} Indian sources available")
            elif region == 'international':
                st.caption(f"📊 {len([s for s in sources if s not in ['the-times-of-india', 'the-hindu']])} International sources available")
            else:
                st.caption(f"📊 {len(sources)} total sources available")
            
            selected_sources = st.multiselect(
                "Filter by sources (optional):",
                options=sources,
                help="Leave empty to search all sources"
            )
            
            if selected_sources:
                st.success(f"✅ {len(selected_sources)} sources selected")
            else:
                st.info("🌐 Searching all available sources")
            
            st.subheader("📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                from_date = st.date_input(
                    "From",
                    value=datetime.now() - timedelta(days=1),
                    max_value=datetime.now()
                )
            with col2:
                to_date = st.date_input(
                    "To",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
                
            st.subheader("⚙️ Settings")
            max_articles = st.slider(
                "Maximum articles:",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )
            
            search_clicked = st.button(
                "🔍 Search Articles",
                type="primary",
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("""
            ### 📊 About
            - **Sources**: 30+ verified publications
            - **Sentiment**: TextBlob + VADER
            - **Update**: 24-hour delay (Free Plan)
            - **API Limit**: 100 requests/day

            ### ⚠️ Free Plan Limits
            - ⏰ 24-hour article delay
            - 📅 Search last 30 days
            - 🔢 100 requests/day
            """)
            
        return {
            'query': query,
            'region': region,
            'strict_match': strict_match,
            'sources': selected_sources if selected_sources else None,
            'from_date': datetime.combine(from_date, datetime.min.time()),
            'to_date': datetime.combine(to_date, datetime.max.time()),
            'max_articles': max_articles,
            'search_clicked': search_clicked
        }
        
    @staticmethod
    def render_metrics(summary: Dict, regional_stats: Optional[Dict] = None):
        """
        Display key metrics in cards
        
        Why metrics?
        - Quick overview of data
        - At-a-glance insights
        - Comparison between regions (NEW!)
        
        Args:
            summary: Overall sentiment summary
            regional_stats: Regional comparison data
        """
        
        st.subheader("📊 Sentiment Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📄 Total Articles",
                value=summary['total_articles']
            )
            
        with col2:
            st.metric(
                label="😊 Positive",
                value=f"{summary['positive_percentage']}%",
                delta=f"{summary['positive_count']} articles"
            )
        
        with col3:
            st.metric(
                label="😐 Neutral",
                value=f"{summary['neutral_percentage']}%",
                delta=f"{summary['neutral_count']} articles"
            )
        
        with col4:
            st.metric(
                label="😞 Negative",
                value=f"{summary['negative_percentage']}%",
                delta=f"{summary['negative_count']} articles"
            )
            
        if regional_stats and regional_stats.get('comparison'):
            st.markdown("---")
            st.subheader("🌍 Regional Comparison")
            
            indian_stats = regional_stats.get('indian')
            intl_stats = regional_stats.get('international')
            comparison = regional_stats.get('comparison')
            
            if indian_stats and intl_stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🇮🇳 Indian News")
                    st.metric(
                        label="Total Articles",
                        value=indian_stats['total_articles']
                    )
                    st.metric(
                        label="Positive",
                        value=f"{indian_stats['positive_pct']}%"
                    )
                    st.metric(
                        label="Avg Score",
                        value=f"{indian_stats['avg_score']:.3f}"
                    )
        
                with col2:
                    st.markdown("### 🌍 International News")
                    st.metric(
                        label="Total Articles",
                        value=intl_stats['total_articles']
                    )
                    st.metric(
                        label="Positive",
                        value=f"{intl_stats['positive_pct']}%"
                    )
                    st.metric(
                        label="Avg Score",
                        value=f"{intl_stats['avg_score']:.3f}"
                    )
                    
                with col3:
                    st.markdown("### 📊 Comparison")
                    if comparison:
                        sentiment_diff = comparison['sentiment_diff']
                        st.metric(
                            label="Sentiment Difference",
                            value=f"{abs(sentiment_diff):.3f}",
                            delta=f"Indian {'more' if comparison['sentiment_diff'] > 0 else 'less'} positive"
                        )
                        st.metric(
                            label="More Positive Region",
                            value=comparison['more_positive'].title()
                        )
                    else:
                        st.write("No comparison data available")
                    
    @staticmethod
    def render_article_card(article: Dict):
        """
        Displays a single article as card
        
        Why cards?
        - Good for visual separation 
        - Easy to scan & Include image preview
        """
        
        def clean_html(text: str) -> str:
            """Remove HTML tags and decode entities"""
            if not text:
                return "No description available"

            # Decode HTML entities (e.g., &amp; → &)
            text = unescape(text)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Truncate if too long
            if len(text) > 300:
                text = text[:200] + "..."

            return text
        
        sentiment = article.get('sentiment_label', article.get('sentiment', 'neutral'))
        if isinstance(sentiment, dict):
            sentiment = sentiment.get('sentiment', 'neutral')
            
        sentiment_class = f"sentiment-{sentiment.lower()}"
        
        title = clean_html(article.get('text', article.get('title', 'No Title')))
        description = clean_html(article.get('description', article.get('full_text', '')))
        
        timestamp = article.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
                
        time_str = timestamp.strftime('%Y-%m-%d %H:%M') if isinstance(timestamp, datetime) else str(timestamp)
        
        with st.container():
            st.markdown(f"### {title}")
            st.write(description)

            sentiment_colors = {
                'positive': '🟢',
                'negative': '🔴',
                'neutral': '🟡'
            }
            sentiment_emoji = sentiment_colors.get(sentiment.lower(), '⚪')
            st.markdown(f"{sentiment_emoji} **{sentiment.upper()}**")

            # Metadata in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📰 {article.get('source', 'Unknown')}")
            with col2:
                st.caption(f"🌍 {article.get('region', 'unknown').title()}")
            with col3:
                st.caption(f"📅 {time_str}")

            # Read more link
            url = article.get('url', '#')
            if url != '#':
                st.markdown(f"[📖 Read Full Article →]({url})")

            st.markdown("---")
        
    @staticmethod
    def render_articles_table(df: pd.DataFrame):
        """
        This displays articles in interactive table
        
        Why table?
        - Sortable and Filterable columns
        - Exportable into Files (CSV)
        """
        
        if df.empty:
            st.warning("No articles to display")
            return

        st.subheader("📋 Article Details")
        
        display_columns = [
            'text', 'source', 'region', 'sentiment', 'score', 'timestamp'
        ]
        
        available_cols = [col for col in display_columns if col in df.columns]
        
        if not available_cols:
            st.error("❌ Required columns not found")
            return
        
        display_df = df[available_cols].copy()
        
        def clean_html(text):
            if not isinstance(text, str):
                return text
            text = unescape(text)
            text = re.sub(r'<[^>]+>', '',text)
            return ' '.join(text.split())
        
        if 'text' in display_df.columns:
            display_df['text'] = display_df['text'].apply(clean_html)
        
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            
        if 'sentiment' in display_df.columns:
            display_df['sentiment'] = display_df['sentiment'].apply(
                lambda x: x.get('sentiment', x) if isinstance(x, dict) else x
            )   
                 
        display_df.columns = [
            'Title', 'Source', 'Region', 'Sentiment', 'Score', 'Published'
        ]
        
        st.dataframe(
            display_df,
            width='stretch',
            height=400
        )
        
        st.caption(f"📊 **Source Distribution:** {df['source'].nunique()} unique sources")
        
         # Show top 5 sources
        top_sources = df['source'].value_counts().head(5)
        st.caption("**Top Sources:** " + ", ".join([f"{src} ({count})" for src, count in top_sources.items()]))
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"news_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="csv_download",
            use_container_width=True
        )
        
if __name__ == "__main__":
    UIComponents.setup_page_config()
    UIComponents.apply_custom_css()
    UIComponents.render_header()
    
    test_summary = {
        'total_articles': 100,
        'positive_count': 45,
        'neutral_count': 30,
        'negative_count': 25,
        'positive_percentage': 45.0,
        'neutral_percentage': 30.0,
        'negative_percentage': 25.0
    }
    
    UIComponents.render_metrics(test_summary)
    
    st.success("✅ UI Components loaded successfully!")