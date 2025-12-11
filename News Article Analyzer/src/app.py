"""
Main Application Starter: News Article Sentiment analyzer
Connects all components together
"""

import sys
import pandas as pd
import streamlit as st
import sqlite3
import json

from pathlib import Path
from datetime import datetime
from ui.charts import SentimentCharts
from api.news_client import NewsClient
from ui.components import UIComponents
from utils.data_adapter import DataAdapter
from analysis.preprocessor import TextPreprocessor
from analysis.sentiment_analyzer import SentimentAnalyzer

sys.path.append(str(Path(__file__).parent))

def main() -> None:
    """
    Flow:
    1. Setup UI
    2. Get user inputs
    3. Fetch articles
    4. Preprocess text
    5. Analyze sentiment
    6. Display results
    """
    
    # Setup Page
    UIComponents.setup_page_config()
    UIComponents.apply_custom_css()
    UIComponents.render_header()
    
    # Initializing Compnents
    @st.cache_resource
    def initialize_clients():
        """
        API clients initializetion (cached for performance)
        """
        
        return NewsClient(), SentimentAnalyzer(), TextPreprocessor()
    
    try:
        news_client, sentiment_analyzer, preprocessor = initialize_clients()
        st.success("✅ Clients initialized successfully")
        
        if hasattr(news_client, 'requests_today'):
            if news_client.requests_today > 50:
                st.warning(f"⚠️ API Usage: {news_client.requests_today}/100 requests used today")
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        st.stop()

    @st.cache_resource
    def get_db_connection():
        connection = sqlite3.connect('articles.db')
        cursor = connection.cursor()

        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA cache_size = 10000")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                source TEXT,
                author TEXT,
                title TEXT,
                description TEXT,
                url TEXT,
                published_at TEXT,
                content TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                region TEXT,
                cleaned_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()
        return connection


    @st.cache_data(ttl=3600)
    def get_cached_sources():
        return news_client.get_sources_by_region(region='both')
        
    # Getting the availables sources
    sources_data = get_cached_sources()
    all_sources = [s['id'] for s in sources_data.get('indian', [])] + \
                  [s['id'] for s in sources_data.get('international', [])]
                  
    filters = UIComponents.render_sidebar_filters(all_sources)
    
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'regional_stats' not in st.session_state:
        st.session_state.regional_stats = None
    if 'filters' not in st.session_state:
        st.session_state.filters = None
    
    if not filters['query']:
        st.info("👈 Enter search keywords in the sidebar to get started")
        st.markdown("---")
        st.subheader("💡 Example Queries: Adani, Apple, Elections")
        st.stop()

    # Logic: Check DB -> Fetxh -> Analyze -> Save
    if filters['search_clicked']:
        st.session_state.analysis_complete = False
        
        with st.spinner('🔍 Checking Local DataBase & Fetching articles...'):
            conn = get_db_connection()
            cursor = conn.cursor()

            today_str = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT title, description, url, published_at, source, content, region, sentiment_label, sentiment_score, cleaned_text
                FROM articles
                WHERE query = ? AND created_at LIKE ?
            """, (filters['query'], f"{today_str}%"))

            db_results  = cursor.fetchall()
            all_articles = []

            if db_results and len(db_results) > 0:
                st.info(f"⚡ Loaded {len(db_results)} articles from Database!")

                for row in db_results:
                    all_articles.append({
                        'title': row[0],
                        'description': row[1],
                        'url': row[2],
                        'published_at': row[3],
                        'source': row[4],
                        'content': row[5],
                        'region': row[6],
                        'sentiment': {
                            'sentiment': row[7],
                            'combined_score': row[8],
                        },
                        'processed_text': row[9]
                    })
            
            else:
                st.info("⚡ No articles found in Local Database")

                # Fetching articles
                raw_articles_dict = news_client.search_articles(
                    query=filters['query'],
                    region=filters['region'],
                    sources=filters['sources'],
                    strict_match=filters['strict_match'],
                    from_date=filters['from_date'],
                    to_date=filters['to_date'],
                )

                raw_articles = raw_articles_dict['combined'][:filters['max_articles']]
        
                if not raw_articles:
                    st.warning(f"⚠️ No articles found for '{filters['query']}'")
                    st.stop()

                st.write(f"🧹 Cleaning & Analyzing {len(raw_articles)} articles...")

                preprocessed_articles = preprocessor.preprocess_batch(raw_articles)

                all_articles = sentiment_analyzer.analyze_batch(preprocessed_articles)

                if all_articles:
                    params = []

                    for article in all_articles:
                        score = article.get('sentiment', {}).get('combined_score', 0.0)
                        label = article.get('sentiment', {}).get('sentiment', 'neutral')

                        params.append((
                            filters['query'],
                            article.get('source', 'Unknown'),
                            article.get('author', ''),
                            article.get('title', ''),
                            article.get('description', ''),
                            article.get('url', ''),
                            article.get('published_at', ''),
                            article.get('content', ''),
                            label,
                            score,
                            article.get('region', 'unknown'),
                            article.get('processed_text', '')
                        ))

                    cursor.executemany("""
                        INSERT INTO articles (
                            query, source, author, title, description, url, published_at, content, sentiment_label, sentiment_score, region, cleaned_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, params)

                    conn.commit()
                    st.toast(f"💾 Saved {len(all_articles)} analyzed results to DB!")
        
        with st.spinner('📊 Preparing Dashboard...'):
            # Conversion from list to DataFrame
            df = DataAdapter.articles_to_dataframe(all_articles)

            # Coversion of the articles to statistics
            summary = sentiment_analyzer.get_sentiment_summary(all_articles)

            # Converting the DataFrame to Regional Comparisons
            regional_stats = None
            if filters['region'] == 'both':
                regional_stats = DataAdapter.group_by_region(df)

        st.session_state.df = df
        st.session_state.summary = summary
        st.session_state.regional_stats = regional_stats
        st.session_state.filters = filters
        st.session_state.analysis_complete = True

    if st.session_state.analysis_complete and st.session_state.df is not None:
        df = st.session_state.df
        summary = st.session_state.summary
        regional_stats = st.session_state.regional_stats
        filters = st.session_state.filters
        
        st.markdown("---")

        # Rendering metrics
        UIComponents.render_metrics(summary, regional_stats)
        # Rendering charts
        SentimentCharts.create_dashboard(df, summary, regional_stats or {})
        st.markdown("---")
        # Rendering articles table
        UIComponents.render_articles_table(df)
        
        # Rendering article card (top 10)
        st.markdown("---")
        st.subheader("📰 Article Preview")
        st.markdown("**Top 10 Articles:**")
        for i, article in enumerate(df.head(10).to_dict('records')):
            display_title = article.get('title', article.get('text', 'No Title'))
            
            if len(display_title) > 80:
                display_title = display_title[:80] + "..."
                
            with st.expander(f"{i+1}. {display_title}"):
                UIComponents.render_article_card(article)
       
        # Rendering export options
        st.markdown("---")
        st.subheader("💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export (already in render_articles_table)
            st.info("📥 CSV export available above table")
            
        with col2:
            # Summary report
            report_content = f"""
            # Sentiment Analysis Report
            **Query:** {filters['query']}
            **Date Range:** {filters['from_date'].date()} to {filters['to_date'].date()}
            ## Summary
            - Total Articles: {summary['total_articles']}
            - Positive: {summary['positive_percentage']}%
            - Neutral: {summary['neutral_percentage']}%
            - Negative: {summary['negative_percentage']}%
            ## Average Scores
            - Sentiment: {summary['average_score']:.3f}
            - Median: {summary['median_score']:.3f}
            - Std Dev: {summary['std_dev']:.3f}
            ## Regional Breakdown
            {'### Indian News' if regional_stats and regional_stats.get('indian') else ''}
            {f"- Total: {regional_stats['indian']['total_articles']}" if regional_stats and regional_stats.get('indian') else ''}
            {f"- Positive: {regional_stats['indian']['positive_pct']}%" if regional_stats and regional_stats.get('indian') else ''}
            {'### International News' if regional_stats and regional_stats.get('international') else ''}
            {f"- Total: {regional_stats['international']['total_articles']}" if regional_stats and regional_stats.get('international') else ''}
            {f"- Positive: {regional_stats['international']['positive_pct']}%" if regional_stats and regional_stats.get('international') else ''}
            """ 
            st.download_button(
                label="📄 Download Summary Report",
                data=report_content,
                file_name=f"sentiment_report_{filters['query']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="report_download",
            )
            
        with col3:
            import json
            json_data = df.to_json(orient='records', date_format='iso')
            
            st.download_button(
                label="📊 Download Full Data (JSON)",
                data=json_data,
                file_name=f"news_data_{filters['query']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="json_download",
            )
            
        st.success("✅ Analysis Complete! Use buttons above to export data")

if __name__ == "__main__":
    main()