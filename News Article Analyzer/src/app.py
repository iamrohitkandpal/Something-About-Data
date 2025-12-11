"""
Main Application Starter: News Article Sentiment analyzer
Connects all components together
"""

import sys
import pandas as pd
import streamlit as st
import sqlite3
import json
import dotenv
import os

dotenv.load_dotenv()

from groq import Groq
from pathlib import Path
from datetime import datetime
from ui.charts import SentimentCharts
from api.news_client import NewsClient
from ui.components import UIComponents
from utils.data_adapter import DataAdapter
from analysis.preprocessor import TextPreprocessor
from analysis.sentiment_analyzer import SentimentAnalyzer

sys.path.append(str(Path(__file__).parent))

def generate_summary(articles, regional_stats, region_filter):
    """
    Generate a summary of the articles
    """

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    indian = [a['title'] for a in articles if a.get('region') == 'indian'][:5] 
    intl = [a['title'] for a in articles if a.get('region') == 'international'][:5] 

    indian_score = regional_stats.get('indian', {}).get('avg_score', 0) if regional_stats else 0
    intl_score = regional_stats.get('international', {}).get('avg_score', 0) if regional_stats else 0
        
    if region_filter == 'indian':
        prompt = f"""
        You are a news analyst. Summarize Indian media coverage in 3-4 simple sentences.
        **Indian Headlines** (Avg Sentiment: {indian_score:.2f}): {indian}
        Answer:
        1. Is the overall tone positive, negative, or mixed?
        2. What are the 2-3 main themes/topics being covered?
        3. Any notable bias or framing patterns?
        Rules: Simple language, no jargon, cite specific headline examples.
        """    
    elif region_filter == 'international':
        prompt = f"""
        You are a news analyst. Summarize International media coverage in 3-4 simple sentences.
        **International Headlines** (Avg Sentiment: {intl_score:.2f}): {intl}
        Answer:
        1. Is the overall tone positive, negative, or mixed?
        2. What are the 2-3 main themes/topics being covered?
        3. Any notable bias or framing patterns?
        Rules: Simple language, no jargon, cite specific headline examples.
        """
    else:
        prompt = f"""
        You are a news analyst. Compare Indian vs International media coverage in a SIMPLE, CONVERSATIONAL way.

        **Indian Headlines** (Score: {indian_score:.2f}): {indian}
        **International Headlines** (Score: {intl_score:.2f}): {intl}

        Write exactly 3-4 sentences that answer:
        1. Which side is more positive/negative and by how much?
        2. What's the KEY difference in how they're framing the story?
        3. One specific example from the headlines showing this difference.

        Rules:
        - Use simple language (8th grade reading level)
        - Start with "Indian media..." or "The key difference..."
        - Avoid jargon like "narrative framing" or "geopolitical"
        - Be specific, cite actual headline examples
        - Do NOT use bullet points or headers
        """

    try:
        summary = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        return summary.choices[0].message.content
    except Exception as e:
        return f"Could not generate summary: {e}"    


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
        connection = sqlite3.connect('articles.db', check_same_thread=False)
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
                ai_summary TEXT,
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
    if 'ai_summary' not in st.session_state:
        st.session_state.ai_summary = None
    
    if not filters['query']:
        st.info("👈 Enter search keywords in the sidebar to get started")
        st.markdown("---")
        st.subheader("💡 Example Queries: Adani, Apple, Elections")
        st.stop()

    # Logic: Check DB -> Fetxh -> Analyze -> Save
    if filters['search_clicked']:
        st.session_state.analysis_complete = False

        status_placeholder = st.empty()
        
        with status_placeholder.container():
            with st.spinner('🔍 Checking Local DataBase & Fetching articles...'):
                conn = get_db_connection()
                cursor = conn.cursor()

                today_str = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT title, description, url, published_at, source, content, region,  sentiment_label, sentiment_score, cleaned_text
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
                                query, source, author, title, description, url, published_at,   content, sentiment_label, sentiment_score, region,    cleaned_text
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, params)

                        conn.commit()
                        st.toast(f"💾 Saved {len(all_articles)} analyzed results to DB!")
        
        status_placeholder.empty()
            
        
        with st.spinner('📊 Preparing Dashboard...'):
            # Conversion from list to DataFrame
            df = DataAdapter.articles_to_dataframe(all_articles)
            # Coversion of the articles to statistics
            summary = sentiment_analyzer.get_sentiment_summary(all_articles)

            # Converting the DataFrame to Regional Comparisons
            regional_stats = None
            if filters['region'] == 'both':
                regional_stats = DataAdapter.group_by_region(df)

            if filters['region'] in ['both', 'indian', 'international'] and regional_stats:
                with st.spinner('Generating AI Summary...'):
                    ai_summary = generate_summary(all_articles, regional_stats, filters['region'])
                st.session_state.ai_summary = ai_summary

                cursor.execute("""
                    UPDATE articles
                    SET ai_summary = ?
                    WHERE query = ? AND created_at LIKE ?
                """, (ai_summary, filters['query'], f"{today_str}%"))
                conn.commit()

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
        ai_summary = st.session_state.ai_summary
        
        st.markdown("---")

        # The AI Summary Part
        if st.session_state.ai_summary:
            st.markdown("### 🧠 News Summary")
            st.markdown(f"""
                <div class="ai_summary-box" style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #4fc3f7;
                    color: white;
                    font-size: 16px;
                    line-height: 1.6;
                    font-weight: 500;
                ">
                    {st.session_state.ai_summary}
                </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

        # Rendering metrics
        UIComponents.render_metrics(summary, regional_stats)
        # Rendering charts
        with st.expander("📊 Detailed Charts View", expanded=False):
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