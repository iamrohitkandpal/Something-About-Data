"""
Main Application Starter: News Article Sentiment analyzer
Connects all components together
"""

import streamlit as st
import sys
from pathlib import Path

# Adding source folder to path
sys.path.append(str(Path(__file__).parent))

from api.news_client import NewsClient
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.preprocessor import TextPreprocessor
from utils.data_adapter import DataAdapter
from ui.components import UIComponents
from ui.charts import SentimentCharts

def main():
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
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        st.info("💡 Make sure your NEWS_API_KEY is set in .env file")
        st.stop()
        
    # Getting the availables sources
    sources_data = news_client.get_sources_by_region(region='both')
    all_sources = [s['id'] for s in sources_data.get('indian', [])] + \
                  [s['id'] for s in sources_data.get('international', [])]
                  
    filters = UIComponents.render_sidebar_filters(all_sources)
    
    if not filters['query']:
        st.info("👈 Enter search keywords in the sidebar to get started")
        
        st.markdown("---")
        st.subheader("💡 Example Queries")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Technology**
            - artificial intelligence            
            - machine learning
            - cryptocurrency
            - tech startups
            """)
            
        with col2:
            st.markdown("""
            **Politics**
            - election results
            - government policy
            - international relations
            - political debates
            """)
        
        with col3:
            st.markdown("""
            **Business**
            - stock market
            - economic growth
            - company earnings
            - trade agreements
            """)
            
        st.markdown("---")
        st.subheader("🌟 Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🇮🇳 Indian News
            - Times of India
            - The Hindu
            - And more...
            """)
        
        with col2:
            st.markdown("""
            ### 🌍 International News
            - BBC News
            - CNN
            - Reuters
            - And more...
            """)
            
        st.stop()
        
    if filters['search_clicked']:
        
        with st.spinner('🔍 Searching for articles...'):
            # Fetching articles
            articles = news_client.search_articles(
                query=filters['query'],
                region=filters['region'],
                sources=filters['sources'],
                from_date=filters['from_date'],
                to_date=filters['to_date'],
            )
            
            all_articles = articles['combined'][:filters['max_articles']]
        
            if not all_articles:
                st.warning(f"⚠️ No articles found for '{filters['query']}'")
                st.info("💡 Try different keywords or date range")
                st.stop()

            st.success(f"✅ Found {len(all_articles)} articles")
        
        with st.spinner('🧹 Preprocessing articles...'):
            processed_articles = preprocessor.preprocess_batch(all_articles)

        with st.spinner('🧠 Analyzing sentiment...'):
            analyzed_articles = sentiment_analyzer.analyze_batch(processed_articles)
            
            ############################3
            if analyzed_articles:
                print(f"\n DEBUG: FIRST ANALYZED ARTICLE:")
                first = analyzed_articles[0]
                print(f"   Sentiment data: {first.get('sentiment')}")
                print(f"   Type: {type(first.get('sentiment'))}")
        
                if isinstance(first.get('sentiment'), dict):
                    print(f"   Score: {first['sentiment'].get('combined_score')} (type: {type(first['sentiment'].get('combined_score'))})")
            ##################################

        with st.spinner('📊 Preparing visualizations...'):
            # Conversion from list to DataFrame
            df = DataAdapter.articles_to_dataframe(analyzed_articles)

            # Coversion of the articles to statistics
            summary = sentiment_analyzer.get_sentiment_summary(analyzed_articles)

            # Converting the DataFrame to Regional Comparisons
            regional_stats = None
            if filters['region'] == 'both':
                regional_stats = DataAdapter.group_by_region(df)

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
            with st.expander(f"{i+1}. {article['text'][:100]}..."):
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
            if st.button("📄 Generate Summary Report"):
                report = f"""
                # Sentiment Analysis Report
                **Query:** {filters['query']}
                **Date Range:** {filters['from_date'].date()} to {filters['to_date'].date()}
                # Summary
                 - Total Articles: {summary['total_articles']}
                 - Positive: {summary['positive_percentage']}%
                 - Neutral: {summary['neutral_percentage']}%
                 - Negative: {summary['negative_percentage']}%
                # Average Scores
                 - Sentiment: {summary['average_score']:.3f}
                 - Median: {summary['median_score']:.3f}
                 - Std Dev: {summary['std_dev']:.3f}
                """
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"sentiment_report_{filters['query']}.md",
                    mime="text/markdown"
                )
        with col3:
            st.success("✅ Analysis Complete!")

if __name__ == "__main__":
    main()