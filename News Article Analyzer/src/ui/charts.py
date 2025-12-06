"""
Charts: Interactive visualizations using Plotly
"""

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Dict, List
from wordcloud import WordCloud

class SentimentCharts:
    """
    This Class Creates interactive charts for sentiment visualization
    
    Why Plotly?
    - Interactive (hover, zoom, pan)
    - Professional Appearance & Easy to export
    """
    
    @staticmethod
    def empty_chart(message: str) -> go.Figure:
        """Return empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    @staticmethod
    def sentiment_distribution_pie(summary: Dict) -> go.Figure:
        """
        This Creates pie chart of sentiment distribution
        
        Why pie chart?
        - Show proportions clearly
        - Easy to understand & Standard for categorical data
        
        Args:
            summary: Sentiment summary dictionary
        
        Returns:
            Plotly figure object
        """
        
        labels = ['Positive', 'Neutral', 'Negative']
        values = [
            summary['positive_count'],
            summary['neutral_count'],
            summary['negative_count']
        ]
        colors = ['#28a745', '#6c757d', '#dc3545']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3,
            textinfo='label+percent',
            textfont_size=14,
            textposition='inside',
            pull=[0.05, 0, 0]
        )])
        
        fig.update_layout(
            title={
                'text': '📊 Sentiment Distribution',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=450,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1,
                font=dict(size=12)
            ),
            margin=dict(l=20, r=150, t=80, b=20)
        )
        
        return fig
    
    @staticmethod
    def sentiment_by_source(df: pd.DataFrame) -> go.Figure:
        """
        This creates stacked bar chart of sentiment by news source
        
        Why stacked bars?
        - Can compare multiple sources
        - Shows sentiment composition
        - Easy for spotting any biased sources
        
        Ex: "BBC has 60% positive, CNN has 40% positive"
        """
        
        if df.empty or 'source' not in df.columns:
            return SentimentCharts.empty_chart("No source data available")
        
        # Group by source and sentiment
        source_sentiment = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
        
        # Sort by total articles
        source_sentiment['total'] = source_sentiment.sum(axis=1)
        source_sentiment = source_sentiment.sort_values('total', ascending=True).head(10)
        source_sentiment = source_sentiment.drop('total', axis=1)
        
        fig = go.Figure()
        
        colors = {
            'positive': '#28a745',
            'neutral': '#6c757d',
            'negative': '#dc3545'
        }
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in source_sentiment.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.title(),
                    x=source_sentiment[sentiment],
                    y=source_sentiment.index,
                    orientation='h',
                    marker_color=colors[sentiment],
                    text=source_sentiment[sentiment],
                    textposition='inside'
                ))
                
        fig.update_layout(
            title='📰 Sentiment by News Source',
            xaxis_title='Number of Articles',
            yaxis_title='Source',
            barmode='stack',
            height=max(400, len(source_sentiment) * 40),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def source_distribution_pie(df: pd.DataFrame) -> go.Figure:
        """
        Shows distribution of articles across sources
        """
        
        if df.empty or 'source' not in df.columns:
            return SentimentCharts.empty_chart("No source data available")
        
        source_counts = df['source'].value_counts().head(10)
        
        fig = go.Figure(data=[go.Pie(
            labels=source_counts.index,
            values=source_counts.values,
            hole=0.3,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title='📰 Article Distribution by Source',
            height=450,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1
            )
        )
        
        return fig
    
    @staticmethod
    def sentiment_by_region(df: pd.DataFrame) -> go.Figure:
        """
        This compares sentiment between Indian and International news
        
        Why important?
        - Detects regional bias
        - Gives cultural perspective differences
        """
        
        if df.empty or 'region' not in df.columns:
            return SentimentCharts.empty_chart("No regional data available")
        
        # Group by region and sentiment
        region_sentiment = df.groupby(['region', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculating in percentages the grouping
        region_sentiment_pct = region_sentiment.div(region_sentiment.sum(axis=1), axis=0) * 100
        
        fig = go.Figure()
        
        colors = {
            'positive': '#28a745',
            'neutral': '#6c757d',
            'negative': '#dc3545'
        }
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in region_sentiment_pct.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.title(),
                    x=region_sentiment_pct.index,
                    y=region_sentiment_pct[sentiment],
                    marker_color=colors[sentiment],
                    text=region_sentiment_pct[sentiment].round(1).astype(str) + '%',
                    textposition='inside'
                ))
                
        fig.update_layout(
            title='🌍 Regional Sentiment Comparison',
            xaxis_title='Region',
            yaxis_title='Percentage',
            barmode='stack',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def sentiment_timeline(df: pd.DataFrame) -> go.Figure:
        """
        This will show the trend chart over time 
        """
        
        if df.empty or 'timestamp' not in df.columns:
            return SentimentCharts.empty_chart("No timeline data available")
        
        # Abhi We will covert the string dates to datetime format
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['timestamp'], errors='coerce').dt.date
        df_copy = df_copy.dropna(subset=['date'])
        
        if df_copy.empty:
            return SentimentCharts.empty_chart("No valid dates found")
        
        # Now daily sentiment score calculation
        daily_sentiment = df_copy.groupby('date')['score'].agg(['mean', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['mean'],
            mode='lines+markers',
            name='Average Sentiment',
            line=dict(color='#1DA1F2', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Date:</b> %{x}<br><b>Score:</b> %{y:.3f}<extra></extra>'
        ))
        
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Neutral"
        )
        
        fig.update_layout(
            title='📈 Sentiment Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Average Sentiment Score',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def sentiment_score_distribution(df: pd.DataFrame) -> go.Figure:
        """
        This will be showing a histogram of sentiment scores
        
        - Easy to see score distribution & statistical insights
        - Can help identify clustering (many neutral vs polarized)
        """
        
        if df.empty or 'score' not in df.columns:
            return SentimentCharts.empty_chart("No score data available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['score'],
            nbinsx=50,
            marker_color='#1DA1F2',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='📊 Sentiment Score Distribution',
            xaxis_title='Sentiment Score',
            yaxis_title='Number of Articles',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def top_scores_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        This will show top news sources by srticle count
        
        - Will indentify most active sources
        - Will help check data balance and to do source diversity assessment
        """
        
        if df.empty or 'source' not in df.columns:
            return SentimentCharts.empty_chart("No source data available")
        
        source_counts = df['source'].value_counts().head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=source_counts.values,
                y=source_counts.index,
                orientation='h',
                marker_color='#1DA1F2',
                text=source_counts.values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f'📰 Top {top_n} News Sources',
            xaxis_title='Number of Articles',
            yaxis_title='Source',
            height=max(400, top_n * 40)
        )
        
        return fig
    
    @staticmethod
    def polarity_vs_subjectivity(df: pd.DataFrame) -> go.Figure:
        """
        Scatter plot: Polarity vs Subjectivity
        
        Why useful?
        - Identify opinion vs fact-based articles
        - Quality assessment (high subjectivity = opinion pieces)
        - Journalistic style analysis
        
        Quadrants:
        - Top-right: Positive opinions
        - Top-left: Negative opinions
        - Bottom-right: Positive facts
        - Bottom-left: Negative facts
        """
        
        if df.empty or 'polarity' not in df.columns or 'subjectivity' not in df.columns:
            return SentimentCharts.empty_chart("No polarity/subjectivity data available")
        
        colors = {
            'positive': '#28a745',
            'neutral': '#6c757d',
            'negative': '#dc3545'
        }
        
        df_copy = df.copy()
        df_copy['color'] = df_copy['sentiment'].map(colors)
        
        fig = go.Figure()
        
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_df = df_copy[df_copy['sentiment'] == sentiment]
            
            if not sentiment_df.empty:
                fig.add_trace(go.Scatter(
                    x=sentiment_df['polarity'],
                    y=sentiment_df['subjectivity'],
                    mode='markers',
                    name=sentiment.title(),
                    marker=dict(
                        color=colors[sentiment],
                        size=8,
                        opacity=0.6
                    ),
                    text=sentiment_df.get('title', sentiment_df['text']).str[:50] + '...',
                    hovertemplate='<b>%{text}</b><br>Polarity: %{x:.2f}<br>Subjectivity: %{y:.2f}<extra></extra>'
                ))
            
        # Add quadrant lines
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='📊 Polarity vs Subjectivity Analysis',
            xaxis_title='Polarity (Negative ← → Positive)',
            yaxis_title='Subjectivity (Objective ← → Subjective)',
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def generate_wordcloud(df: pd.DataFrame, sentiment_filter: str = None):
        """
        This generate word cloud from article title
        - Visual Summary of common topics
        - Quick insight into coverage
        - An engaging visualization
        
        Args:
            sentiment_filter: Filter by sentiment
        """
        
        if df.empty or 'text' not in df.columns:
            st.info("📭 No data available for word cloud")
            return
        
        filtered_df = df.copy()
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
            
        if filtered_df.empty:
            st.info(f"📭 No {sentiment_filter} articles found")
            return
        
        if 'title' in filtered_df.columns:
            text = ' '.join(filtered_df['title'].astype(str).values)
        else:
            text = ' '.join(filtered_df['text'].astype(str).values)
        
        if not text or len(text.strip()) == 0:
            st.info(f"📭 No text content available")
            return
        
        try:
            # This will now generate a word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=50,
                relative_scaling=0.5,
                min_font_size=10,
                stopwords=set(['climate', 'change', 'news', 'said', 'says'])
            ).generate(text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')

            title = f"☁️ Word Cloud"
            if sentiment_filter:
                title += f" ({sentiment_filter.title()} Articles)"

            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"❌ Error generating word cloud: {e}")
            import traceback
            st.code(traceback.format_exc())
        
    @staticmethod
    def create_dashboard(df: pd.DataFrame, summary: Dict, regional_stats: Dict = None):
        """
        This will be our complete dashboard generator with all charts
        """
        
        if df.empty:
            st.warning("⚠️ No data to visualize")
            return
        
        st.markdown("---")
        st.header("📊 Visual Analytics")
        
        source_counts = df['source'].value_counts()
        if len(source_counts) > 0:
            max_source_pct = (source_counts.iloc[0] / len(df)) * 100
            if max_source_pct > 50:
                st.warning(f"⚠️ **Source Imbalance:** {source_counts.index[0]} has {max_source_pct:.1f}% of articles")
    
        
        # Row 1: Pie Chart + Regional Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = SentimentCharts.sentiment_distribution_pie(summary)
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            fig_sources_pie = SentimentCharts.source_distribution_pie(df)
            st.plotly_chart(fig_sources_pie, width='stretch')
                
        # Row 2: Timeline + Source Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            if regional_stats and 'region' in df.columns:
                fig_region = SentimentCharts.sentiment_by_region(df)
                st.plotly_chart(fig_region, width='stretch')
            else:                
                fig_score_dist = SentimentCharts.sentiment_score_distribution(df)
                st.plotly_chart(fig_score_dist, width='stretch')
                
        with col2:
            fig_timeline = SentimentCharts.sentiment_timeline(df)
            st.plotly_chart(fig_timeline, width='stretch')

        # Row 3: Polarity vs Subjectivity + Top Sources
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = SentimentCharts.polarity_vs_subjectivity(df)
            st.plotly_chart(fig_scatter, width='stretch')
        
        with col2:
            fig_top_sources = SentimentCharts.top_scores_chart(df)
            st.plotly_chart(fig_top_sources, width='stretch')
                
        # Row 4: Word Cloud Generator
        st.markdown("---")
        st.subheader("☁️ Word Clouds by Sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Positive Articles**")
            positive_count = len(df[df['sentiment'] == 'positive'])
            st.caption(f"{positive_count} articles")
            SentimentCharts.generate_wordcloud(df, 'positive')
        
        with col2:
            st.markdown("**Neutral Articles**")
            neutral_count = len(df[df['sentiment'] == 'neutral'])
            st.caption(f"{neutral_count} articles")
            SentimentCharts.generate_wordcloud(df, 'neutral')
        
        with col3:
            st.markdown("**Negative Articles**")
            negative_count = len(df[df['sentiment'] == 'negative'])
            st.caption(f"{negative_count} articles")
            SentimentCharts.generate_wordcloud(df, 'negative')
    
        
if __name__ == "__main__":
    import numpy as np
    
    st.set_page_config(layout="wide")
    st.title("📊 Test Charts")
    
    np.random.seed(42)
    n_articles = 100
    
    sample_df = pd.DataFrame({
        'text': [f'Article {i}' for i in range(n_articles)],
        'source': np.random.choice(['BBC', 'CNN', 'Reuters'], n_articles),
        'region': np.random.choice(['indian', 'international'], n_articles),
        'sentiment': np.random.choice(['positive', 'neutral', 'negative'], n_articles, p=[0.4, 0.3, 0.3]),
        'score': np.random.randn(n_articles) * 0.3,
        'polarity': np.random.randn(n_articles) * 0.4,
        'subjectivity': np.random.uniform(0, 1, n_articles),
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=n_articles, freq='H')
    })
    
    sample_summary = {
        'total_articles': n_articles,
        'positive_count': 40,
        'neutral_count': 30,
        'negative_count': 30,
        'positive_percentage': 40.0,
        'neutral_percentage': 30.0,
        'negative_percentage': 30.0
    }
    
    SentimentCharts.create_dashboard(sample_df, sample_summary)
    
    st.success("✅ All charts rendered successfully")
