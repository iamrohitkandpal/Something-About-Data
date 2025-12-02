# 📰 News Article Sentiment Analyzer

Real-time sentiment analysis of news articles from 80,000+ sources worldwide.

## 🎯 Features

- 🔍 Search articles by keywords across multiple news sources
- 📊 Sentiment analysis (Positive/Neutral/Negative)
- 📈 Interactive visualizations and charts
- 🌐 Support for 14 languages and 50+ countries
- 📅 Historical news analysis with date range filtering
- 🏢 Publication-specific sentiment tracking

## 🚀 Quick Start

### 1. Get News API Key
Visit [News API](https://newsapi.org/register) and get your free API key (1000 requests/day)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy example and add your API key
copy .env.example .env
# Edit .env and add: NEWS_API_KEY=your_actual_key
```

### 4. Run Application
```bash
streamlit run src/app.py
```

## 📊 Architecture

```
src/
├── api/
│   └── news_client.py          # News API integration
├── analysis/
│   ├── preprocessor.py         # Text cleaning & preparation
│   └── sentiment_analyzer.py   # Sentiment classification
├── ui/
│   ├── components.py           # Streamlit UI components
│   └── charts.py               # Plotly visualizations
├── utils/
│   └── data_adapter.py         # Data format conversion
└── app.py                       # Main application
```

## 🔧 How It Works

1. **NewsClient**: Fetches articles from News API
2. **Preprocessor**: Cleans article text (longer than tweets!)
3. **SentimentAnalyzer**: Uses TextBlob + VADER (same as Twitter project)
4. **DataAdapter**: Converts news articles → tweet-compatible format
5. **Streamlit UI**: Interactive dashboard with charts

## 📚 Comparison with Twitter Sentiment Analysis

| Feature | Twitter | News Articles |
|---------|---------|---------------|
| Text Length | 280 chars | 1000+ words |
| Update Frequency | Real-time | Hourly |
| Sources | User tweets | 80k+ publications |
| Preprocessing | Hashtags, mentions | Headlines, content |

## 🎓 Learning Objectives

- API integration patterns
- Handling longer text content
- Multi-source data aggregation
- Publication bias analysis

## 📖 News API Details

- **Free Tier**: 1000 requests/day
- **Sources**: BBC, CNN, TechCrunch, Reuters, etc.
- **Coverage**: 14 languages, 50+ countries
- **Docs**: https://newsapi.org/docs

---