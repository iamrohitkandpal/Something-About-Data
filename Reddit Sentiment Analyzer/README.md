# 🔴 Reddit Sentiment Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green.svg)](https://en.wikipedia.org/wiki/Sentiment_analysis)
[![Status](https://img.shields.io/badge/Status-API%20Pending-yellow.svg)](https://github.com)

> 🎭 **Track Reddit sentiment in real-time!** Analyze subreddit discussions, discover community opinions, and understand the pulse of Reddit communities with AI-powered sentiment analysis.

---

## ⚠️ Current Project Status

**🟡 Development Paused - API Access Pending**

This project is currently **on hold** due to Reddit API access issues. The developer has submitted the Reddit Developer Application but is experiencing technical difficulties with Reddit's application form, preventing API credential approval.

**What's Complete:**
- ✅ Full dashboard UI/UX
- ✅ Sentiment analysis engine
- ✅ Data visualization components
- ✅ All features tested with simulated data

**What's Pending:**
- ⏳ Reddit API credentials approval
- ⏳ Live data integration
- ⏳ Production deployment

**Expected Timeline:**
- Work will resume immediately after Reddit API access is granted
- Estimated integration time: 2-3 days post-approval

---

## 🌟 What This Project Does

This real-time Reddit sentiment analysis system provides:

- 🔍 **Subreddit Monitoring** - Track posts from any subreddit
- 🧠 **AI-Powered Analysis** - Dual sentiment analysis using TextBlob + VADER
- 📊 **Interactive Dashboard** - Real-time visualizations and metrics
- 🎯 **Customizable Searches** - Filter by keywords, time, and popularity
- 📈 **Trend Tracking** - Monitor sentiment changes over time
- 💬 **Engagement Metrics** - Track upvotes, comments, and post scores

---

## 📂 Project Structure

```
Reddit Sentiment Analyzer/
├── 📄 main.py              # Main Streamlit application
├── 🔐 .env                 # API credentials (not included)
├── 📁 reddit-env/          # Python virtual environment
│   ├── Scripts/            # Environment activation scripts
│   └── Lib/                # Installed packages
└── 📖 README.md            # This documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **Reddit API credentials** (pending approval)

### Installation

1. **Navigate to project directory:**
   ```bash
   cd "Reddit Sentiment Analyzer"
   ```

2. **Activate the virtual environment:**
   ```bash
   # Windows
   reddit-env\Scripts\activate

   # macOS/Linux
   source reddit-env/bin/activate
   ```

3. **Install dependencies (if needed):**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USER_AGENT=your_app_name_v1.0
   ```

   > **Note:** Currently, API credentials are pending. The app runs with simulated data for demonstration.

5. **Launch the dashboard:**
   ```bash
   streamlit run main.py
   ```

   The dashboard will automatically open in your browser at `http://localhost:8501`

---

## 🛠️ Technology Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Primary programming language |
| **Streamlit** | 1.28+ | Interactive web dashboard framework |
| **PRAW** | 7.7+ | Reddit API wrapper |
| **Pandas** | 2.0+ | Data manipulation and analysis |
| **Plotly** | 5.17+ | Interactive visualizations |

### NLP & Sentiment Analysis

| Library | Purpose |
|---------|---------|
| **TextBlob** | General text sentiment analysis |
| **VADER Sentiment** | Social media-specific sentiment analysis |
| **NLTK** | Natural language processing toolkit |

### Utilities

| Library | Purpose |
|---------|---------|
| **python-dotenv** | Environment variable management |
| **datetime** | Timestamp handling |
| **os** | System operations |

---

## 📊 Features & Functionality

### 1. **Dashboard Controls**

#### Search Configuration
- **Subreddit Selection** - Enter any subreddit name (e.g., "python", "technology")
- **Search Query** - Filter posts by keywords (optional)
- **Post Limit** - Adjust number of posts to fetch (10-100)
- **Time Filter** - Choose from: hour, day, week, month, year, all
- **Sort Options** - hot, new, top, rising, controversial

#### Real-Time Controls
- **Fetch New Posts** - Manually trigger data refresh
- **Clear Data** - Reset dashboard and start fresh
- **Auto-refresh** - (Planned) Automatic periodic updates

### 2. **Sentiment Analysis Engine**

The project uses a **hybrid sentiment analysis approach**:

```python
# Combined TextBlob + VADER analysis
textblob_score = TextBlob(text).sentiment.polarity  # -1 to +1
vader_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']

# Decision logic
if textblob_score > 0.1 and vader_score > 0.05:
    sentiment = "Positive"
elif textblob_score < -0.1 and vader_score < -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"
```

**Why Two Algorithms?**
- **TextBlob**: Better for formal text and general sentiment
- **VADER**: Optimized for social media, handles emojis and slang
- **Combined**: Improves accuracy and reduces false classifications

### 3. **Key Metrics Display**

The dashboard shows:

```
📊 Total Posts: 245        😊 Positive: 45.2%
😞 Negative: 23.8%         😐 Neutral: 31.0%
```

Additional metrics:
- Average confidence score
- Total engagement (upvotes + comments)
- Most active users
- Peak posting times

### 4. **Visualizations**

#### Sentiment Distribution (Pie Chart)
- Visual breakdown of positive/negative/neutral posts
- Color-coded: Green (positive), Red (negative), Gray (neutral)
- Interactive hover details

#### Sentiment Timeline (Line Chart)
- Track sentiment changes over time
- Identify trends and patterns
- Spot sudden sentiment shifts

#### Top Posts by Engagement
- Ranked list of most popular posts
- Shows title, score, comments, and sentiment
- Sortable and filterable

#### Recent Posts Table
- Live feed of analyzed posts
- Columns: Title, Sentiment, Score, Comments, Time, User
- Interactive sorting and filtering

### 5. **Simulated Data Mode**

**Current Functionality:**

Since Reddit API access is pending, the application generates realistic fake data for testing:

```python
sample_texts = [
    "This new Python library is amazing! Highly recommend it for data analysis.",
    "Having major issues with this framework. Documentation is terrible.",
    "Just launched my first data pipeline project! Feeling accomplished.",
    # ... more diverse samples
]
```

**Simulated data includes:**
- ✅ Realistic post titles and text
- ✅ Random but plausible scores (1-1000)
- ✅ Variable comment counts (0-500)
- ✅ Timestamps within specified time range
- ✅ Random usernames
- ✅ Upvote ratios (0.0-1.0)

---

## 🎯 Use Cases

### Business Applications

1. **Brand Monitoring**
   - Track mentions of your company/product on Reddit
   - Monitor customer sentiment and feedback
   - Identify potential issues early

2. **Market Research**
   - Understand community opinions on topics
   - Discover trending discussions
   - Analyze competitor sentiment

3. **Product Development**
   - Gather feature requests from communities
   - Identify pain points and common complaints
   - Validate product ideas

### Research & Analysis

1. **Social Media Research**
   - Study online community behavior
   - Analyze discussion trends
   - Track sentiment evolution

2. **Event Impact Analysis**
   - Monitor sentiment during product launches
   - Track reactions to news events
   - Measure campaign effectiveness

3. **Content Strategy**
   - Identify popular topics in communities
   - Understand what resonates with audiences
   - Optimize content timing and approach

---

## 🔐 API Setup Guide

### Getting Reddit API Credentials

**⚠️ Currently Experiencing Issues**

The developer has encountered problems with Reddit's developer application form, causing delays in API approval. Here's the standard process (which will be completed once the form issue is resolved):

1. **Create Reddit Account**
   - Sign up at [reddit.com](https://reddit.com)
   - Verify email address

2. **Access Developer Portal**
   - Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
   - Scroll to "Developed Applications"
   - Click "Create App" or "Create Another App"

3. **Fill Application Form**
   - **Name**: Your application name
   - **App Type**: Select "script"
   - **Description**: Brief app description
   - **About URL**: (optional)
   - **Redirect URI**: http://localhost:8080
   - Click "Create app"

4. **Get Credentials**
   - **Client ID**: Under app name (alphanumeric string)
   - **Client Secret**: Click "edit" to reveal
   - **User Agent**: Format: `platform:app_id:version (by /u/username)`

5. **Configure .env File**
   ```env
   REDDIT_CLIENT_ID=your_14_char_client_id
   REDDIT_CLIENT_SECRET=your_27_char_secret
   REDDIT_USER_AGENT=windows:myapp:v1.0 (by /u/your_username)
   ```

**Current Status:** Form submission blocked due to technical issues on Reddit's end. Awaiting resolution from Reddit support.

---

## 🏗️ Architecture & Code Structure

### Main Components

#### 1. RedditClient Class
```python
class RedditClient:
    """Handles Reddit API interactions"""
    
    def __init__(self):
        # Initialize PRAW client with credentials
        # Handle connection errors gracefully
        
    def search_subreddit_posts(self, subreddit_name, query, limit, time_filter, sort_by):
        # Fetch posts from specified subreddit
        # Apply filters and sorting
        # Return structured post data
        
    def _generate_fake_reddit_data(self, count, subreddit_name):
        # Generate simulated data for testing
        # Used when API unavailable
```

#### 2. SentimentAnalyzer Class
```python
class SentimentAnalyzer:
    """Performs sentiment analysis on text"""
    
    def __init__(self):
        # Initialize TextBlob and VADER
        
    def analyze_sentiment(self, text):
        # Clean and preprocess text
        # Run both algorithms
        # Combine results
        # Return sentiment + confidence score
```

#### 3. Dashboard Functions
```python
def fetch_reddit_posts(subreddit, query, count, time_filter, sort_by):
    """Fetch and analyze posts"""
    
def display_metrics(df):
    """Show key statistics"""
    
def display_charts(df):
    """Render visualizations"""
    
def display_recent_posts(df):
    """Show posts table"""
```

### Data Flow

```
User Input → RedditClient → Raw Posts
              ↓
        SentimentAnalyzer → Analyzed Data
              ↓
        Pandas DataFrame → Aggregations
              ↓
        Streamlit Dashboard → Visualizations
```

---

## 📈 Sentiment Analysis Details

### Algorithm Comparison

| Feature | TextBlob | VADER |
|---------|----------|-------|
| **Strength** | General text | Social media |
| **Output** | Polarity (-1 to +1) | Compound score (-1 to +1) |
| **Emoji Support** | Limited | Excellent |
| **Slang Recognition** | Basic | Advanced |
| **Capitalization** | Ignores | Considers (EMPHASIS) |
| **Punctuation** | Basic | Considers (!!! intensity) |

### Decision Logic

```python
# Positive criteria
textblob_polarity > 0.1 AND vader_compound > 0.05

# Negative criteria
textblob_polarity < -0.1 AND vader_compound < -0.05

# Neutral (everything else)
# Includes mixed sentiment, unclear tone, or neutral statements
```

### Confidence Score Calculation

```python
# For clear sentiment (positive/negative)
confidence = (abs(textblob_polarity) + abs(vader_compound)) / 2

# For neutral sentiment
confidence = 1 - abs(textblob_polarity - vader_compound)

# Range: 0.0 (uncertain) to 1.0 (very confident)
```

---

## 🔧 Configuration Options

### Environment Variables

```env
# Required
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Optional (future features)
DATABASE_URL=postgresql://...
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

### Customization

Modify `main.py` to adjust:

- **Post limits**: Change slider range (line ~320)
- **Colors**: Update sentiment color scheme (line ~305)
- **Refresh intervals**: Adjust timing (line ~340)
- **Sample texts**: Edit fake data samples (line ~90)

---

## 🐛 Troubleshooting

### Common Issues

#### "Reddit API Initialization Failed"
- **Cause**: Missing or invalid API credentials
- **Solution**: Check `.env` file, verify credentials
- **Workaround**: App will use simulated data automatically

#### "No posts found for query"
- **Cause**: Search query too specific or subreddit empty
- **Solution**: Broaden search terms, try different subreddit
- **Check**: Verify subreddit name spelling

#### "Module not found" errors
- **Cause**: Missing dependencies
- **Solution**: 
  ```bash
  pip install -r requirements.txt
  ```

#### Dashboard not loading
- **Cause**: Port already in use or Streamlit issue
- **Solution**:
  ```bash
  # Try different port
  streamlit run main.py --server.port 8502
  
  # Or kill existing Streamlit process
  taskkill /F /IM streamlit.exe  # Windows
  pkill -9 streamlit  # Mac/Linux
  ```

---

## 🚧 Roadmap & Future Features

### Phase 1: API Integration (Pending)
- [ ] Reddit API access approval
- [ ] Live data fetching
- [ ] Real-time subreddit monitoring
- [ ] Rate limit handling

### Phase 2: Enhanced Features
- [ ] Historical data storage (SQLite/PostgreSQL)
- [ ] User authentication system
- [ ] Saved search configurations
- [ ] Custom dashboards
- [ ] Export to CSV/JSON
- [ ] Email/Slack alerts

### Phase 3: Advanced Analytics
- [ ] Trend prediction using time series
- [ ] Topic modeling (LDA)
- [ ] User influence scoring
- [ ] Sentiment correlation analysis
- [ ] Multi-subreddit comparison
- [ ] Network analysis (user interactions)

### Phase 4: Machine Learning
- [ ] Custom sentiment model training
- [ ] Spam/bot detection
- [ ] Automatic topic categorization
- [ ] Anomaly detection
- [ ] Sentiment forecasting

---

## 📝 Dependencies

### Required Packages

```txt
streamlit>=1.28.0
praw>=7.7.0
pandas>=2.0.0
plotly>=5.17.0
textblob>=0.17.1
vaderSentiment>=3.3.2
python-dotenv>=1.0.0
nltk>=3.8.1
```

### Installation

```bash
# Install all at once
pip install -r requirements.txt

# Or individually
pip install streamlit praw pandas plotly textblob vaderSentiment python-dotenv nltk
```

---

## 🤝 Contributing

Contributions are welcome once the API access is secured!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Commit with clear messages**
   ```bash
   git commit -m "Add: Amazing new feature"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation accordingly
- Test with both real and simulated data

---

## 📄 License

This project is available for educational and personal use. Please credit the original developer when using or modifying the code.

---

## 👤 Developer

**Project Status:** On hold pending Reddit API approval  
**Contact:** [Your Email/GitHub]  
**Last Updated:** November 2025

---

## 🙏 Acknowledgments

- **Streamlit** - For the amazing dashboard framework
- **PRAW Team** - For the excellent Reddit API wrapper
- **NLTK & TextBlob** - For NLP capabilities
- **VADER Sentiment** - For social media sentiment analysis
- **Reddit Community** - For being a great data source (once API works!)

---

## 📚 Additional Resources

### Learning Materials
- [PRAW Documentation](https://praw.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Sentiment Analysis Guide](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Reddit API Rules](https://www.reddit.com/wiki/api)

### Related Projects
- Twitter Sentiment Analysis (sister project - fully functional)
- Check `PROJECT_DOCUMENTATION_HINGLISH.md` for detailed technical explanations

---

**Note:** This project demonstrates sentiment analysis concepts and dashboard development. Full functionality will be available once Reddit API access is granted. The code is production-ready and tested with simulated data.

---

*Built with ❤️ and Python | Waiting for Reddit API approval 🕐*
