import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import re

# Page Configuration
st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Zinc Background
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #27272a;
        color: #fafafa;
    }
    [data-testid="stHeader"] {
        background-color: #18181b;
    }
    [data-testid="stSidebar"] {
        background-color: #18181b;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #3f3f46;
        color: #fafafa;
        border: 1px solid #52525b;
    }
    .stButton > button {
        background-color: #3f3f46;
        color: #fafafa;
        border: 1px solid #52525b;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #52525b;
    }
    .movie-card {
        background-color: #3f3f46;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #a1a1aa;
        margin: 10px 0;
    }
    .movie-title {
        color: #fafafa;
        font-weight: bold;
        font-size: 16px;
    }
    .movie-genre {
        color: #a1a1aa;
        font-size: 14px;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    movies = pd.read_csv("movies.csv")
    # Load ratings with optimized dtypes to reduce memory
    ratings = pd.read_csv(
        "ratings.csv",
        dtype={
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float32',
            'timestamp': 'int32'
        }
    )
    return movies, ratings

@st.cache_resource
def initialize_models(movies):
    def clean_title(title):
        return re.sub("[^a-zA-Z0-9 ]", "", title)
    
    movies["clean_title"] = movies["title"].apply(clean_title)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(movies["clean_title"])
    
    tfidf_sparse = csr_matrix(tfidf)
    nn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_sparse)
    
    return vectorizer, tfidf, nn_model, movies

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

def search_movies(title, vectorizer, tfidf, nn_model, movies):
    title_clean = clean_title(title)
    query_vec = vectorizer.transform([title_clean])
    distances, indices = nn_model.kneighbors(query_vec)
    return movies.iloc[indices.flatten()][::-1]

def find_similar_movies(movie_id, movies, ratings):
    # Use optimized filtering with smaller dtypes
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 5)]["userId"].unique()
    
    if len(similar_users) == 0:
        return pd.DataFrame()
    
    similar_users_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_users_recs = similar_users_recs.value_counts() / len(similar_users)
    similar_users_recs = similar_users_recs[similar_users_recs > .1]
    
    if len(similar_users_recs) == 0:
        return pd.DataFrame()
    
    all_users = ratings[(ratings["movieId"].isin(similar_users_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    
    rec_percentages = pd.concat([similar_users_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# Load data with progress indicator
with st.spinner("⏳ Loading movie database..."):
    movies, ratings = load_data()
    vectorizer, tfidf, nn_model, movies = initialize_models(movies)

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("## 🎬")
with col2:
    st.markdown("## Movie Recommendation System")
    
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    rec_count = st.slider("Number of Recommendations", 5, 20, 10)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    🎯 **Hybrid Recommendation Engine**
    - Collaborative Filtering
    - Content-Based Analysis
    - Genre Matching
    """)
    
tab1, tab2 = st.tabs(["🔍 Search & Recommend", "📈 Explore"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("🔎 Search for a movie...", placeholder="Enter movie title")
    
    with col2:
        genre_filter = st.selectbox(
            "Filter by Genre",
            options=['All'] + sorted(list(movies['genres'].str.split('|').explode().unique()))
        )
    
    if search_query and len(search_query) > 2:
        results = search_movies(search_query, vectorizer, tfidf, nn_model, movies)
        
        if genre_filter != 'All':
            results = results[results['genres'].str.contains(genre_filter, na=False, case=False)]
        
        if len(results) > 0:
            st.markdown("### 🎬 Found Movies")
            selected_movie = st.selectbox(
                "Select a movie to get recommendations:",
                options=results.index,
                format_func=lambda x: f"{results.loc[x, 'title']} ({results.loc[x, 'genres']})"
            )
            
            if st.button("Get Recommendations", use_container_width=True):
                with st.spinner("🔍 Finding recommendations..."):
                    movie_id = results.loc[selected_movie, 'movieId']
                    recommendations = find_similar_movies(movie_id, movies, ratings)
                    
                    if len(recommendations) > 0:
                        st.markdown("### ⭐ Recommended Movies")
                        for idx, (_, row) in enumerate(recommendations.head(rec_count).iterrows(), 1):
                            st.markdown(f"""
                            <div class="movie-card">
                                <div class="movie-title">#{idx} {row['title']}</div>
                                <div class="movie-genre">{row['genres']}</div>
                                <small>Score: {row['score']:.2f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations found for this movie.")
        else:
            st.warning("No movies found. Try a different search.")

with tab2:
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{len(movies):,}")
    with col2:
        st.metric("Total Ratings", f"{len(ratings):,}")
    with col3:
        st.metric("Total Users", f"{ratings['userId'].nunique():,}")
    with col4:
        st.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")
    
    st.markdown("---")
    st.markdown("### 🏆 Top Rated Movies")
    top_movies = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    top_movies.columns = ['movieId', 'avg_rating', 'count']
    top_movies = top_movies[top_movies['count'] >= 50].nlargest(10, 'avg_rating')
    top_movies = top_movies.merge(movies, on='movieId')[['title', 'genres', 'avg_rating', 'count']]
    
    st.dataframe(top_movies, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a1a1aa; font-size: 12px;">
🎭 Discover your next favorite movie | Built with ❤️ for movie enthusiasts
</div>
""", unsafe_allow_html=True)