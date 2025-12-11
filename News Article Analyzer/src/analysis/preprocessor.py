"""
Preprocessing Unit of our project
- Cleaning and preparing news text for sentiment analysis
- Also for region based preprocessing
"""

import re
import string
from typing import Dict, List
import nltk
import ssl

def _setup_nltk():
    """
    Setting up nltk with ssl for data downloading
    """
    try:
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
            
        print("✅ NLTK paths configured")
        
    except Exception as e:
        print(f"⚠️ Could not setup NLTK: {e}")
        
def _download_nltk_data():
    """
    This lets us download all required NLTK datasets
    """
    import nltk
    try:
        datasets = ['punkt', 'punkt_tab', 'stopwords', 'brown', 'vader_lexicon']
        
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                print(f"📥 Downloading NLTK dataset: {dataset}")
                nltk.download(dataset, quiet=True)
        
        print("✅ NLTK data downloaded successfully")
        
    except Exception as e:
        print(f"⚠️ Could not download NLTK data: {e}")
        print("💡 Try manually: python -m nltk.downloader stopwaords, punkt, vader_lexicon")


_setup_nltk()
_download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
    
class TextPreprocessor:
    """
    This class will preporcess news articles with region-specific handling
    
    Why region-specific?
    - Indian English: Different terminology, names, places
    - International: Standard English conventions
    - Mixed language: Handle Hindi words in English text
    """
    
    def __init__(self):
        """
        This initializes class with stopwords and regional terms
        """
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("⚠️ Stopwords not found, downloading...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        # For standard English stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # For Indian-specific stopwords
        self.indian_neutral_terms = {
            'india', 'indian', 'delhi', 'mumbai', 'bangalore', 'bjp', 'congress', 'pm', 'cm', 'govt'
        }
        
        # For International neutral terms
        self.intl_neutral_terms = {
            'america', 'american', 'washington', 'london', 'paris',
            'president', 'minister', 'government'
        }
        
        # Some sentiment-carrying words
        self.sentiment_words = {
            'good', 'bad', 'great', 'terrible', 'amazing', 'awful',
            'excellent', 'poor', 'wonderful', 'horrible', 'fantastic', 'killing', 'rape', 'unemployment', 'mandir', 'temple', 'jihad'
        }
    
    def clean_text(self, text: str, region: str = 'international') -> str:
        """
        This will clean and normalize the text
        
        Args:
            text: Raw article text
            region: 'indian' or 'international'
        
        Returns:
            Cleaned text string
        """ 
        
        if not text or len(text.strip()) == 0:
            return ""
        
        text = text.lower()
        
        # Removing URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Removing email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Region-specific Cleaning
        if region == 'indian':
            # Handling Indian number formatting
            text = self._normalize_indian_numbers(text)
            
            # Handling Rupee symbols
            text = text.replace('₹', 'rs ')
            text = text.replace('rs.', 'rs ')
            
        else: 
            # Handle dollar/euro symbols
            text = text.replace('$', 'dollar ')
            text = text.replace('€', 'euro ')
            text = text.replace('£', 'pound ')
            
        # Removing special characters but keeping sentence structure
        text = re.sub(r'[^\w\s.]', '', text)
        
        text = ' '.join(text.split())
        
        return text.strip()
    

    def remove_stopwords(
        self,
        text: str, 
        region: str = 'international',
        keep_sentiment: bool = True
    ) -> str:
        """
        Now this removes stopwords while preserving the sentiment
        
        Args:
            text: Cleaned text
            region: 'indian' or 'international'
            keep_sentiment: Keep words that carry sentiment
            
            Returns:
                Text without stopwords
        """
        
        words = word_tokenize(text)
        
        if region == 'indian':
            stopwords_set = self.stop_words - self.sentiment_words
            stopwords_set.update(self.indian_neutral_terms)
        else:
            stopwords_set = self.stop_words - self.sentiment_words
            stopwords_set.update(self.intl_neutral_terms)
            
        if keep_sentiment:
            filtered_words = [
                word for word in words
                if word not in stopwords_set or word in self.sentiment_words
            ]
        else:
            filtered_words = [word for word in words if word not in stopwords_set]
            
        return ' '.join(filtered_words)
    
    def preprocess_article(self, article: Dict) -> Dict:
        """
        This preprocesses complete article with all text fields
        
        Returns article with 'processed_text' field added
        """
        
        region = article.get('region', 'international')
        
        # Get text fields
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        
        # Clean each text field
        clean_title = self.clean_text(title, region)
        clean_desc = self.clean_text(description, region)
        clean_content = self.clean_text(content[:1000], region)
        
        # Removing stopwords
        processed_title = self.remove_stopwords(clean_title, region)
        processed_desc = self.remove_stopwords(clean_desc, region)
        processed_content = self.remove_stopwords(clean_content, region)
        
        # Creating weighted combination
        processed_text = (
            f"{processed_title} {processed_title} {processed_title} "
            f"{processed_desc} {processed_desc} "
            f"{processed_content}"
        )
        
        # Adding the preprocessed text in the article
        article_with_processing = article.copy()
        article_with_processing['processed_text'] = processed_text
        article_with_processing['clean_title'] = clean_title
        article_with_processing['clean_description'] = clean_desc
        
        return article_with_processing
    
    def preprocess_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Preprocess multiple articles efficiently
        """
        
        processed = []
        
        for i, article in enumerate(articles):
            processed_article = self.preprocess_article(article)
            processed.append(processed_article)
            
            if (i + 1) % 10 == 0:
                print(f"🔧 Preprocessed {i + 1}/{len(articles)} articles...")
                
        return processed
    
    def _normalize_indian_numbers(self, text: str) -> str:
        """
        This will Convert Indian number system to international
        
        Examples:
        - "10 lakhs" → "1 million"
        - "5 crores" → "50 million"
        """
        
        text = re.sub(r'(\d+\.?\d*)\s*lakh', lambda m: f"{float(m.group(1))/10} million", text)
        text = re.sub(r'(\d+\.?\d*)\s*lac', lambda m: f"{float(m.group(1))/10} million", text)
        
        text = re.sub(r'(\d+\.?\d*)\s*crore', lambda m: f"{float(m.group(1))*10} million", text)
        
        return text
    
    
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    indian_article = {
        'title': 'BJP wins Delhi by 5 lakh votes!',
        'description': 'PM Modi congratulates party workers',
        'content': 'The government announced ₹10 crore relief package...',
        'region': 'indian'
    }
    
    processed = preprocessor.preprocess_article(indian_article)
    
    print("🇮🇳 Processed Indian article:")
    print(f"  Clean title: {processed['clean_title']}")
    print(f"  Processed: {processed['processed_text'][:100]}...")
        