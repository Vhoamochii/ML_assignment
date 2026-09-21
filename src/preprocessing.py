"""Text preprocessing and feature extraction"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Text cleaning and preprocessing"""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> list:
        """Tokenize text"""
        return word_tokenize(text)
    
    def remove_stopwords_tokens(self, tokens: list) -> list:
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: list) -> list:
        """Lemmatize tokens"""
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_tokens(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Apply preprocessing to entire dataframe"""
        df['cleaned_text'] = df[text_column].apply(self.preprocess)
        return df


class FeatureExtractor:
    """Extract features from text"""
    
    def __init__(self, method: str = 'tfidf', max_features: int = 5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        elif method == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()


if __name__ == '__main__':
    # Test preprocessing
    sample_text = "This is a SAMPLE text with URLs http://example.com and numbers 123!"
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
