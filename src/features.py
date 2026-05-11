import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, Optional
import joblib
from scipy.sparse import hstack

from src.logger import app_logger
from src.config import config

class FeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.svd = None
        self.is_fitted = False
        
    def create_tfidf_features(self, texts: np.ndarray, max_features: int = None) -> np.ndarray:
        """Create TF-IDF features with proper handling for small datasets"""
        max_features = max_features or config.model.max_features
        
        if not self.is_fitted:
            # Parameters that work for both small and large datasets
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words=self._get_stop_words(),
                sublinear_tf=True,
                min_df=1,  # Critical: allows terms that appear in 1 document
                max_df=0.95,
                token_pattern=r'(?u)\b\w+\b'
            )
            
            features = self.tfidf_vectorizer.fit_transform(texts)
            self.is_fitted = True
            app_logger.info(f"Created TF-IDF features: {features.shape}")
            
        else:
            features = self.tfidf_vectorizer.transform(texts)
            
        return features
    
    def _get_stop_words(self) -> list:
        """Get stop words for Kiswahili and English"""
        return [
            'na', 'ya', 'wa', 'ni', 'kwa', 'cha', 'vya', 'za', 'la', 'ma',
            'a', 'an', 'and', 'the', 'of', 'to', 'for', 'in', 'on', 'at',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        ]
    
    def reduce_dimensions(self, features: np.ndarray, n_components: int = 100) -> np.ndarray:
        """Reduce dimensionality using SVD"""
        if features.shape[1] > n_components and features.shape[0] > n_components:
            self.svd = TruncatedSVD(n_components=min(n_components, features.shape[0] - 1), random_state=42)
            reduced = self.svd.fit_transform(features)
            app_logger.info(f"Reduced dimensions from {features.shape[1]} to {reduced.shape[1]}")
            return reduced
        return features
    
    def combine_features(self, tfidf_features: np.ndarray, 
                        additional_features: np.ndarray) -> np.ndarray:
        """Combine TF-IDF with additional features"""
        return hstack([tfidf_features, additional_features])
    
    def save(self, path: str = 'models/production/vectorizer.pkl'):
        """Save feature extractor"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'svd': self.svd
        }, path)
        app_logger.info(f"Feature extractor saved to {path}")
    
    def load(self, path: str = 'models/production/vectorizer.pkl'):
        """Load feature extractor"""
        data = joblib.load(path)
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.svd = data['svd']
        self.is_fitted = True
        app_logger.info(f"Feature extractor loaded from {path}")

def prepare_features(df: pd.DataFrame, 
                    text_column: str = 'processed_text',
                    engineer: Optional[FeatureEngineer] = None) -> Tuple[np.ndarray, np.ndarray, FeatureEngineer]:
    """Prepare all features for training"""
    
    engineer = engineer or FeatureEngineer()
    
    # Extract texts
    texts = df[text_column].fillna('').values
    
    # Create TF-IDF features
    tfidf_features = engineer.create_tfidf_features(texts)
    
    # Extract additional features if they exist
    additional_cols = ['url_count', 'phone_count', 'msg_length', 'exclamation_count', 'money_keyword_count']
    available_cols = [col for col in additional_cols if col in df.columns]
    
    if available_cols and len(available_cols) > 0:
        additional_features = df[available_cols].fillna(0).values
        features = engineer.combine_features(tfidf_features, additional_features)
    else:
        features = tfidf_features
    
    # Extract labels
    labels = df['label'].map({'phishing': 1, 'safe': 0}).values
    
    return features, labels, engineer