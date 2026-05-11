import numpy as np
from typing import Dict, Any
from datetime import datetime

from src.logger import app_logger
from src.preprocess import preprocessor
from src.features import FeatureEngineer
from src.models.logistic_regression import LogisticRegressionModel
from src.config import config

class PredictionService:
    """Prediction service for production use"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load trained model and feature engineer"""
        try:
            # Load model
            self.model = LogisticRegressionModel()
            self.model.load(config.production.model_path)
            
            # Load feature engineer
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.load(config.production.vectorizer_path)
            
            app_logger.info("Prediction service artifacts loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load artifacts: {e}")
            raise
    
    def predict(self, message: str) -> Dict[str, Any]:
        """Make prediction for a single message"""
        
        # Preprocess
        cleaned_text = preprocessor.process(message)
        
        # Extract features
        features = self.feature_engineer.create_tfidf_features([cleaned_text])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        result = {
            'message': message,
            'cleaned_text': cleaned_text,
            'prediction': 'phishing' if prediction == 1 else 'safe',
            'confidence': float(probability[1] if prediction == 1 else probability[0]),
            'is_phishing': bool(prediction),
            'phishing_probability': float(probability[1]),
            'safe_probability': float(probability[0]),
            'timestamp': datetime.now().isoformat()
        }
        
        app_logger.info(f"Prediction made: {result['prediction']} with confidence {result['confidence']:.3f}")
        
        return result