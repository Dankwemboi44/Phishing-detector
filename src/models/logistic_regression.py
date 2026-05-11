import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import cross_val_score
import numpy as np
from typing import Dict, Optional
import joblib

from src.logger import app_logger
from src.config import config
from src.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression implementation"""
    
    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = SklearnLR(
            max_iter=1000,
            random_state=config.model.random_state,
            class_weight=config.model.class_weight,
            C=1.0,
            solver='liblinear'
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        
        app_logger.info(f"Training {self.name} on {X_train.shape[0]} samples")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        app_logger.info(f"Cross-validation F1 scores: {cv_scores}")
        app_logger.info(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Train final model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
            return metrics
        
        return {'cv_mean_f1': cv_scores.mean(), 'cv_std_f1': cv_scores.std()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save model to disk"""
        joblib.dump(self.model, path)
        app_logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        self.model = joblib.load(path)
        self.is_trained = True
        app_logger.info(f"Model loaded from {path}")