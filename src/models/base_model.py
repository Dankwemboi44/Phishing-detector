from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.logger import app_logger

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_score': auc,
            'confusion_matrix': cm.tolist()
        }
        
        app_logger.info(f"Evaluation metrics for {self.name}: {metrics}")
        return metrics