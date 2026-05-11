import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

from src.logger import app_logger
from src.preprocess import TextPreprocessor
from src.features import FeatureEngineer, prepare_features
from src.models.logistic_regression import LogisticRegressionModel
from src.config import config

class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.metrics = {}
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load raw data and apply preprocessing"""
        app_logger.info("Loading raw data...")
        
        raw_path = config.data.raw_data_path
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")
        
        df = pd.read_csv(raw_path)
        app_logger.info(f"Loaded {len(df)} messages")
        app_logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
        
        if 'message' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'message' and 'label' columns")
        
        df_processed = self.preprocessor.preprocess_dataframe(df, text_column='message')
        df_processed = self.preprocessor.add_features(df_processed)
        
        config.data.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(config.data.processed_data_path, index=False)
        app_logger.info(f"Saved processed data to {config.data.processed_data_path}")
        
        return df_processed
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare features for training"""
        app_logger.info("Preparing features...")
        
        if 'processed_text' not in df.columns:
            df = self.preprocessor.preprocess_dataframe(df, text_column='message')
        
        X, y, self.feature_engineer = prepare_features(df, engineer=self.feature_engineer)
        
        # Convert sparse matrix to dense for indexing (fixes the subscriptable error)
        # For 153 features, dense is fine. For larger, we'd keep sparse.
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        n_samples = X.shape[0]
        app_logger.info(f"Total samples: {n_samples}, Features: {X.shape[1]}")
        
        test_size = min(config.model.test_size, 0.3)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=config.model.random_state, stratify=y
        )
        
        # Validation split
        X_val, y_val = None, None
        if len(X_train) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.25,
                random_state=config.model.random_state, stratify=y_train
            )
            app_logger.info(f"Validation set: {len(X_val)}")
        else:
            app_logger.warning("Not enough data for validation split")
        
        app_logger.info(f"Training set: {len(X_train)}")
        app_logger.info(f"Test set: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the model"""
        app_logger.info("Training model...")
        
        self.model = LogisticRegressionModel()
        metrics = self.model.train(X_train, y_train, X_val, y_val)
        
        self.metrics = metrics
        app_logger.info(f"Training complete. Metrics: {metrics}")
        
        return self.model
    
    def save_artifacts(self, X_test, y_test):
        """Save model, vectorizer, and metrics"""
        
        model_path = config.production.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_path))
        
        vectorizer_path = config.production.vectorizer_path
        self.feature_engineer.save(str(vectorizer_path))
        
        test_metrics = self.model.evaluate(X_test, y_test)
        
        all_metrics = {
            'training_metrics': self.metrics,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_features': config.model.max_features,
                'test_size': config.model.test_size,
                'random_state': config.model.random_state
            }
        }
        
        metrics_path = Path('reports/metrics/training_metrics.json')
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        app_logger.info(f"Saved metrics to {metrics_path}")
        
        return test_metrics
    
    def run(self):
        """Run complete training pipeline"""
        app_logger.info("=" * 50)
        app_logger.info("Starting training pipeline")
        app_logger.info("=" * 50)
        
        try:
            df = self.load_and_preprocess_data()
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features(df)
            self.train_model(X_train, X_val, y_train, y_val)
            test_metrics = self.save_artifacts(X_test, y_test)
            
            app_logger.info("=" * 50)
            app_logger.info("Training pipeline completed successfully!")
            app_logger.info(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
            app_logger.info(f"Test F1-Score: {test_metrics['f1_score']:.3f}")
            app_logger.info("=" * 50)
            
            return test_metrics
            
        except Exception as e:
            app_logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()